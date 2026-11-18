import unittest

from specdec import KVCachePool, Request, Scheduler, SchedulerConfig


def make_config(**overrides) -> SchedulerConfig:
    base = dict(total_kv_pages=64, tokens_per_page=16, max_draft_tokens=8,
                target_batch_size=2, max_pending_queue=4,
                verification_latency_rounds=1)
    base.update(overrides)
    return SchedulerConfig(**base)


class TestKVCachePool(unittest.TestCase):
    def test_allocate_free(self):
        pool = KVCachePool(total_pages=4, tokens_per_page=16)
        self.assertTrue(pool.allocate(3))
        self.assertFalse(pool.allocate(2))
        pool.free(3)
        self.assertEqual(pool.free_pages, 4)

    def test_pages_for_rounds_up(self):
        pool = KVCachePool(total_pages=4, tokens_per_page=16)
        self.assertEqual(pool.pages_for(0), 0)
        self.assertEqual(pool.pages_for(1), 1)
        self.assertEqual(pool.pages_for(17), 2)


class TestHighAcceptance(unittest.TestCase):
    def test_all_candidates_fully_accepted(self):
        sched = Scheduler(make_config())
        sched.add_request(Request("r", prompt_tokens=8, remaining_tokens=30,
                                  acceptance_rate=1.0))
        records = sched.run(max_rounds=100)
        self.assertTrue(sched.is_finished())
        # full acceptance -> longest drafts, nothing wasted
        self.assertEqual(records[0].drafted, [("r", 8)])
        self.assertEqual(sched.metrics()["acceptance_ratio"], 1.0)
        self.assertEqual(sched.total_fallback_tokens, 0)
        # 9 tokens committed per verification (8 accepted + 1 bonus)
        self.assertLessEqual(sched.round_index, 10)
        self.assertEqual(sched.pool.used_pages, 0)  # everything released


class TestAllRejection(unittest.TestCase):
    def test_rejected_pages_reclaimed_immediately(self):
        sched = Scheduler(make_config(min_acceptance_rate=0.0),
                          verifier=lambda req, k: 0)
        sched.add_request(Request("r", prompt_tokens=8, remaining_tokens=12,
                                  acceptance_rate=0.9))
        baseline = sched.pool.pages_for(8)
        records = sched.run(max_rounds=100)
        self.assertTrue(sched.is_finished())
        verified = [v for rec in records for v in rec.verified]
        self.assertTrue(verified)
        # every candidate fully rejected: only the bonus token commits
        self.assertTrue(all(accepted == 0 for _, _, accepted in verified))
        self.assertEqual(sched.total_accepted, 0)
        # KV pages never exceed committed tokens + one in-flight candidate
        max_committed_pages = sched.pool.pages_for(8 + len(verified))
        max_candidate_pages = sched.pool.pages_for(8)
        for rec in records:
            self.assertLessEqual(rec.kv_pages_used,
                                 max_committed_pages + max_candidate_pages)
        self.assertGreaterEqual(baseline, 1)

    def test_low_acceptance_abandons_speculation(self):
        sched = Scheduler(make_config())  # min_acceptance_rate = 0.2
        sched.add_request(Request("r", prompt_tokens=8, remaining_tokens=5,
                                  acceptance_rate=0.1))
        records = sched.run(max_rounds=50)
        self.assertTrue(sched.is_finished())
        self.assertTrue(all(k == 0 for rec in records for _, k in rec.drafted))
        self.assertEqual(sched.total_drafted, 0)
        self.assertGreater(sched.total_fallback_tokens, 0)


class TestKVPressure(unittest.TestCase):
    def test_never_exceeds_page_budget(self):
        cfg = make_config(total_kv_pages=4, tokens_per_page=16,
                          max_pending_queue=8)
        sched = Scheduler(cfg)
        for i in range(3):
            sched.add_request(Request(f"r{i}", prompt_tokens=16,
                                      remaining_tokens=20,
                                      acceptance_rate=0.9))
        records = sched.run(max_rounds=300)
        self.assertTrue(sched.is_finished())
        for rec in records:
            self.assertLessEqual(rec.kv_pages_used, cfg.total_kv_pages)
        # pressure forced shorter drafts than the max at least once
        self.assertTrue(any(k < cfg.max_draft_tokens
                            for rec in records for _, k in rec.drafted))

    def test_admission_rejected_when_pool_full(self):
        sched = Scheduler(make_config(total_kv_pages=1))
        sched.add_request(Request("a", prompt_tokens=16, remaining_tokens=4,
                                  acceptance_rate=0.9))
        with self.assertRaises(RuntimeError):
            sched.add_request(Request("b", prompt_tokens=16,
                                      remaining_tokens=4))


class TestConcurrency(unittest.TestCase):
    def test_short_request_not_starved(self):
        cfg = make_config(target_batch_size=1, max_pending_queue=2,
                          aging_weight=1.0)
        sched = Scheduler(cfg)
        sched.add_request(Request("long", priority=5, prompt_tokens=8,
                                  remaining_tokens=120, acceptance_rate=0.9))
        sched.add_request(Request("short", priority=0, prompt_tokens=8,
                                  remaining_tokens=3, acceptance_rate=0.9))
        sched.run(max_rounds=500)
        self.assertTrue(sched.is_finished())
        short = next(r for r in sched.requests if r.request_id == "short")
        # aging (weight 1) closes a priority gap of 5 in at most 5 rounds;
        # allow batching + verification latency slack
        self.assertLessEqual(short.finished_round, 20)

    def test_all_requests_finish(self):
        sched = Scheduler(make_config())
        for i in range(6):
            sched.add_request(Request(f"r{i}", priority=i % 3, prompt_tokens=8,
                                      remaining_tokens=10 + i,
                                      acceptance_rate=0.4 + 0.1 * (i % 5)))
        sched.run(max_rounds=500)
        self.assertTrue(sched.is_finished())
        self.assertEqual(sched.pool.used_pages, 0)


class TestModelVersionSwitch(unittest.TestCase):
    def test_stale_candidates_dropped_and_kv_reclaimed(self):
        sched = Scheduler(make_config())
        sched.add_request(Request("r", prompt_tokens=16, remaining_tokens=30,
                                  acceptance_rate=0.9))
        sched.step()  # drafts a candidate under v1
        self.assertTrue(sched.pending)
        committed_pages = sched.pool.pages_for(16)
        self.assertGreater(sched.pool.used_pages, committed_pages)

        sched.set_model_version("v2")
        self.assertEqual(sched.pending, [])
        self.assertEqual(sched.in_flight, [])
        # only committed tokens still hold pages
        self.assertEqual(sched.pool.used_pages, committed_pages)

        records = sched.run(max_rounds=100)
        self.assertTrue(sched.is_finished())
        # re-drafted and verified under the new version
        self.assertTrue(any(v for rec in records for v in rec.verified))


if __name__ == "__main__":
    unittest.main()
