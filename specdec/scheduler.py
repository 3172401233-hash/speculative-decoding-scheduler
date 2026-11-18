"""Speculative decoding scheduler.

A draft model proposes several candidate tokens per request; a target model
verifies them in batches.  The scheduler decides, per round:

* how many draft tokens each request may speculate (candidate length),
* which pending candidates form the next verification batch,
* which candidates to abandon under load or KV-page pressure.

Rejected tokens release their KV pages immediately, and an aging mechanism
guarantees short requests are not starved by long drafts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SchedulerConfig:
    total_kv_pages: int = 64
    tokens_per_page: int = 16
    max_draft_tokens: int = 8
    min_acceptance_rate: float = 0.2
    target_batch_size: int = 4
    max_pending_queue: int = 8
    max_batch_wait_rounds: int = 1
    verification_latency_rounds: int = 1
    aging_weight: float = 1.0
    kv_high_watermark: float = 0.85
    ema_alpha: float = 0.5


class KVCachePool:
    """Page-granular KV cache.  Freed pages are reusable immediately."""

    def __init__(self, total_pages: int, tokens_per_page: int) -> None:
        if total_pages <= 0 or tokens_per_page <= 0:
            raise ValueError("pool dimensions must be positive")
        self.total_pages = total_pages
        self.tokens_per_page = tokens_per_page
        self.used_pages = 0
        self.peak_pages = 0

    def pages_for(self, tokens: int) -> int:
        return math.ceil(tokens / self.tokens_per_page) if tokens > 0 else 0

    @property
    def free_pages(self) -> int:
        return self.total_pages - self.used_pages

    @property
    def pressure(self) -> float:
        return self.used_pages / self.total_pages

    def allocate(self, pages: int) -> bool:
        if pages < 0 or self.used_pages + pages > self.total_pages:
            return False
        self.used_pages += pages
        self.peak_pages = max(self.peak_pages, self.used_pages)
        return True

    def free(self, pages: int) -> None:
        if pages < 0 or pages > self.used_pages:
            raise ValueError("invalid free")
        self.used_pages -= pages


@dataclass
class Request:
    request_id: str
    priority: int = 0
    prompt_tokens: int = 0
    remaining_tokens: int = 1
    acceptance_rate: float = 0.0
    model_version: str = ""
    arrival_round: int = 0

    committed_tokens: int = field(default=0, init=False)
    kv_pages: int = field(default=0, init=False)
    wait_rounds: int = field(default=0, init=False)
    done: bool = field(default=False, init=False)
    finished_round: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        self.committed_tokens = self.prompt_tokens


@dataclass
class Candidate:
    request: Request
    length: int
    drafted_round: int
    model_version: str


@dataclass
class RoundRecord:
    round_index: int
    drafted: list = field(default_factory=list)    # (req_id, candidate_len)
    verified: list = field(default_factory=list)   # (req_id, candidate_len, accepted)
    abandoned: list = field(default_factory=list)  # req_id
    kv_pages_used: int = 0
    kv_pages_free: int = 0


class Scheduler:
    def __init__(self, config: SchedulerConfig, verifier=None) -> None:
        self.config = config
        self.pool = KVCachePool(config.total_kv_pages, config.tokens_per_page)
        self.model_version = "v1"
        self.round_index = 0
        self.requests: list[Request] = []
        self.pending: list[Candidate] = []
        self.in_flight: list[tuple[int, list[Candidate]]] = []
        self._verifier = verifier or self._default_verifier
        self.total_drafted = 0
        self.total_accepted = 0
        self.total_fallback_tokens = 0

    # -- admission ---------------------------------------------------------

    def add_request(self, request: Request) -> None:
        request.arrival_round = self.round_index
        request.model_version = self.model_version
        pages = self.pool.pages_for(request.prompt_tokens)
        if not self.pool.allocate(pages):
            raise RuntimeError(f"cannot admit {request.request_id}: KV pool full")
        request.kv_pages = pages
        self.requests.append(request)

    def set_model_version(self, version: str) -> None:
        """Switch target model.  Candidates drafted for the old version are
        invalid: drop them and reclaim their KV pages immediately."""
        if version == self.model_version:
            return
        self.model_version = version
        for cand in list(self.pending):
            self._drop_candidate(cand)
        self.in_flight.clear()

    # -- policy ------------------------------------------------------------

    def _effective_priority(self, req: Request) -> float:
        return req.priority + self.config.aging_weight * req.wait_rounds

    def _candidate_length(self, req: Request, active: int) -> int:
        cfg = self.config
        if req.acceptance_rate < cfg.min_acceptance_rate:
            return 0  # abandon speculation, fall back to 1-token decoding
        k = 1 + round(req.acceptance_rate * (cfg.max_draft_tokens - 1))
        k = min(k, req.remaining_tokens - 1) if req.remaining_tokens > 1 else 0
        # shrink drafts when the verification queue is congested
        if len(self.pending) > cfg.target_batch_size:
            k = max(1, k // 2)
        # shrink drafts under KV pressure and cap by a fair share of free pages
        if self.pool.pressure > cfg.kv_high_watermark:
            k = max(1, k // 2)
        fair_share = (self.pool.free_pages * cfg.tokens_per_page) // max(1, active)
        return max(0, min(k, fair_share))

    # -- round driver ------------------------------------------------------

    def step(self) -> RoundRecord:
        cfg = self.config
        record = RoundRecord(round_index=self.round_index)

        # 1. complete verification batches whose latency has elapsed
        for ready_round, batch in list(self.in_flight):
            if ready_round <= self.round_index:
                self.in_flight.remove((ready_round, batch))
                for cand in batch:
                    self._verify(cand, record)

        # 2. age requests that are still waiting
        for req in self.requests:
            if not req.done and not self._has_candidate(req):
                req.wait_rounds += 1

        # 3. draft new candidates for requests that need them
        active = sum(1 for r in self.requests if not r.done)
        for req in self._schedulable_requests():
            length = self._candidate_length(req, active)
            # reserve pages for the draft plus the target model's bonus token
            extra = self.pool.pages_for(req.committed_tokens + length + 1) - req.kv_pages
            allocated = extra <= 0 or self.pool.allocate(extra)
            while not allocated and self._evict_victim(req):
                allocated = self.pool.allocate(extra)
            if not allocated:
                record.abandoned.append(req.request_id)
                continue
            req.kv_pages += extra
            req.model_version = self.model_version
            self.pending.append(Candidate(req, length, self.round_index, self.model_version))
            req.wait_rounds = 0
            record.drafted.append((req.request_id, length))
            self.total_drafted += length

        # 4. abandon overflow candidates when the queue is congested
        self.pending.sort(key=lambda c: -self._effective_priority(c.request))
        while len(self.pending) > cfg.max_pending_queue:
            cand = self.pending.pop()  # lowest effective priority
            record.abandoned.append(cand.request.request_id)
            self._drop_candidate(cand)

        # 5. form a verification batch
        batch = self._form_batch()
        if batch:
            for cand in batch:
                self.pending.remove(cand)
            ready = self.round_index + cfg.verification_latency_rounds
            self.in_flight.append((ready, batch))

        record.kv_pages_used = self.pool.used_pages
        record.kv_pages_free = self.pool.free_pages
        self.round_index += 1
        return record

    def run(self, max_rounds: int = 1000) -> list[RoundRecord]:
        records = []
        while not self.is_finished() and self.round_index < max_rounds:
            records.append(self.step())
        return records

    def is_finished(self) -> bool:
        return all(r.done for r in self.requests) and not self.pending and not self.in_flight

    # -- internals ---------------------------------------------------------

    def _schedulable_requests(self) -> list[Request]:
        reqs = [r for r in self.requests if not r.done and not self._has_candidate(r)]
        reqs.sort(key=lambda r: -self._effective_priority(r))
        return reqs

    def _has_candidate(self, req: Request) -> bool:
        if any(c.request is req for c in self.pending):
            return True
        return any(c.request is req for _, batch in self.in_flight for c in batch)

    def _form_batch(self) -> list[Candidate]:
        cfg = self.config
        if not self.pending:
            return []
        oldest_wait = self.round_index - min(c.drafted_round for c in self.pending)
        if len(self.pending) >= cfg.target_batch_size:
            return self.pending[: cfg.target_batch_size]
        if oldest_wait >= cfg.max_batch_wait_rounds:
            return list(self.pending)
        return []  # idle: accumulate one more round to avoid tiny batches

    def _verify(self, cand: Candidate, record: RoundRecord) -> None:
        req = cand.request
        if cand.model_version != self.model_version:
            self._drop_candidate(cand)
            return
        if cand.length == 0:  # speculation abandoned: plain single-token decode
            accepted, bonus = 0, 1
            self.total_fallback_tokens += 1
        else:
            accepted = min(self._verifier(req, cand.length), cand.length)
            bonus = 1
        self.total_accepted += accepted
        # EMA update of the historical acceptance rate
        alpha = self.config.ema_alpha
        observed = accepted / cand.length if cand.length else 0.0
        req.acceptance_rate = alpha * observed + (1 - alpha) * req.acceptance_rate

        new_committed = req.committed_tokens + accepted + bonus
        req.remaining_tokens -= accepted + bonus
        # rejected tokens release their KV pages immediately
        new_pages = self.pool.pages_for(new_committed)
        self.pool.free(req.kv_pages - new_pages)
        req.kv_pages = new_pages
        req.committed_tokens = new_committed

        if req.remaining_tokens <= 0:
            req.done = True
            req.finished_round = self.round_index
            self.pool.free(req.kv_pages)
            req.kv_pages = 0
        record.verified.append((req.request_id, cand.length, accepted))

    def _drop_candidate(self, cand: Candidate) -> None:
        if cand in self.pending:
            self.pending.remove(cand)
        req = cand.request
        keep = self.pool.pages_for(req.committed_tokens)
        self.pool.free(req.kv_pages - keep)
        req.kv_pages = keep

    def _evict_victim(self, requester: Request) -> bool:
        """Preempt the lowest-priority idle request: free all its KV pages.

        The victim keeps its committed token ids and recomputes the KV cache
        when rescheduled (recompute-on-eviction, as in vLLM)."""
        requester_prio = self._effective_priority(requester)
        victims = [r for r in self.requests
                   if r is not requester and not r.done and r.kv_pages > 0
                   and not self._has_candidate(r)
                   and self._effective_priority(r) <= requester_prio]
        if not victims:
            return False
        victim = min(victims, key=self._effective_priority)
        self.pool.free(victim.kv_pages)
        victim.kv_pages = 0
        return True

    @staticmethod
    def _default_verifier(req: Request, length: int) -> int:
        return round(req.acceptance_rate * length)

    # -- metrics -----------------------------------------------------------

    def metrics(self) -> dict:
        latencies = [r.finished_round - r.arrival_round for r in self.requests if r.done]
        return {
            "rounds": self.round_index,
            "total_drafted": self.total_drafted,
            "total_accepted": self.total_accepted,
            "acceptance_ratio": (self.total_accepted / self.total_drafted
                                 if self.total_drafted else 0.0),
            "fallback_tokens": self.total_fallback_tokens,
            "kv_peak_pages": self.pool.peak_pages,
            "worst_latency_rounds": max(latencies) if latencies else 0,
        }
