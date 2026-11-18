"""Demo: run the speculative decoding scheduler and print each round."""

from specdec import Request, Scheduler, SchedulerConfig


def main() -> None:
    config = SchedulerConfig(
        total_kv_pages=48,
        tokens_per_page=16,
        max_draft_tokens=8,
        target_batch_size=3,
        verification_latency_rounds=1,
    )
    scheduler = Scheduler(config)
    scheduler.add_request(Request("short-hi", priority=2, prompt_tokens=8,
                                  remaining_tokens=6, acceptance_rate=0.9))
    scheduler.add_request(Request("long-lo", priority=0, prompt_tokens=32,
                                  remaining_tokens=40, acceptance_rate=0.7))
    scheduler.add_request(Request("mid-mid", priority=1, prompt_tokens=16,
                                  remaining_tokens=20, acceptance_rate=0.5))
    scheduler.add_request(Request("cold", priority=1, prompt_tokens=8,
                                  remaining_tokens=10, acceptance_rate=0.1))

    records = scheduler.run(max_rounds=200)
    for rec in records:
        drafted = ", ".join(f"{rid}:k={k}" for rid, k in rec.drafted) or "-"
        verified = ", ".join(f"{rid}:{a}/{k}" for rid, k, a in rec.verified) or "-"
        dropped = ",".join(rec.abandoned) or "-"
        print(f"round {rec.round_index:3d} | draft [{drafted}] "
              f"| verify [{verified}] | abandon [{dropped}] "
              f"| kv {rec.kv_pages_used}/{config.total_kv_pages} pages")

    print("\nmetrics:")
    for key, value in scheduler.metrics().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
