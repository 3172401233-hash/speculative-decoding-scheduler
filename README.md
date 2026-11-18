# speculative-decoding-scheduler
Python 投机解码调度库，处理候选 token、验证批次、接受率和 KV 页预算

## 结构

- `specdec/scheduler.py` — 核心调度器：`Scheduler`（每轮决策）、`KVCachePool`（页式 KV 缓存）、`Request` / `Candidate` / `RoundRecord`
- `main.py` — 演示：构造多条请求，逐轮打印候选长度、验证接受数、放弃与 KV 页占用
- `tests/test_scheduler.py` — 单元测试（`python -m unittest`）

## 调度策略

每轮 `Scheduler.step()` 依次执行：

1. **完成到期验证**：验证延迟 `verification_latency_rounds` 届满的批次结算，被拒绝 token 的 KV 页**立即回收**（`pool.free` 同步发生，页可当场复用）。
2. **老化（aging）**：等待中的请求有效优先级 = `priority + aging_weight × wait_rounds`，保证短请求不会被长草稿无限抢占。
3. **起草候选**：候选长度 `k = 1 + round(acceptance_rate × (max_draft_tokens − 1))`，并随负载收缩——验证队列积压或 KV 压力超过 `kv_high_watermark` 时减半，且不超过空闲页的公平份额；`acceptance_rate < min_acceptance_rate` 时放弃投机（k=0，回退单 token 解码）。接受率用 EMA 在线更新。
4. **放弃（abandon）**：待验证队列超过 `max_pending_queue` 时，丢弃有效优先级最低的候选并立即回收其 KV 页；KV 分配失败时按优先级驱逐空闲请求的页（重算式抢占，同 vLLM），仍不够则本轮放弃。
5. **组成验证批次**：队列达到 `target_batch_size` 立即成批（忙时大批量）；空闲时最多等 `max_batch_wait_rounds` 轮攒批，避免小批次浪费吞吐。

模型版本切换 `set_model_version(v)`：旧版本起草的候选全部作废，KV 页立即回收，按新版本重新起草。

## 调度指标

`Scheduler.metrics()` 输出：

| 指标 | 含义 |
|---|---|
| `rounds` | 总调度轮数 |
| `total_drafted` / `total_accepted` | 起草 / 接受的 token 总数 |
| `acceptance_ratio` | 接受率 = accepted / drafted，衡量草稿质量 |
| `fallback_tokens` | 放弃投机后单 token 解码的 token 数 |
| `kv_peak_pages` | KV 页占用峰值（容量规划依据） |
| `worst_latency_rounds` | 所有请求中最大的完成轮数 − 到达轮数 |

每轮 `RoundRecord` 记录：本轮起草的 `(请求, 候选长度)`、验证的 `(请求, 候选长度, 接受数)`、被放弃的请求、KV 页使用/空闲。

## 最坏延迟分析

设优先级最大差为 `ΔP`，老化权重为 `a = aging_weight`，攒批等待上限 `W = max_batch_wait_rounds`，验证延迟 `L = verification_latency_rounds`。

- **饥饿上界**：被抢占请求每等待 1 轮有效优先级 +`a`，最多 `⌈ΔP / a⌉` 轮后成为全场最高有效优先级，必然进入下一批次。
- **单周期上界**：从被选中到验证完成最多 `W + L` 轮，因此一个投机周期不超过 `T = ⌈ΔP / a⌉ + W + L` 轮。
- **请求级上界**：每周期至少提交 1 个 token（全拒绝时回退保底），故生成 `N` 个 token 的请求最坏延迟为 `N × T` 轮；接受率 `ρ`、候选长度 `k` 时期望约为 `N / (ρk + 1) × (W + L)` 轮。
- **KV 驱逐**只增加重算开销，不影响上界：被驱逐请求保留已提交 token，老化持续累积，重新调度时按同一上界恢复。

## 运行

```bash
python main.py            # 逐轮输出候选与接受数
python -m unittest        # 运行全部测试
```

测试覆盖：高接受率（全部接受、零浪费）、全拒绝（KV 页立即回收）、低接受率回退、KV 页紧张（不超预算 + 抢占恢复）、并发请求（短请求抗饥饿）、模型版本切换（旧候选作废回收）。
