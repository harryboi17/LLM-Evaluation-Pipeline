---
name: perf-test
description: Run the async load generator and summarize TTFT / TPOT / p95
allowed-tools:
  - read
  - exec
permissions:
  allow:
    - Exec(make perf)
    - Exec(uv run python -m perf.load_test)
---

Run the Part C load test and summarize key performance metrics.

1. Verify the server is up: `curl -sf http://127.0.0.1:8000/health`. If down, stop and tell
   the user to run `/start-vllm`.
2. Run `make perf` (writes `metrics.csv`).
3. Parse `metrics.csv` and print a markdown table with these columns per configuration:
   - concurrency
   - prompt_mix (short | long)
   - TTFT p50 / p95 (ms)
   - TPOT p50 / p95 (tokens/s)
   - end-to-end latency p50 / p95 / p99 (ms)
4. Highlight the best-case throughput and the configuration that caused it.
5. Note any anomalies: request failures, saturated GPU utilization (>95%), or wildly
   skewed tail latencies (p99 > 10× p50).

Do not re-run perf if it already produced a valid `metrics.csv` in this session. Offer to
re-run explicitly.
