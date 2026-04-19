# Part C — Performance & Scaling

Load generator + metric post-processor + plotting notebook. Measures TTFT, TPOT, and P50/P95/P99 latency under mixed short/long prompt concurrency. Optional `nvidia-smi` sampler for GPU utilisation.

## What's here

| File | Purpose |
|---|---|
| `load_test.py` | `asyncio`-based load generator. Fires `--num-requests` at `--concurrency` across a pool that mixes short and long prompts in a configurable ratio. Emits one row per request to `metrics.csv`. |
| `metrics.py` | Post-processor: groups rows by `(prompt_kind, mode)` and computes per-group percentiles, TPOT, error rate, and throughput. |
| `gpu_monitor.py` | Standalone `nvidia-smi` sampler. Writes one CSV row per GPU per sample; merges with `metrics.csv` by timestamp. Skips gracefully when `nvidia-smi` is unavailable. |
| `analysis.ipynb` | Plots latency KDE, TTFT boxplot, and TPOT boxplot from `metrics.csv`, with ≤200-word commentary at the bottom. |

## Quickstart

```bash
# Offline (mock backend, sub-second sanity check):
LLMEVAL_MOCK_BACKEND=true python -m perf.load_test \
    --num-requests 32 --concurrency 8 \
    --output /tmp/perf_smoke.csv

python -m perf.metrics --input /tmp/perf_smoke.csv

# Real run against a live vLLM server:
python -m perf.load_test \
    --num-requests 200 --concurrency 16 --mode stream \
    --output results/perf/metrics.csv

# In parallel, record GPU utilisation (if nvidia-smi present):
python -m perf.gpu_monitor --output results/perf/gpu.csv \
    --interval 0.5 --duration 180

# Summarise:
python -m perf.metrics --input results/perf/metrics.csv --summary results/perf/summary.csv

# Render the notebook to HTML:
make perf-analyze
```

## Metric definitions

* **Wall latency** — `ended_at - started_at` per request.
* **TTFT** (stream only) — monotonic time from POST to first non-empty delta.
* **TPOT** — `output_tokens / (wall_s - ttft_s)` when TTFT is known; `output_tokens / wall_s` otherwise. The decode-phase-only flavour is what most papers report.
* **Throughput** — `ok_requests / (max(ended_at) - min(started_at))` per group.

## CSV schema (`metrics.csv`)

```
idx,prompt_kind,prompt_chars,max_tokens,mode,wall_s,ttft_s,output_tokens,status,error,started_at,ended_at
```

* `prompt_kind` is one of `short` / `long`.
* `mode` is `generate` or `stream`.
* `status` is `ok` / `timeout` / `error`.
* `ttft_s` is empty for non-stream rows.

## Tests

`tests/perf/`:

- `test_load_test.py` — end-to-end against mock backend (both `generate` and `stream` modes), CSV round-trip, CLI output shape.
- `test_metrics.py` — percentile helper edge cases, TPOT math, `summarize()` groupings including rows with errors.
- `test_gpu_monitor.py` — `nvidia-smi` parsing, fallback path when binary is missing (no real `nvidia-smi` invoked).
