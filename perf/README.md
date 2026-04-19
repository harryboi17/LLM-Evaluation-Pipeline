# Part C — Performance & Scaling

Async load generator and analysis notebook.

- `load_test.py` — configurable concurrency, prompt mix (short/long), stop sequences.
- `metrics.py` — post-processing → `metrics.csv` with TTFT, TPOT, P50/P95/P99.
- `analysis.ipynb` — plots + 200-word commentary.
- `gpu_monitor.py` — optional `nvidia-smi` sampler.

Implemented in Part C.
