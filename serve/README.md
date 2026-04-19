# Part A — Serving

vLLM OpenAI-compatible server + async Python client.

- `serve.py` — launches `vllm serve` with flags from `common.config:Settings`.
- `client.py` — re-exports `common.vllm_client.VLLMClient` + a CLI demo.
- `examples/demo.py` — three sample generations (short, long, with stops).
- `examples/concurrent_demo.py` — `asyncio.gather` over N clients to prove batching.

Implemented in Part A (after Phase 0 is committed). This README is a placeholder.
