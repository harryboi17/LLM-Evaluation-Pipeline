# Part A — Serving

vLLM OpenAI-compatible server + async Python client.

## What's here

| File | Purpose |
|---|---|
| `serve.py` | Launches `vllm serve` with flags sourced from `common.config:Settings`. Pure `build_serve_command()` is unit-testable without invoking vLLM. |
| `client.py` | CLI wrapper around `common.vllm_client.VLLMClient` — one prompt in, completion on `stdout`, JSON summary on `stderr`. |
| `examples/demo.py` | Three sample generations: short greedy, long with stops, streaming. |
| `examples/concurrent_demo.py` | `asyncio.gather` over N requests vs. sequential baseline, reports a speedup ratio. |

## Quickstart

All examples honor the `LLMEVAL_MOCK_BACKEND=true` switch for offline dev; drop the env var to hit a real server.

```bash
# Launch vLLM (background); takes a minute on first run while weights download
make serve

# One-shot prompt (non-streaming):
uv run python -m serve.client "What is the capital of France?" --max-tokens 16

# Streaming:
uv run python -m serve.client "Write a haiku about systems." --stream --max-tokens 64

# Three-prompt demo (short greedy, long w/ stops, streaming):
make client-demo
# or: uv run python -m serve.examples.demo

# Concurrency / batching proof (8 parallel requests vs. 8 sequential):
uv run python -m serve.examples.concurrent_demo --concurrency 8
```

## Configuration

Every tunable is read from the environment with the `LLMEVAL_` prefix (see `.env.example`). Nothing is hardcoded; `build_serve_command` is the single place the vLLM argv is assembled.

| Env var | Default | Effect |
|---|---|---|
| `LLMEVAL_MODEL_NAME` | `meta-llama/Llama-3.2-1B-Instruct` | Served model id |
| `LLMEVAL_VLLM_HOST` / `_PORT` | `127.0.0.1` / `8000` | Bind address |
| `LLMEVAL_VLLM_API_KEY` | `EMPTY` | Bearer token |
| `LLMEVAL_VLLM_MAX_MODEL_LEN` | unset | Override context window |
| `LLMEVAL_VLLM_DTYPE` | `auto` | `float16` / `bfloat16` / `float32` |
| `LLMEVAL_VLLM_GPU_MEMORY_UTILIZATION` | `0.9` | Fraction of GPU RAM vLLM may use |
| `LLMEVAL_VLLM_TENSOR_PARALLEL_SIZE` | `1` | # GPUs for TP sharding |
| `LLMEVAL_VLLM_MAX_NUM_SEQS` | unset | Max concurrent sequences / step |
| `LLMEVAL_VLLM_TRUST_REMOTE_CODE` | `false` | Pass `--trust-remote-code` to vLLM |
| `LLMEVAL_MOCK_BACKEND` | `false` | If `true`, `VLLMClient` returns canned responses and `serve.serve` exits cleanly without launching vLLM |

Any of these can also be given as CLI flags to `python -m serve.serve` for one-off overrides (`--model`, `--port`, `--dtype`, ...).

## Tests

`tests/serve/`:

- `test_serve.py` — asserts `build_serve_command` produces the expected argv for various settings / overrides, and that `main()` refuses to spawn anything in mock mode or when `vllm` is missing.
- `test_client.py` — exercises the CLI in generate + stream modes against the mock backend.
- `test_examples.py` — smoke-tests both demos against the mock backend.
