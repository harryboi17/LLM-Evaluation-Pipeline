# Part D — Guardrails & Determinism

Two reliability floors for the rest of the pipeline:

1. **Structural output validation** — every model completion we trust downstream must match either a JSON schema or a fast-path regex validator.
2. **Determinism verification** — the same prompt, same seed, same decoding params, run against a warm vLLM server must produce byte-identical completions. If not, Part E's paired-bootstrap comparisons lose their statistical footing.

## What's here

| File | Purpose |
|---|---|
| `validate.py` | `validate_output(text, schema)`, `validate_short_answer(text)`, `validate_batch(outputs)`, and `verify_determinism(prompt, n_runs)`. CLI entry at `python -m guardrails.validate {validate,determinism}`. |
| `schemas/short_answer.json` | JSON Schema for `custom_qa` outputs: 1-3 ASCII tokens, no trailing punctuation, no edge whitespace. |
| `schemas/mcq_letter.json` | JSON Schema for multiple-choice outputs: single uppercase `A-E` optionally with trailing period. |

## Quickstart

```bash
# Validate a JSONL of outputs against a schema:
python -m guardrails.validate validate outputs.jsonl --schema short_answer

# Or use the built-in short-answer regex validator (no schema file needed):
python -m guardrails.validate validate outputs.jsonl --short-answer

# Determinism check (requires a running vLLM or LLMEVAL_MOCK_BACKEND=true):
python -m guardrails.validate determinism \
    "What is the capital of France?" \
    --n-runs 5 --max-tokens 16
```

## Output format

Both subcommands emit a one-line JSON summary on `stdout`:

```json
{"valid": 47, "invalid": 3}
{"identical": true, "first_divergence_at": null}
```

And optionally a full report to `--report report.json` with per-record details.

Exit code is `0` when every output is valid / every completion matched, `1` otherwise.

## What was tested

Under `LLMEVAL_MOCK_BACKEND=true` (offline):

- `validate_short_answer` accepts well-formed outputs and rejects: empty, whitespace-wrapped, >3 tokens, non-ASCII, trailing `.?!`.
- `load_schema` + `validate_output` accepts valid MCQ letters (`A`, `C.`) and rejects empty / lowercase / multi-char.
- `verify_determinism` reports `identical=True` when the same prompt is sent five times with `temperature=0`, `top_p=1`, and a fixed seed — the mock backend returns a deterministic canned string, so this exercises the code path but doesn't stress the real server.

## Where non-determinism still leaks through (on real hardware)

None of these fail `verify_determinism` in practice with a single warmed-up server, but they can:

1. **Batch-shape-dependent fp16 reductions.** vLLM's GEMM kernels accumulate in fp16 (or bf16). Different batch shapes pick different kernel implementations, which change the reduction order, which can flip the last ULP of logits. At `temperature=0` the argmax is usually well-separated and this is invisible, but on a close 2-way tie between continuations the chosen token can differ run-to-run. Mitigation on the user side: either accept it (and report bootstrap CIs, like Part E does), or switch decode to fp32 (a large throughput cost).
2. **Server-side scheduler ordering.** When the server batches our prompt with unrelated concurrent requests, the forward-pass shape changes and (1) can bite. Measured as "solo" determinism in Part B / D tests — the prompt is the only in-flight request.
3. **CUDA kernel autotuning cold start.** On first use vLLM picks kernel variants and caches them. The very first request after server start can differ from subsequent requests. Warming up the server with a small throwaway prompt before benchmarking removes this.
4. **Sampler RNG state.** At `temperature > 0` the `seed` param covers everything deterministically across restarts. Our eval and improve paths use `temperature=0` so the seed is irrelevant, but Part E's self-consistency variants (if enabled with `temperature > 0`) must pin a seed explicitly; we do.

## Tests

`tests/guardrails/`:

- `test_validate.py` — schema loading, short-answer validator edges, batch validation stats.
- `test_determinism.py` — `verify_determinism` against the mock backend; first-divergence logic under injected drift (via monkeypatching the mock to return a different string on one run).
- `test_cli.py` — both subcommands end-to-end via `main()`, including exit codes.
