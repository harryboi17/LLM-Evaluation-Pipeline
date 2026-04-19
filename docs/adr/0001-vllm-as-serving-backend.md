# ADR 0001 — vLLM as the serving backend

**Status:** Accepted
**Date:** 2026-04-19
**Deciders:** project scaffolding

## Context

The assignment requires a "high-throughput inference engine powering production-grade
deployments" exposed via an OpenAI-compatible `/generate` endpoint, with continuous
batching and paged attention. Candidate options:

1. **vLLM** (PagedAttention, continuous batching, OpenAI-compatible server built in)
2. **Text Generation Inference (TGI)** from HuggingFace
3. **llama.cpp server** (CPU-first, small-model-friendly)
4. **Roll our own** FastAPI + transformers (never appropriate here)

## Decision

Use **vLLM**.

- The assignment names it explicitly as the expected baseline.
- PagedAttention + continuous batching are defaults, so Part A's requirements are
  effectively "don't turn them off".
- vLLM ships an OpenAI-compatible server out of the box (`vllm serve`), which means
  Part A's REST layer is a flag rather than a new codebase.
- The same OpenAI interface makes Part B's lm-evaluation-harness wrapper simpler —
  we only implement the parts of the `LM` protocol that actually matter
  (`loglikelihood`, `generate_until`), calling the existing HTTP endpoint instead of
  owning a Python model object.

## Consequences

Positive:

- Zero custom serving code; we own only the client and the eval wrapper.
- Streaming, batching, and caching are inherited from vLLM.
- Mock fallback (`LLMEVAL_MOCK_BACKEND=true`) keeps the stack testable without a GPU.

Negative:

- Requires a CUDA GPU in practice. Mitigation: mock backend for dev; GPU for real runs.
- Binary install is large (~5 GB with CUDA wheels). Mitigation: optional extra
  (`pip install ".[serve]"`) so dev-only workflows skip it.
- Feature set is pinned to a specific vLLM minor version via `pyproject.toml`;
  behavior changes on upgrade will surface in the lock file.

## Alternatives considered

| Option | Why not |
|--------|---------|
| TGI | Functionally similar, but the assignment names vLLM specifically. Choosing TGI would mean justifying the deviation with no tangible win. |
| llama.cpp server | CPU-first; misses the point of "production-grade". Also lacks first-class continuous batching for the same throughput. |
| Own FastAPI server | Would require implementing continuous batching ourselves — weeks of work for a regression vs vLLM. |
