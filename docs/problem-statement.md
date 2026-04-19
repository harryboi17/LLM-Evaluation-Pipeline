# LLM Systems & Evaluation Interview — Problem Breakdown

**Time budget:** 4 hours
**Tools required:** [vLLM](https://github.com/vllm-project/vllm), [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
**Model:** any open-weight model (Llama 3 / Mistral / Phi), kept fixed across all parts

This document restates and interprets the assignment so we have a single reference while building.

---

## TL;DR

Build a miniature production-style LLM evaluation pipeline in five stacked parts:

| Part | Focus | What's really being tested |
|------|-------|----------------------------|
| A | Serving | Can you stand up vLLM + a streaming client? |
| B | Evaluation | Can you plug a served model into lm-eval-harness? |
| C | Performance | Can you measure TTFT / TPOT / p95 under load? |
| D | Guardrails | Can you make runs deterministic and validated? |
| E | Benchmark improvement | Can you raise a benchmark score at inference time, with stats? |

Parts A and B are plumbing. **C, D, and E are where candidates are actually differentiated** — especially E, which has concrete numeric targets and a statistical-significance requirement.

---

## Part A — Serving

**Goal:** Spin up a vLLM inference server and a Python client that talks to it.

### Requirements
- vLLM server exposing an OpenAI-compatible `/generate` endpoint.
- **Continuous batching** and **paged attention** enabled (these are vLLM defaults — just don't disable them).
- Python client supporting:
  - Streaming token generation
  - Configurable decoding params: `max_tokens`, `temperature`, `top_p`, `stop`
- Validate multiple concurrent clients don't degrade performance (proves batching works).

### Deliverables
```
serve/
├── serve.py       # launches vLLM
└── client.py      # streaming + configurable client
```
- One-line startup: `make serve` or `python serve.py`
- Sample script that runs a few prompt generations

---

## Part B — Evaluation

**Goal:** Make the vLLM endpoint look like a first-class model to `lm-evaluation-harness`.

### Tasks
- Write a custom model wrapper implementing the harness's model interface but routing calls to our vLLM endpoint.
- Evaluate on:
  - **Two official tasks** (e.g., MMLU and HellaSwag)
  - **One small custom JSON-based benchmark** we design ourselves
- Add **prompt-level caching** so reruns are deterministic and cheap.

### Deliverables
```
eval_runner/
├── vllm_model.py   # harness-compatible wrapper
└── run_eval.py     # runner script
results/            # benchmark outputs + summary table
```

---

## Part C — Performance & Scaling

**Goal:** Measure the serving stack under load.

### Tasks
- Build a load generator firing concurrent requests with **short vs long prompts**.
- Log:
  - **TTFT** — time to first token
  - **TPOT** — tokens per second
  - **P50 / P95 / P99** end-to-end latency
  - GPU utilization (if GPU available)
- Sweep across batch sizes, caching on/off, stop-sequence variations.

### Deliverables
```
perf/
├── load_test.py
├── metrics.csv
└── analysis.ipynb   # plots + short commentary
```

---

## Part D — Guardrails & Determinism

**Goal:** Make runs reproducible and outputs validated.

### Tasks
- **Deterministic mode:** fix seeds, `temperature=0`, `top_p=1`.
- Verify identical prompts → identical responses.
- Add lightweight **regex or schema validation** for the custom task's outputs.
- Honestly document where nondeterminism still leaks through (CUDA kernels, batch ordering, fp16 reductions, etc.).

### Deliverables
```
guardrails/
└── validate.py
```
Plus a short README describing what was tested and where nondeterminism persists.

---

## Part E — Benchmark Improvement (main event)

**Goal:** Pick **one** of HellaSwag / MMLU / ARC-Challenge and raise its score **without touching model weights**.

### Hard constraints
- Same model, same vLLM config as earlier parts.
- **No finetuning, no parameter updates.**
- Improvement must be reproducible and **statistically significant (p < 0.05)**.

### Target lifts
| Benchmark | Target |
|-----------|--------|
| HellaSwag | **+3.0** accuracy |
| MMLU (subject group) | **+2.0** accuracy |
| ARC-Challenge | **+2.5** accuracy |

### Allowed levers (inference-time only)

**Prompt optimization**
- Template rewriting and instruction design
- Automatic few-shot selection (semantic similarity or clustering)
- Chain-of-thought / rationale-augmented prompts
- **Self-consistency** — sample k times, majority vote
- Prompt ensembling across phrasing variants

**Decoding optimization**
- `temperature`, `top_p`, `top_k` tuning
- Stop-sequence refinement
- Output normalization or regex-based mapping

**Retrieval augmentation**
- Deterministic retrieval from a local static corpus

**Confidence calibration**
- Filter / rescore using logprobs or entropy

### Deliverables
```
improve/
├── prepare_data.py
├── optimize_prompt.py
├── infer.py
├── eval.sh
└── report.md         # 400–700 words
```

`report.md` must contain:
- Baseline vs improved results **with 95% confidence intervals**
- **Ablation study** showing per-change impact
- **10+ before/after examples** with short analysis
- **Cost and latency trade-offs** (e.g., self-consistency with k=10 costs ~10× compute — quantify it)
- Exact seeds, decoding settings, configurations

---

## How the parts interlock

```
A (serve) ──► B (eval wrapper uses A) ──► E (uses A + B to improve a score)
                      │
                      ├──► C (loads A, measures perf)
                      └──► D (wraps B outputs, enforces determinism)
```

A and B are plumbing. C, D, E are the scored surface area.

---

## Suggested 4-hour time split

| Part | Time | Notes |
|------|------|-------|
| A | 30–40 min | vLLM is mostly config; get it running and move on |
| B | 45–60 min | Harness wrapper is the fiddly bit; caching is a thin decorator |
| C | 30–45 min | `asyncio` load gen + pandas plots |
| D | 20–30 min | Small but checks a real reliability box |
| E | **60–90 min** | The differentiator — hit the target lift, compute CIs, write the report |

**Main trap:** over-polishing A/B and running out of time for E. E is where the numeric targets live and where the report is graded.

---

## Submission

Email a GitHub repo or zip to `mle-interviewers@mercor.com` containing:

```
serve/
eval_runner/
perf/
guardrails/
improve/
results/
metrics.csv
Makefile
README.md
```

Plus a short final summary: the story of the best improvement and what we learned.

---

## Practical notes / gotchas

- **Harness integration:** the cleanest path is subclassing `lm_eval.api.model.LM` and implementing `loglikelihood`, `loglikelihood_rolling`, and `generate_until`. MMLU and ARC use loglikelihood; HellaSwag uses loglikelihood; generative custom task uses `generate_until`.
- **Logprobs from vLLM:** request `logprobs=1` and `prompt_logprobs` so the harness can score multiple-choice options by summing token logprobs over the completion.
- **Caching:** key on `(prompt, decoding_params)` → response. SQLite or a simple JSON-line file both work.
- **Determinism:** even with `temperature=0`, vLLM can produce tiny nondeterminism from batching order and fp16 reduction. Document it rather than pretending it doesn't exist.
- **Statistical significance for Part E:** use a **paired bootstrap** over per-example correctness (baseline vs improved on the same items). 10k resamples, report the 2.5 / 97.5 percentiles as the 95% CI for the difference in accuracy. p < 0.05 ⟺ 0 is outside that CI.
- **Highest-ROI improvement combo** for HellaSwag / ARC in practice: cleaner prompt template + self-consistency (k=5–10) + regex answer extraction. For MMLU: 5-shot with semantically-similar exemplars + CoT + majority vote.
