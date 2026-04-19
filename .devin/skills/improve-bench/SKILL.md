---
name: improve-bench
description: Drive a Part E benchmark improvement iteration
argument-hint: "<benchmark: hellaswag | mmlu | arc_challenge>"
allowed-tools:
  - read
  - edit
  - grep
  - glob
  - exec
permissions:
  allow:
    - Exec(bash improve/eval.sh)
    - Exec(uv run python -m improve.infer)
    - Exec(uv run python -m improve.optimize_prompt)
---

Run one iteration of the Part E inference-time improvement pipeline for benchmark `$1`.

Ground rules (from the assignment and `AGENTS.md`):
- Same model and vLLM configuration as Part B.
- **No finetuning / weight updates.** Only: prompt rewrites, few-shot selection, CoT,
  self-consistency, decoding tuning, regex normalization, retrieval from a static local
  corpus, logprob filtering.
- Must be statistically significant (p < 0.05) via paired bootstrap from `common/stats.py`.

Steps:
1. Run `uv run python -m improve.prepare_data --benchmark $1` to get aligned dev/test splits.
2. Run `uv run python -m improve.optimize_prompt --benchmark $1` to produce the best template
   + few-shot set under `improve/templates/`.
3. Run `bash improve/eval.sh $1` which:
   - Evaluates the baseline (same prompt as Part B)
   - Evaluates the improved pipeline
   - Writes `improve/results/$1/{baseline,improved}.json`
4. Compute statistics using `common.stats.paired_bootstrap`: print diff accuracy,
   95% CI, and p-value.
5. Append the ablation row (template / +few-shot / +CoT / +self-consistency) to
   `improve/report.md`.
6. Verify the target lift is met:
   - hellaswag: +3.0 acc
   - mmlu: +2.0 acc
   - arc_challenge: +2.5 acc

If the target is not met or p >= 0.05, stop and suggest the next lever to try.
Do NOT silently keep tweaking; report the current state and let the user decide.
