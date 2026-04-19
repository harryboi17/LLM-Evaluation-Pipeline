# Improvement Log

Chronological journal of every Part E change and its measured impact. One
entry per distinct run; each entry is self-contained so future readers
(and future-me) can reconstruct exactly what was tried, why, and what
happened.

Every row should also exist in `results/result-log.csv` with the matching
`run_id`. The result log is the source of truth for numbers; this file
explains the story behind them.

## Entries


## baseline — hellaswag
- **When:** 2026-04-19T20:19+00:00
- **Commit:** `c1f8fa0`
- **Task / limit / seed:** hellaswag / 30 / 42
- **Model:** `mock/test-1b` (mock_backend=True)
- **Accuracy:** 0.0000
- **Delta vs baseline:** n/a (baseline)
- **95% CI (paired bootstrap, 10k):** n/a (baseline)
- **Two-sided p-value:** n/a (baseline)
- **Significance:** n/a (baseline)
- **Wall time:** 0.3s
- **Notes:** baseline ablation

## length_norm — hellaswag
- **When:** 2026-04-19T20:19+00:00
- **Commit:** `c1f8fa0`
- **Task / limit / seed:** hellaswag / 30 / 42
- **Model:** `mock/test-1b` (mock_backend=True)
- **Accuracy:** 0.2000
- **Delta vs baseline:** +0.2000
- **95% CI (paired bootstrap, 10k):** [+0.0667, +0.3667]
- **Two-sided p-value:** 0.0010
- **Significance:** significantly better (95% CI > 0)
- **Wall time:** 0.0s
- **Notes:** length_norm ablation

## byte_norm — hellaswag
- **When:** 2026-04-19T20:19+00:00
- **Commit:** `c1f8fa0`
- **Task / limit / seed:** hellaswag / 30 / 42
- **Model:** `mock/test-1b` (mock_backend=True)
- **Accuracy:** 0.0000
- **Delta vs baseline:** +0.0000
- **95% CI (paired bootstrap, 10k):** [+0.0000, +0.0000]
- **Two-sided p-value:** 1.0000
- **Significance:** not significant (95% CI includes 0)
- **Wall time:** 0.0s
- **Notes:** byte_norm ablation

## clean_prompt — hellaswag
- **When:** 2026-04-19T20:19+00:00
- **Commit:** `c1f8fa0`
- **Task / limit / seed:** hellaswag / 30 / 42
- **Model:** `mock/test-1b` (mock_backend=True)
- **Accuracy:** 0.0000
- **Delta vs baseline:** +0.0000
- **95% CI (paired bootstrap, 10k):** [+0.0000, +0.0000]
- **Two-sided p-value:** 1.0000
- **Significance:** not significant (95% CI includes 0)
- **Wall time:** 0.4s
- **Notes:** clean_prompt ablation

## clean_length_norm — hellaswag
- **When:** 2026-04-19T20:19+00:00
- **Commit:** `c1f8fa0`
- **Task / limit / seed:** hellaswag / 30 / 42
- **Model:** `mock/test-1b` (mock_backend=True)
- **Accuracy:** 0.2000
- **Delta vs baseline:** +0.2000
- **95% CI (paired bootstrap, 10k):** [+0.0667, +0.3667]
- **Two-sided p-value:** 0.0010
- **Significance:** significantly better (95% CI > 0)
- **Wall time:** 0.1s
- **Notes:** clean_length_norm ablation

## fewshot_random_5 — hellaswag
- **When:** 2026-04-19T20:19+00:00
- **Commit:** `c1f8fa0`
- **Task / limit / seed:** hellaswag / 30 / 42
- **Model:** `mock/test-1b` (mock_backend=True)
- **Accuracy:** 0.2000
- **Delta vs baseline:** +0.2000
- **95% CI (paired bootstrap, 10k):** [+0.0667, +0.3667]
- **Two-sided p-value:** 0.0010
- **Significance:** significantly better (95% CI > 0)
- **Wall time:** 0.8s
- **Notes:** fewshot_random_5 ablation

## fewshot_semantic_5 — hellaswag
- **When:** 2026-04-19T20:19+00:00
- **Commit:** `c1f8fa0`
- **Task / limit / seed:** hellaswag / 30 / 42
- **Model:** `mock/test-1b` (mock_backend=True)
- **Accuracy:** 0.2000
- **Delta vs baseline:** +0.2000
- **95% CI (paired bootstrap, 10k):** [+0.0667, +0.3667]
- **Two-sided p-value:** 0.0010
- **Significance:** significantly better (95% CI > 0)
- **Wall time:** 1.5s
- **Notes:** fewshot_semantic_5 ablation

## baseline — hellaswag
- **When:** 2026-04-19T20:20+00:00
- **Commit:** `c1f8fa0`
- **Task / limit / seed:** hellaswag / 3 / 42
- **Model:** `mock/test-model` (mock_backend=True)
- **Accuracy:** 0.3333
- **Delta vs baseline:** n/a (baseline)
- **95% CI (paired bootstrap, 10k):** n/a (baseline)
- **Two-sided p-value:** n/a (baseline)
- **Significance:** n/a (baseline)
- **Wall time:** 0.3s
- **Notes:** (none)

## length_norm — hellaswag
- **When:** 2026-04-19T20:20+00:00
- **Commit:** `c1f8fa0`
- **Task / limit / seed:** hellaswag / 3 / 42
- **Model:** `mock/test-model` (mock_backend=True)
- **Accuracy:** 0.3333
- **Delta vs baseline:** +0.0000
- **95% CI (paired bootstrap, 10k):** [-1.0000, +1.0000]
- **Two-sided p-value:** 1.0000
- **Significance:** not significant (95% CI includes 0)
- **Wall time:** 0.0s
- **Notes:** length-normalised scoring

## baseline — hellaswag
- **When:** 2026-04-19T20:20+00:00
- **Commit:** `c1f8fa0`
- **Task / limit / seed:** hellaswag / 3 / 42
- **Model:** `mock/test-model` (mock_backend=True)
- **Accuracy:** 0.3333
- **Delta vs baseline:** n/a (baseline)
- **95% CI (paired bootstrap, 10k):** n/a (baseline)
- **Two-sided p-value:** n/a (baseline)
- **Significance:** n/a (baseline)
- **Wall time:** 0.1s
- **Notes:** (none)

## length_norm — hellaswag
- **When:** 2026-04-19T20:20+00:00
- **Commit:** `c1f8fa0`
- **Task / limit / seed:** hellaswag / 3 / 42
- **Model:** `mock/test-model` (mock_backend=True)
- **Accuracy:** 0.3333
- **Delta vs baseline:** +0.0000
- **95% CI (paired bootstrap, 10k):** [-1.0000, +1.0000]
- **Two-sided p-value:** 1.0000
- **Significance:** not significant (95% CI includes 0)
- **Wall time:** 0.0s
- **Notes:** length-normalised scoring

## baseline — hellaswag
- **When:** 2026-04-19T20:21+00:00
- **Commit:** `c1f8fa0`
- **Task / limit / seed:** hellaswag / 6 / 42
- **Model:** `mock/smoke` (mock_backend=True)
- **Accuracy:** 0.3333
- **Delta vs baseline:** n/a (baseline)
- **95% CI (paired bootstrap, 10k):** n/a (baseline)
- **Two-sided p-value:** n/a (baseline)
- **Significance:** n/a (baseline)
- **Wall time:** 0.2s
- **Notes:** (none)

## length_norm — hellaswag
- **When:** 2026-04-19T20:21+00:00
- **Commit:** `c1f8fa0`
- **Task / limit / seed:** hellaswag / 6 / 42
- **Model:** `mock/smoke` (mock_backend=True)
- **Accuracy:** 0.3333
- **Delta vs baseline:** +0.0000
- **95% CI (paired bootstrap, 10k):** [+0.0000, +0.0000]
- **Two-sided p-value:** 1.0000
- **Significance:** not significant (95% CI includes 0)
- **Wall time:** 0.0s
- **Notes:** (none)

## baseline — hellaswag
- **When:** 2026-04-19T20:40+00:00
- **Commit:** `6e99ba7`
- **Task / limit / seed:** hellaswag / 3 / 42
- **Model:** `mock/test-model` (mock_backend=True)
- **Accuracy:** 0.3333
- **Delta vs baseline:** n/a (baseline)
- **95% CI (paired bootstrap, 10k):** n/a (baseline)
- **Two-sided p-value:** n/a (baseline)
- **Significance:** n/a (baseline)
- **Wall time:** 0.1s
- **Notes:** (none)

## length_norm — hellaswag
- **When:** 2026-04-19T20:40+00:00
- **Commit:** `6e99ba7`
- **Task / limit / seed:** hellaswag / 3 / 42
- **Model:** `mock/test-model` (mock_backend=True)
- **Accuracy:** 0.3333
- **Delta vs baseline:** +0.0000
- **95% CI (paired bootstrap, 10k):** [-1.0000, +1.0000]
- **Two-sided p-value:** 1.0000
- **Significance:** not significant (95% CI includes 0)
- **Wall time:** 0.0s
- **Notes:** length-normalised scoring

## baseline — hellaswag
- **When:** 2026-04-19T20:42+00:00
- **Commit:** `6e99ba7`
- **Task / limit / seed:** hellaswag / 3 / 42
- **Model:** `mock/test-model` (mock_backend=True)
- **Accuracy:** 0.3333
- **Delta vs baseline:** n/a (baseline)
- **95% CI (paired bootstrap, 10k):** n/a (baseline)
- **Two-sided p-value:** n/a (baseline)
- **Significance:** n/a (baseline)
- **Wall time:** 0.1s
- **Notes:** (none)

## length_norm — hellaswag
- **When:** 2026-04-19T20:42+00:00
- **Commit:** `6e99ba7`
- **Task / limit / seed:** hellaswag / 3 / 42
- **Model:** `mock/test-model` (mock_backend=True)
- **Accuracy:** 0.3333
- **Delta vs baseline:** +0.0000
- **95% CI (paired bootstrap, 10k):** [-1.0000, +1.0000]
- **Two-sided p-value:** 1.0000
- **Significance:** not significant (95% CI includes 0)
- **Wall time:** 0.0s
- **Notes:** length-normalised scoring

## baseline — hellaswag
- **When:** 2026-04-19T20:42+00:00
- **Commit:** `6e99ba7`
- **Task / limit / seed:** hellaswag / 3 / 42
- **Model:** `mock/test-model` (mock_backend=True)
- **Accuracy:** 0.3333
- **Delta vs baseline:** n/a (baseline)
- **95% CI (paired bootstrap, 10k):** n/a (baseline)
- **Two-sided p-value:** n/a (baseline)
- **Significance:** n/a (baseline)
- **Wall time:** 0.1s
- **Notes:** (none)

## length_norm — hellaswag
- **When:** 2026-04-19T20:42+00:00
- **Commit:** `6e99ba7`
- **Task / limit / seed:** hellaswag / 3 / 42
- **Model:** `mock/test-model` (mock_backend=True)
- **Accuracy:** 0.3333
- **Delta vs baseline:** +0.0000
- **95% CI (paired bootstrap, 10k):** [-1.0000, +1.0000]
- **Two-sided p-value:** 1.0000
- **Significance:** not significant (95% CI includes 0)
- **Wall time:** 0.0s
- **Notes:** length-normalised scoring

## baseline — hellaswag
- **When:** 2026-04-19T20:43+00:00
- **Commit:** `6e99ba7`
- **Task / limit / seed:** hellaswag / 6 / 42
- **Model:** `mock/smoke` (mock_backend=True)
- **Accuracy:** 0.3333
- **Delta vs baseline:** n/a (baseline)
- **95% CI (paired bootstrap, 10k):** n/a (baseline)
- **Two-sided p-value:** n/a (baseline)
- **Significance:** n/a (baseline)
- **Wall time:** 0.3s
- **Notes:** (none)

## length_norm — hellaswag
- **When:** 2026-04-19T20:43+00:00
- **Commit:** `6e99ba7`
- **Task / limit / seed:** hellaswag / 6 / 42
- **Model:** `mock/smoke` (mock_backend=True)
- **Accuracy:** 0.3333
- **Delta vs baseline:** +0.0000
- **95% CI (paired bootstrap, 10k):** [+0.0000, +0.0000]
- **Two-sided p-value:** 1.0000
- **Significance:** not significant (95% CI includes 0)
- **Wall time:** 0.1s
- **Notes:** (none)

## baseline — hellaswag
- **When:** 2026-04-19T21:04+00:00
- **Commit:** `827b481`
- **Task / limit / seed:** hellaswag / 3 / 42
- **Model:** `mock/test-model` (mock_backend=True)
- **Accuracy:** 0.3333
- **Delta vs baseline:** n/a (baseline)
- **95% CI (paired bootstrap, 10k):** n/a (baseline)
- **Two-sided p-value:** n/a (baseline)
- **Significance:** n/a (baseline)
- **Wall time:** 0.1s
- **Notes:** (none)

## length_norm — hellaswag
- **When:** 2026-04-19T21:04+00:00
- **Commit:** `827b481`
- **Task / limit / seed:** hellaswag / 3 / 42
- **Model:** `mock/test-model` (mock_backend=True)
- **Accuracy:** 0.3333
- **Delta vs baseline:** +0.0000
- **95% CI (paired bootstrap, 10k):** [-1.0000, +1.0000]
- **Two-sided p-value:** 1.0000
- **Significance:** not significant (95% CI includes 0)
- **Wall time:** 0.0s
- **Notes:** length-normalised scoring

## baseline — hellaswag
- **When:** 2026-04-19T21:17+00:00
- **Commit:** `0bf8fa8`
- **Task / limit / seed:** hellaswag / 3 / 42
- **Model:** `mock/test-model` (mock_backend=True)
- **Accuracy:** 0.3333
- **Delta vs baseline:** n/a (baseline)
- **95% CI (paired bootstrap, 10k):** n/a (baseline)
- **Two-sided p-value:** n/a (baseline)
- **Significance:** n/a (baseline)
- **Wall time:** 0.1s
- **Notes:** (none)

## length_norm — hellaswag
- **When:** 2026-04-19T21:17+00:00
- **Commit:** `0bf8fa8`
- **Task / limit / seed:** hellaswag / 3 / 42
- **Model:** `mock/test-model` (mock_backend=True)
- **Accuracy:** 0.3333
- **Delta vs baseline:** +0.0000
- **95% CI (paired bootstrap, 10k):** [-1.0000, +1.0000]
- **Two-sided p-value:** 1.0000
- **Significance:** not significant (95% CI includes 0)
- **Wall time:** 0.0s
- **Notes:** length-normalised scoring
<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======

## baseline — hellaswag
- **When:** 2026-04-19T22:05+00:00
- **Commit:** `ab32e50`
- **Task / limit / seed:** hellaswag / 3 / 42
- **Model:** `mock/test-model` (mock_backend=True)
- **Accuracy:** 0.3333
>>>>>>> Stashed changes
- **Delta vs baseline:** n/a (baseline)
- **95% CI (paired bootstrap, 10k):** n/a (baseline)
- **Two-sided p-value:** n/a (baseline)
- **Significance:** n/a (baseline)
<<<<<<< Updated upstream
- **Wall time:** 11.0s
- **Notes:** ablation run via improve/eval.sh

## length_norm — hellaswag
- **When:** 2026-04-19T22:05+00:00
- **Commit:** `ab32e50`
- **Task / limit / seed:** hellaswag / 3 / 42
- **Model:** `mock/test-model` (mock_backend=True)
- **Accuracy:** 0.3333
- **Delta vs baseline:** +0.0000
- **95% CI (paired bootstrap, 10k):** [-1.0000, +1.0000]
- **Two-sided p-value:** 1.0000
- **Significance:** not significant (95% CI includes 0)
- **Wall time:** 0.0s
- **Notes:** length-normalised scoring
>>>>>>> Stashed changes

## baseline — hellaswag
- **When:** 2026-04-19T22:45+00:00
- **Commit:** `2409c7d`
- **Task / limit / seed:** hellaswag / 3 / 42
- **Model:** `mock/test-model` (mock_backend=True)
- **Accuracy:** 0.3333
- **Delta vs baseline:** n/a (baseline)
- **95% CI (paired bootstrap, 10k):** n/a (baseline)
- **Two-sided p-value:** n/a (baseline)
- **Significance:** n/a (baseline)
- **Wall time:** 0.1s
- **Notes:** (none)

## length_norm — hellaswag
- **When:** 2026-04-19T22:45+00:00
- **Commit:** `2409c7d`
- **Task / limit / seed:** hellaswag / 3 / 42
- **Model:** `mock/test-model` (mock_backend=True)
- **Accuracy:** 0.3333
- **Delta vs baseline:** +0.0000
- **95% CI (paired bootstrap, 10k):** [-1.0000, +1.0000]
- **Two-sided p-value:** 1.0000
- **Significance:** not significant (95% CI includes 0)
- **Wall time:** 0.0s
- **Notes:** length-normalised scoring
