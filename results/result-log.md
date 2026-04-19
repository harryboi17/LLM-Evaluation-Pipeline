# Result Log

All evaluation runs in one place. Newest runs at the bottom.
Source of truth: [`result-log.csv`](result-log.csv).

| # | When | Commit | Task | Method | Limit | Metric | Value | Δ | 95% CI | Wall | Mock |
|---|------|--------|------|--------|-------|--------|-------|----|--------|------|------|
| 1 | 2026-04-19T17:26:14 | `035bba6` | hellaswag | `baseline` | 30 | acc | 0.0000 |  |  | 0.3s | yes |
| 2 | 2026-04-19T17:26:28 | `035bba6` | hellaswag | `length_norm` | 30 | acc | 0.2000 | +0.2000 | [0.066667, 0.366667] | 0.0s | yes |
| 3 | 2026-04-19T17:26:28 | `035bba6` | hellaswag | `byte_norm` | 30 | acc | 0.0000 | +0.0000 | [0.000000, 0.000000] | 0.0s | yes |
| 4 | 2026-04-19T17:26:29 | `035bba6` | hellaswag | `clean_prompt` | 30 | acc | 0.0000 | +0.0000 | [0.000000, 0.000000] | 0.4s | yes |
| 5 | 2026-04-19T17:26:29 | `035bba6` | hellaswag | `clean_length_norm` | 30 | acc | 0.2000 | +0.2000 | [0.066667, 0.366667] | 0.1s | yes |
| 6 | 2026-04-19T17:26:30 | `035bba6` | hellaswag | `fewshot_random_5` | 30 | acc | 0.2000 | +0.2000 | [0.066667, 0.366667] | 0.7s | yes |
