# Result Log

All evaluation runs in one place. Newest runs at the bottom.
Source of truth: [`result-log.csv`](result-log.csv).

| # | When | Commit | Task | Method | Limit | Metric | Value | Δ | 95% CI | Wall | Mock |
|---|------|--------|------|--------|-------|--------|-------|----|--------|------|------|
| 1 | 2026-04-19T20:19:43 | `c1f8fa0` | hellaswag | `baseline` | 30 | acc | 0.0000 |  |  | 0.3s | yes |
| 2 | 2026-04-19T20:19:43 | `c1f8fa0` | hellaswag | `length_norm` | 30 | acc | 0.2000 | +0.2000 | [0.066667, 0.366667] | 0.0s | yes |
| 3 | 2026-04-19T20:19:43 | `c1f8fa0` | hellaswag | `byte_norm` | 30 | acc | 0.0000 | +0.0000 | [0.000000, 0.000000] | 0.0s | yes |
| 4 | 2026-04-19T20:19:44 | `c1f8fa0` | hellaswag | `clean_prompt` | 30 | acc | 0.0000 | +0.0000 | [0.000000, 0.000000] | 0.4s | yes |
| 5 | 2026-04-19T20:19:44 | `c1f8fa0` | hellaswag | `clean_length_norm` | 30 | acc | 0.2000 | +0.2000 | [0.066667, 0.366667] | 0.1s | yes |
| 6 | 2026-04-19T20:19:46 | `c1f8fa0` | hellaswag | `fewshot_random_5` | 30 | acc | 0.2000 | +0.2000 | [0.066667, 0.366667] | 0.8s | yes |
| 7 | 2026-04-19T20:19:47 | `c1f8fa0` | hellaswag | `fewshot_semantic_5` | 30 | acc | 0.2000 | +0.2000 | [0.066667, 0.366667] | 1.5s | yes |
| 8 | 2026-04-19T22:07:18 | `ab32e50` | mmlu_stem | `real_gpu_Qwen/Qwen2.5-1.5B-Instruct` | 200 | acc,none | 0.5392 |  |  | 645.4s | no |
| 9 | 2026-04-19T22:07:18 | `ab32e50` | hellaswag | `real_gpu_Qwen/Qwen2.5-1.5B-Instruct` | 200 | acc_norm,none | 0.6300 |  |  | 645.4s | no |
| 10 | 2026-04-19T22:07:18 | `ab32e50` | custom_qa | `real_gpu_Qwen/Qwen2.5-1.5B-Instruct` | 200 | exact_match,exact_match | 0.5600 |  |  | 645.4s | no |
| 11 | 2026-04-19T23:43:50 | `cd17c83` | mmlu_stem | `real_gpu_Qwen/Qwen2.5-1.5B-Instruct` | 200 | acc,none | 0.5381 |  |  | 575.6s | no |
| 12 | 2026-04-19T23:43:50 | `cd17c83` | hellaswag | `real_gpu_Qwen/Qwen2.5-1.5B-Instruct` | 200 | acc_norm,none | 0.6300 |  |  | 575.6s | no |
| 13 | 2026-04-19T23:43:50 | `cd17c83` | custom_qa | `real_gpu_Qwen/Qwen2.5-1.5B-Instruct` | 200 | exact_match,exact_match | 0.5600 |  |  | 575.6s | no |
| 14 | 2026-04-19T23:45:14 | `cd17c83` | hellaswag | `baseline` | 200 | acc | 0.5250 |  |  | 39.0s | no |
| 15 | 2026-04-19T23:45:26 | `cd17c83` | hellaswag | `length_norm` | 200 | acc | 0.6250 | +0.1000 | [0.025000, 0.175000] | 7.6s | no |
| 16 | 2026-04-19T23:45:38 | `cd17c83` | hellaswag | `byte_norm` | 200 | acc | 0.6450 | +0.1200 | [0.040000, 0.200000] | 7.4s | no |
| 17 | 2026-04-19T23:46:14 | `cd17c83` | hellaswag | `clean_prompt` | 200 | acc | 0.5100 | -0.0150 | [-0.055000, 0.025000] | 31.5s | no |
| 18 | 2026-04-19T23:46:27 | `cd17c83` | hellaswag | `clean_length_norm` | 200 | acc | 0.6100 | +0.0850 | [0.000000, 0.170000] | 8.5s | no |
| 19 | 2026-04-19T23:48:12 | `cd17c83` | hellaswag | `fewshot_random_5` | 200 | acc | 0.6350 | +0.1100 | [0.035000, 0.190000] | 101.4s | no |
