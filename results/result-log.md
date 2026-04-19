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
| 11 | 2026-04-19T22:08:04 | `ab32e50` | hellaswag | `baseline` | 30 | acc | 0.1000 |  |  | 11.0s | no |
| 12 | 2026-04-19T22:08:16 | `ab32e50` | hellaswag | `length_norm` | 30 | acc | 0.5667 | +0.4667 | [0.300000, 0.633333] | 8.6s | no |
| 13 | 2026-04-19T22:08:26 | `ab32e50` | hellaswag | `byte_norm` | 30 | acc | 0.3667 | +0.2667 | [0.133333, 0.433333] | 7.2s | no |
| 14 | 2026-04-19T22:08:39 | `ab32e50` | hellaswag | `clean_prompt` | 30 | acc | 0.0000 | -0.1000 | [-0.200000, 0.000000] | 9.8s | no |
| 15 | 2026-04-19T22:08:50 | `ab32e50` | hellaswag | `clean_length_norm` | 30 | acc | 0.6333 | +0.5333 | [0.366667, 0.700000] | 6.8s | no |
| 16 | 2026-04-19T22:09:07 | `ab32e50` | hellaswag | `fewshot_random_5` | 30 | acc | 1.0000 | +0.9000 | [0.800000, 1.000000] | 13.4s | no |
| 17 | 2026-04-19T22:09:32 | `ab32e50` | hellaswag | `fewshot_semantic_5` | 30 | acc | 1.0000 | +0.9000 | [0.800000, 1.000000] | 21.3s | no |
| 18 | 2026-04-19T22:48:39 | `ab32e50` | hellaswag | `baseline` | 30 | acc | 0.1000 |  |  | 6.8s | no |
| 19 | 2026-04-19T22:48:50 | `ab32e50` | hellaswag | `length_norm` | 30 | acc | 0.5667 | +0.4667 | [0.300000, 0.633333] | 6.8s | no |
| 20 | 2026-04-19T22:49:01 | `ab32e50` | hellaswag | `byte_norm` | 30 | acc | 0.3667 | +0.2667 | [0.133333, 0.433333] | 7.4s | no |
| 21 | 2026-04-19T22:49:12 | `ab32e50` | hellaswag | `clean_prompt` | 30 | acc | 0.0000 | -0.1000 | [-0.200000, 0.000000] | 7.9s | no |
| 22 | 2026-04-19T22:49:23 | `ab32e50` | hellaswag | `clean_length_norm` | 30 | acc | 0.6333 | +0.5333 | [0.366667, 0.700000] | 7.8s | no |
| 23 | 2026-04-19T22:49:33 | `ab32e50` | hellaswag | `fewshot_random_5` | 30 | acc | 1.0000 | +0.9000 | [0.800000, 1.000000] | 7.0s | no |
| 24 | 2026-04-19T22:49:48 | `ab32e50` | hellaswag | `fewshot_semantic_5` | 30 | acc | 1.0000 | +0.9000 | [0.800000, 1.000000] | 11.7s | no |
