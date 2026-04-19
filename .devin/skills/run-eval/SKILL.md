---
name: run-eval
description: Run the full evaluation suite (MMLU + HellaSwag + custom) and summarize
subagent: true
allowed-tools:
  - read
  - grep
  - glob
  - exec
permissions:
  allow:
    - Exec(make eval)
    - Exec(uv run python -m eval_runner.run_eval)
---

Run the full lm-evaluation-harness suite against the vLLM server.

1. Verify the server is up: `curl -sf http://127.0.0.1:8000/health` (or the host/port from `.env`).
   If down, stop and tell the user to run `/start-vllm` first — do NOT try to start it yourself.
2. Run `make eval` (equivalent to `python -m eval_runner.run_eval --task mmlu,hellaswag,custom`).
3. Summarize `results/summary.md`: pull the accuracy per task, note any failures, and print a markdown table.
4. Verify `results/<task>.json` was created for each task; warn if any are missing.
5. Report total wall-clock time and cache hit rate (from the run log).

If `make eval` fails, do NOT retry. Summarize the failure mode (harness wrapper error, timeout, OOM, task config issue) and where in the log it first appeared.
