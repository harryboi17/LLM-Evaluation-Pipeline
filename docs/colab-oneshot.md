# Colab one-cell runbook for Part E

The Colab notebook in `notebooks/colab_run.ipynb` works when run end-to-end
from a fresh session, but it's easy to land in a state where a previous
session's stale clone + stale data files silently produce synthetic numbers
across reruns. This is a **single self-contained Colab cell** that is
idempotent and refuses to produce bad data.

## Prerequisites (one-time)

1. Your fork of this repo is pushed to GitHub (public, or Colab-accessible).
2. The HEAD of your main branch contains commit `bacff58` or later (the one
   that adds "Always re-prepare" to `improve/eval.sh`). Verify with
   `git log --oneline | grep bacff58` locally.

## The cell

Paste this into a **fresh Colab notebook** on a T4 runtime (*Runtime →
Change runtime type → T4 GPU*). Edit `REPO_URL` + `HF_TOKEN` at the top,
then *Runtime → Run all*. It's the only cell you need — it handles
everything.

```python
# === CONFIG (edit these two) =============================================
REPO_URL = "https://github.com/harryboi17/LLMEvalSystem.git"
HF_TOKEN = ""   # leave empty for un-gated Qwen; set if you switch MODEL to Llama

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N_EVAL = 200     # real Part E eval size (200 is a good balance of CI width vs wall time)
N_POOL = 500     # few-shot pool size (disjoint from eval subset)

# === BOOT (everything below is automatic) ================================
import os, shutil, subprocess, sys, time, urllib.request, urllib.error, json, csv
from pathlib import Path

# 1. Nuke any previous clone. This is the safeguard against "stale workdir"
#    that we hit twice: a session that was connected before the latest fix
#    keeps re-running the old code.
os.chdir("/content")
shutil.rmtree("LLMEvalSystem", ignore_errors=True)
shutil.rmtree("/root/.cache/pip", ignore_errors=True)  # belt + braces

# 2. Fresh clone.
subprocess.check_call(["git", "clone", "--depth=50", REPO_URL, "LLMEvalSystem"])
os.chdir("/content/LLMEvalSystem")
print("HEAD:", subprocess.check_output(["git", "log", "--oneline", "-1"]).decode().strip())

# 3. Hard requirement: eval.sh must contain the "Always re-prepare" fix.
eval_sh = Path("improve/eval.sh").read_text()
if "Always re-prepare" not in eval_sh:
    raise RuntimeError(
        "improve/eval.sh is missing the 'Always re-prepare' fix. "
        "This means REPO_URL is pointing at a pre-bacff58 fork. Push your "
        "latest main to the URL you set and restart this cell."
    )

# 4. Sync deps with every extra. Persistent warm-cache keyed on uv.lock.
subprocess.check_call(["curl", "-LsSf", "https://astral.sh/uv/install.sh", "-o", "/tmp/uv.sh"])
subprocess.check_call(["bash", "/tmp/uv.sh"])
os.environ["PATH"] = f"{os.path.expanduser('~/.local/bin')}:{os.environ['PATH']}"
subprocess.check_call(["uv", "sync", "--extra", "serve", "--extra", "eval", "--extra", "perf", "--extra", "improve"])

# 5. HF auth (only if token set — Qwen is ungated).
if HF_TOKEN:
    subprocess.check_call(["uv", "run", "huggingface-cli", "login", "--token", HF_TOKEN, "--add-to-git-credential"])

# 6. Settings via env. LLMEVAL_MOCK_BACKEND=false is the most important one.
os.environ.update({
    "LLMEVAL_MOCK_BACKEND": "false",
    "LLMEVAL_MODEL_NAME": MODEL,
    "LLMEVAL_VLLM_HOST": "127.0.0.1",
    "LLMEVAL_VLLM_PORT": "8000",
    "LLMEVAL_VLLM_API_KEY": "EMPTY",
    "LLMEVAL_VLLM_DTYPE": "float16",            # T4: Turing, no bf16
    "LLMEVAL_VLLM_MAX_MODEL_LEN": "2048",
    "LLMEVAL_VLLM_GPU_MEMORY_UTILIZATION": "0.85",
    "LLMEVAL_VLLM_TIMEOUT_S": "300",
    "LLMEVAL_SEED": "42",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
})

# 7. Boot vLLM in background, wait until ready.
shutil.rmtree("results/improve", ignore_errors=True)  # paranoia; there shouldn't be one in a fresh clone
serve_log = open("vllm_serve.log", "w")
serve_proc = subprocess.Popen(["uv", "run", "python", "-m", "serve.serve"], stdout=serve_log, stderr=subprocess.STDOUT)
print(f"vllm pid={serve_proc.pid}")

deadline = time.monotonic() + 900
while time.monotonic() < deadline:
    if serve_proc.poll() is not None:
        subprocess.check_call(["tail", "-n", "60", "vllm_serve.log"])
        raise RuntimeError("vllm died early -- see log above")
    try:
        urllib.request.urlopen(
            urllib.request.Request("http://127.0.0.1:8000/v1/models", headers={"Authorization": "Bearer EMPTY"}),
            timeout=3,
        )
        break
    except Exception:
        time.sleep(5)
print("vllm ready")

# 8. One-shot sanity completion.
subprocess.check_call(["uv", "run", "python", "-m", "serve.client",
                       "What is the capital of France?", "--max-tokens", "16"])

# 9. Part B + C (lm-eval + load test + GPU monitor).
subprocess.check_call(["uv", "run", "python", "-m", "eval_runner.run_eval",
                       "--task", "mmlu_stem,hellaswag,custom_qa",
                       "--limit", str(N_EVAL), "--max-concurrency", "16",
                       "--method", f"real_gpu_{MODEL}", "--notes", "one-shot Colab run"])

gpu_proc = subprocess.Popen(
    ["uv", "run", "python", "-m", "perf.gpu_monitor",
     "--output", "results/perf/gpu.csv", "--interval", "1", "--duration", "180"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
subprocess.check_call(["uv", "run", "python", "-m", "perf.load_test",
                       "--num-requests", "200", "--concurrency", "16", "--mode", "stream",
                       "--output", "results/perf/metrics.csv"])
subprocess.check_call(["uv", "run", "python", "-m", "perf.metrics",
                       "--input", "results/perf/metrics.csv",
                       "--summary", "results/perf/summary.csv"])
gpu_proc.terminate()
try: gpu_proc.wait(timeout=5)
except subprocess.TimeoutExpired: gpu_proc.kill()

# 10. Part E ablation ladder on real HellaSwag.
os.environ["N_EVAL"] = str(N_EVAL)
os.environ["N_POOL"] = str(N_POOL)
os.environ["MAX_CONCURRENCY"] = "16"
subprocess.check_call(["bash", "improve/eval.sh"])

# 11. Part D determinism.
subprocess.check_call(["uv", "run", "python", "-m", "guardrails.validate", "determinism",
                       "The capital of France is", "--n-runs", "5", "--max-tokens", "16",
                       "--seed", "42", "--report", "results/determinism_report.json"])

# 12. SANITY GATE -- refuse to pack if anything looks synthetic.
eval_jsonl = Path("results/improve/hellaswag_eval.jsonl")
first = json.loads(eval_jsonl.read_text().splitlines()[0])
assert not str(first.get("ind", "")).startswith("syn-"), (
    f"Part E scored against synthetic data (ind={first.get('ind')!r}). Aborting."
)

rows = list(csv.DictReader(open("results/result-log.csv")))
real_part_e = [r for r in rows
               if r["task"] == "hellaswag" and r["mock_backend"] == "false"
               and r["limit"] == str(N_EVAL)
               and r["method"] != f"real_gpu_{MODEL}"]
assert len(real_part_e) >= 7, f"expected 7 real Part E variant rows with limit={N_EVAL}, got {len(real_part_e)}"

# Also require that at least one variant accuracy is *strictly between*
# 0.0 and 1.0 -- synthetic data famously pegs fewshot_*_5 at exactly 1.0.
for r in real_part_e:
    v = float(r["value"])
    assert 0.0 < v < 1.0, f"variant {r['method']} has value={v} -- suspicious, likely synthetic"
print("sanity gate passed")

# 13. Stop vLLM + pack.
import signal
serve_proc.send_signal(signal.SIGTERM)
try: serve_proc.wait(timeout=30)
except subprocess.TimeoutExpired: serve_proc.kill()

subprocess.check_call(["tar", "czf", "/tmp/llmeval_results.tgz",
                       "results/", "docs/improvement-log.md", "improve/report.md", "vllm_serve.log"])
subprocess.check_call(["ls", "-lh", "/tmp/llmeval_results.tgz"])

# 14. Download.
from google.colab import files
files.download("/tmp/llmeval_results.tgz")
```

## What this does differently from the notebook

- **Force-deletes any previous `LLMEvalSystem` clone** at `/content/` before
  cloning. This is the fix for the "same stale working dir across reruns"
  trap.
- **Asserts `improve/eval.sh` contains the post-fix text** before running.
  If your fork is behind, the cell fails immediately with a message telling
  you to push.
- **Asserts Part E variant accuracies are strictly between 0 and 1.** Real
  Qwen2.5-1.5B on real HellaSwag has baseline ~0.50, fewshot ~0.65; synthetic
  data pegs `fewshot_*_5` at exactly 1.0000.
- **Refuses to pack the tarball** if any of the above fail. You cannot
  accidentally ship synthetic numbers.

## Back on the Mac

```bash
cd path/to/LLMEvalSystem
git fetch origin
git reset --hard origin/main           # start from the clean-log state

tar xzf ~/Downloads/llmeval_results.tgz

# One-line paranoid verification before git-add:
python3 -c "
import json, csv
first = json.loads(open('results/improve/hellaswag_eval.jsonl').read().splitlines()[0])
assert not first['ind'].startswith('syn-'), 'SYNTHETIC'
rows = list(csv.DictReader(open('results/result-log.csv')))
part_e = [r for r in rows if r['task']=='hellaswag' and r['mock_backend']=='false'
          and r['limit']=='200' and not r['method'].startswith('real_gpu_')]
assert len(part_e) >= 7 and all(0 < float(r['value']) < 1 for r in part_e)
print('OK:', len(part_e), 'real Part E rows')"

git add -A
git commit -m "chore: real Part E ablation numbers"
git push
```

If the paranoid check prints `OK: 7 real Part E rows` you are safe to
commit. If it raises, do NOT commit — delete what you unpacked and rerun
the Colab cell.
