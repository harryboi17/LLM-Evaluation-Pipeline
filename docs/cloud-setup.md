# Cloud setup guide

The project is designed to run on a CUDA-capable Linux box. Local Mac and
corporate-proxy environments have two structural limits — no GPU and SSL
interception — that neither the code nor you can work around in software.
This doc is the short answer to "where do I run this to get real numbers".

## TL;DR: rent an hourly GPU, run one script

| Provider | $/hr | GPU | Time to first `make serve` |
|---|---|---|---|
| **[Google Colab free](https://colab.research.google.com)** | **$0** | T4 (16 GB) | **~3 min** (browser only) |
| **[Kaggle Notebooks free](https://kaggle.com/code)** | $0 | T4 ×2 (30 GB) | ~3 min |
| **[RunPod](https://runpod.io)** | $0.20–0.50 | RTX 3090 / A5000 | ~5 min |
| **[Lambda Labs](https://lambdalabs.com)** | $0.75–1.29 | A10 / A100 | ~5 min |
| **AWS EC2 g5.xlarge** | $1.01 (spot: ~$0.40) | A10G 24 GB | ~15 min |
| **GitHub Codespaces GPU** | $0.36 | Tesla T4 | ~3 min (1 click) |

**For a one-shot submission run, use Colab free.** The project fits in a
single notebook (`notebooks/colab_run.ipynb`), a full real run takes ~45
minutes, and you don't need a credit card. Details in the next section.

A full Part B / C / E run on a 1–4 B-parameter model fits in under an
hour on any 16 GB card, so **$0.25–$1.50 covers paid options end-to-end**
if you prefer something beefier than a T4.

## Colab free (cheapest path to real numbers)

The repo ships a fully-wired [`notebooks/colab_run.ipynb`](../notebooks/colab_run.ipynb)
that handles everything: installs deps, boots vLLM, runs eval + perf +
improve, packages results, and downloads a tarball to your laptop.

**Steps:**

1. Push your fork of the repo to GitHub (Colab clones via public URL).
2. Open [Google Colab](https://colab.research.google.com) → `File` → `Upload notebook`
   → pick `notebooks/colab_run.ipynb` from your local checkout.
3. `Runtime` → `Change runtime type` → **T4 GPU**.
4. Edit the config cell: set `REPO_URL` to your GitHub URL.
   Default `MODEL` is `Qwen/Qwen2.5-1.5B-Instruct` — **not gated**, no
   HF token needed. Alternatives in the notebook comments.
5. `Runtime` → `Run all`. Total wall time ~45–60 min. Leave the tab
   visible (Colab idles you out if the browser backgrounds you too long).
6. The last cell downloads `llmeval_results.tgz`. Back on your laptop:

   ```bash
   cd path/to/LLM-Evaluation-Pipeline
   tar xzf ~/Downloads/llmeval_results.tgz
   git add results/ docs/improvement-log.md improve/report.md
   git commit -m "chore: real numbers from Colab T4 run"
   ```

**Why Qwen2.5-1.5B instead of Llama-3.2-1B?** Both are fine choices under
the problem statement ("any open-weight model: Llama 3 / Mistral / Phi").
Qwen avoids the HF gated-model dance (accept license → get token → paste
into notebook). If you already have a Llama token, flip the `MODEL`
variable and paste `HF_TOKEN`.

**Kaggle** is a drop-in alternative: same notebook works, just upload
through the Kaggle notebooks interface and pick the T4×2 accelerator. Use
it if Colab's free tier has been idle-killing you.

## RunPod walkthrough (paid, faster GPU, persistent storage)

1. Account → add $10 → Deploy Pod
2. GPU: `RTX 3090` or `A5000`; Template: *PyTorch 2.4*; Storage 40 GB;
   Expose port `8000` (TCP)
3. SSH in using the credentials the dashboard gives you
4. One-shot bootstrap:

   ```bash
   export REPO_URL=https://github.com/harryboi17/LLM-Evaluation-Pipeline.git
   export HF_TOKEN=hf_xxxxxxxxxxxxx   # Llama is gated; get a token on huggingface.co
   curl -fsSL https://raw.githubusercontent.com/harryboi17/LLM-Evaluation-Pipeline/main/scripts/cloud_bootstrap.sh | bash
   ```

   (Or: clone the repo first and run `bash scripts/cloud_bootstrap.sh` in the
   checkout.)

5. Real run:

   ```bash
   cd LLM-Evaluation-Pipeline

   # Boot vLLM (first time downloads ~2 GB of weights)
   make serve &

   # Wait for the "Uvicorn running on http://0.0.0.0:8000" log line, then:
   make eval
   make perf
   python -m perf.gpu_monitor --output results/perf/gpu.csv --duration 180 &
   bash improve/eval.sh    # N_EVAL=500 bash improve/eval.sh for full numbers
   ```

6. Pull results home:

   ```bash
   # On your Mac / laptop:
   scp -P <port> -r root@<pod-ip>:/workspace/LLM-Evaluation-Pipeline/results ./results
   scp -P <port> -r root@<pod-ip>:/workspace/LLM-Evaluation-Pipeline/docs/improvement-log.md ./docs/
   ```

7. **Stop the pod when done** (RunPod doesn't auto-stop; you'll pay for
   idle time). Saved volume means re-renting same pod resumes where you
   left off, including the cached model weights.

## Lambda Labs / Paperspace / AWS EC2

Same shape as RunPod — SSH in, run `scripts/cloud_bootstrap.sh`, run
`make serve`. The only per-provider variance is how you allocate the box
and open port 8000.

For AWS specifically, use the **Deep Learning AMI (Ubuntu 22.04)** so
CUDA is pre-installed; a fresh Ubuntu AMI will need an extra
`nvidia-driver-535` + reboot step.

## GitHub Codespaces GPU

Zero-setup. In the repo's GitHub page:

1. `<> Code` → Codespaces → *New with options*
2. Machine type: **GPU (4-core, 16 GB)**, region: closest to you
3. Shell in the VS Code browser view; the repo is already there, and VS
   Code picks up `AGENTS.md` / `.claude/` / `.cursor/rules/` automatically

Then:

```bash
bash scripts/cloud_bootstrap.sh
make serve        # in one VS Code terminal
bash improve/eval.sh  # in another
```

## If you must use a corporate GPU box (Zscaler + internal proxy)

All the SSL / HF Hub issues we hit locally come back. Mitigations, in
rough order of preference:

1. **Trust-bundle the Zscaler root CA into the venv's `certifi` file**
   once. This unblocks `pip`, `huggingface_hub`, `requests`, and
   `sentence-transformers` in a single shot:

   ```bash
   ZSCALER_CERT=/etc/ssl/certs/ZscalerRootCertificate.pem   # adjust to your env
   CERTIFI_BUNDLE=$(uv run python -c "import certifi; print(certifi.where())")
   cat "$ZSCALER_CERT" >> "$CERTIFI_BUNDLE"
   ```

   Plus export these in `~/.bashrc` so every tool agrees:

   ```bash
   export SSL_CERT_FILE=$CERTIFI_BUNDLE
   export REQUESTS_CA_BUNDLE=$CERTIFI_BUNDLE
   export CURL_CA_BUNDLE=$CERTIFI_BUNDLE
   ```

2. **Pre-download models from a non-intercepted network**, copy them
   into `~/.cache/huggingface/`. Works for `transformers` and
   `sentence-transformers` alike.

3. **Set `LLMEVAL_MOCK_BACKEND=true`** for any CI / smoke workload that
   doesn't need real model output. The project is explicitly structured
   so every code path has a mock-backend fall-through.

If none of these are workable, **pay for a rented GPU box for an hour**
— it's cheaper than the engineering hours to fight a corporate proxy.

## What about local options on the Mac?

Both work for dev, neither gives you production-grade numbers:

- **Ollama** — install with `brew install ollama`, `ollama serve`,
  `ollama pull llama3.2:1b-instruct`. Point `.env` at `127.0.0.1:11434`.
  See the main README for the exact env-var config. Good enough for
  `custom_qa` generative eval; **loglikelihood scoring (HellaSwag, MMLU)
  may be flaky** because Ollama's `echo` + `logprobs` support is partial.

- **LM Studio** — GUI alternative, same caveats.

Use them for iterating on prompts; use a rented GPU for the final
numbers that go in `improve/report.md`.

## Recommended flow

1. Clone on Mac, run `bash scripts/smoke.sh` to verify the dev loop
   (mock backend, no external deps).
2. Iterate locally using Ollama for fast generative-eval feedback.
3. When you want real numbers for `improve/report.md`: rent a RunPod
   box for an hour, run `scripts/cloud_bootstrap.sh`, then `make eval`
   + `bash improve/eval.sh`. `scp` the `results/` directory back.
4. Commit the updated `results/` + `docs/improvement-log.md` to the
   branch you're submitting.

Total spend from zero to a full real run: **< $2**.
