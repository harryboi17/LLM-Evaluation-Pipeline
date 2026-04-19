#!/usr/bin/env bash
# Cloud-box bootstrap. Run on a fresh Linux + CUDA box (RunPod, Lambda Labs,
# AWS g5, GCE with L4, etc.) to get from SSH to a working `make serve` +
# `make eval` + `bash improve/eval.sh` in one shot.
#
# Assumes:
#   - Ubuntu 20.04 / 22.04 / 24.04 (or similar glibc-based Linux)
#   - CUDA 12.x already installed (true on every mainstream PyTorch AMI)
#   - Internet access with no SSL MITM (corporate proxies like Zscaler need
#     extra work; see docs/cloud-setup.md for the workaround)
#
# Usage:
#   curl -fsSL <repo-raw>/scripts/cloud_bootstrap.sh | bash
# OR, if you already cloned the repo:
#   bash scripts/cloud_bootstrap.sh
#
# Environment overrides:
#   HF_TOKEN        -- HuggingFace token (required for gated Llama models)
#   REPO_URL        -- git URL to clone if not already in a checkout
#   LLMEVAL_BRANCH  -- branch / tag to check out (default: main)
set -euo pipefail

log() { printf "\033[36m[bootstrap]\033[0m %s\n" "$*"; }
die() { printf "\033[31m[bootstrap error]\033[0m %s\n" "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# System sanity checks
# ---------------------------------------------------------------------------

log "verifying GPU access via nvidia-smi"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi not found. This script assumes a CUDA-capable Linux box. On a CPU box, skip cloud_bootstrap.sh and just run 'uv sync && bash scripts/smoke.sh'."
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

log "verifying git + curl + build-essential"
for bin in git curl gcc make; do
    command -v "$bin" >/dev/null 2>&1 || die "$bin not installed; run 'apt-get update && apt-get install -y $bin'"
done

# ---------------------------------------------------------------------------
# uv (fast Python package manager; also manages the 3.11 interpreter)
# ---------------------------------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    # persist for future shells
    if ! grep -q '\.local/bin' "${HOME}/.bashrc" 2>/dev/null; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "${HOME}/.bashrc"
    fi
fi
uv --version

# ---------------------------------------------------------------------------
# Project checkout
# ---------------------------------------------------------------------------

if [ ! -f pyproject.toml ]; then
    if [ -z "${REPO_URL:-}" ]; then
        die "not in a project checkout and REPO_URL isn't set; either cd into the repo first or export REPO_URL=<git url>"
    fi
    log "cloning ${REPO_URL}"
    git clone "${REPO_URL}" LLM-Evaluation-Pipeline
    cd LLM-Evaluation-Pipeline
    if [ -n "${LLMEVAL_BRANCH:-}" ]; then
        git checkout "${LLMEVAL_BRANCH}"
    fi
fi

# ---------------------------------------------------------------------------
# Python deps -- every extra at once so we don't thrash the wheel cache
# ---------------------------------------------------------------------------

log "syncing uv project with every extra (first run pulls torch + CUDA libs; can be several minutes)"
uv sync --extra serve --extra eval --extra perf --extra improve

log "confirming the key packages import cleanly"
uv run python - <<'PY'
import importlib
for mod in ("torch", "vllm", "lm_eval", "transformers", "sentence_transformers", "pandas"):
    try:
        importlib.import_module(mod)
        print(f"ok   {mod}")
    except ImportError as exc:
        print(f"FAIL {mod}: {exc}")
        raise
PY

# ---------------------------------------------------------------------------
# HuggingFace auth (optional, needed for gated models like Llama-3.2)
# ---------------------------------------------------------------------------

if [ -n "${HF_TOKEN:-}" ]; then
    log "configuring HuggingFace token"
    uv run huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential || true
else
    log "HF_TOKEN not set -- gated models (e.g. Llama-3.2) will fail to download"
    log "  export HF_TOKEN=hf_xxx before running 'make serve' if using a gated model"
fi

# ---------------------------------------------------------------------------
# Smoke -- mock backend only, proves the stack imports and runs
# ---------------------------------------------------------------------------

log "running full smoke against the mock backend"
bash scripts/smoke.sh

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

cat <<'MSG'

====================================================================
 bootstrap complete. next steps:

   # terminal 1: boot vLLM (takes ~60s first time while weights download)
   make serve

   # terminal 2: real evaluation numbers
   make eval                                    # MMLU-STEM + HellaSwag + custom_qa
   make perf                                    # TTFT / TPOT / P99 under load
   bash improve/eval.sh                         # Part E full ablation

   # pull the results back to your local machine:
   #   scp -r user@<this-box>:<repo>/results ./results
====================================================================
MSG
