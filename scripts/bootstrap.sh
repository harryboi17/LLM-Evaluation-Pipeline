#!/usr/bin/env bash
# First-time setup for a fresh clone. Idempotent.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required but not installed. Install from https://github.com/astral-sh/uv" >&2
  exit 1
fi

echo "[bootstrap] installing base + dev deps"
uv sync

echo "[bootstrap] installing pre-commit hooks"
uv run pre-commit install

if [ ! -f .env ]; then
  echo "[bootstrap] creating .env from .env.example"
  cp .env.example .env
fi

if [ ! -f .secrets.baseline ]; then
  echo "[bootstrap] creating detect-secrets baseline"
  uv run detect-secrets scan > .secrets.baseline
fi

echo "[bootstrap] done. Next steps:"
echo "  make serve      # launch the vLLM server (needs GPU)"
echo "  make eval       # run benchmarks"
echo "  make all        # lint + typecheck + test"
