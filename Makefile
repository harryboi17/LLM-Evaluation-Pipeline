# LLM Eval System — Makefile
# Single entry point for every operation. All targets are phony.

SHELL := /usr/bin/env bash

PY ?= uv run python
UV ?= uv

# --- Settings (sourced from .env if present) ------------------------------
ifneq (,$(wildcard .env))
  include .env
  export
endif

MODEL ?= $(LLMEVAL_MODEL_NAME)
MODEL := $(if $(MODEL),$(MODEL),meta-llama/Llama-3.2-1B-Instruct)
VLLM_HOST ?= $(if $(LLMEVAL_VLLM_HOST),$(LLMEVAL_VLLM_HOST),127.0.0.1)
VLLM_PORT ?= $(if $(LLMEVAL_VLLM_PORT),$(LLMEVAL_VLLM_PORT),8000)

.PHONY: help install install-all serve client-demo eval perf perf-analyze improve \
        lint format typecheck test test-cov smoke clean all bootstrap

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# --- Environment ----------------------------------------------------------

install: ## Install base + dev deps (no GPU libs)
	$(UV) sync
	$(UV) run pre-commit install

install-all: ## Install base + dev + all optional extras (needs GPU + internet)
	$(UV) sync --all-extras
	$(UV) run pre-commit install

bootstrap: ## First-time setup (install + .env + secrets baseline)
	bash scripts/bootstrap.sh

# --- Serving (Part A) -----------------------------------------------------

serve: ## Launch vLLM OpenAI-compatible server
	$(PY) -m serve.serve --model "$(MODEL)" --host "$(VLLM_HOST)" --port $(VLLM_PORT)

client-demo: ## Run a few sample generations against a running server
	$(PY) -m serve.examples.demo

# --- Evaluation (Part B) --------------------------------------------------

eval: ## Run MMLU, HellaSwag, and custom benchmark
	$(PY) -m eval_runner.run_eval --task mmlu,hellaswag,custom

# --- Performance (Part C) -------------------------------------------------

perf: ## Run load test, produce metrics.csv
	$(PY) -m perf.load_test --output metrics.csv

perf-analyze: ## Execute perf/analysis.ipynb to HTML
	$(UV) run jupyter nbconvert --to html --execute perf/analysis.ipynb

# --- Improve (Part E) -----------------------------------------------------

improve: ## Run the Part E benchmark-improvement pipeline
	bash improve/eval.sh

# --- Dev loop -------------------------------------------------------------

lint: ## Ruff lint
	$(UV) run ruff check .

format: ## Ruff format (+ auto-fix lint)
	$(UV) run ruff format .
	$(UV) run ruff check . --fix

typecheck: ## Mypy
	$(UV) run mypy common serve eval_runner guardrails improve

test: ## Pytest (no coverage gate)
	$(UV) run pytest

test-cov: ## Pytest with HTML coverage report
	$(UV) run pytest --cov-report=html

smoke: ## End-to-end sanity check (bootstrap + mock inference + one eval item)
	bash scripts/smoke.sh

all: lint typecheck test ## Lint, typecheck, and test

clean: ## Remove caches and temporary artifacts
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage .coverage.*
	rm -rf .cache results/tmp results/raw
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
