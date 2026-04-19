---
name: start-vllm
description: Launch the vLLM OpenAI-compatible server for this project
allowed-tools:
  - read
  - exec
permissions:
  allow:
    - Exec(make serve)
    - Exec(uv run python -m serve.serve)
---

Start the vLLM server using the project's configured model.

1. Confirm the server is not already running: `curl -sf http://127.0.0.1:8000/health` (or the host/port from `.env`). If it responds, report the model it's serving and stop.
2. Check `.env` exists; if missing, copy from `.env.example` and warn the user.
3. Run `make serve` to launch the server (`python -m serve.serve` under the hood, which wraps `vllm serve`).
4. Tail the first ~30 lines of output to verify the server reached "Started server" / "Uvicorn running on ...".
5. Report the served model name, host, and port.

If launching fails, do not retry blindly — surface the error (common causes: missing GPU, out of VRAM, model not downloaded, port already in use) and let the user decide.
