"""Launch a vLLM OpenAI-compatible server using settings from :mod:`common.config`.

This is a thin wrapper around the ``vllm serve`` CLI. The command-building logic
lives in :func:`build_serve_command` as a pure function so it can be unit-tested
without actually spawning vLLM.

Usage::

    # Defaults from LLMEVAL_* env vars / .env:
    python -m serve.serve

    # Override per-invocation:
    python -m serve.serve --model meta-llama/Llama-3.2-3B-Instruct --port 9000

The process forwards ``SIGINT`` / ``SIGTERM`` to the child so ``Ctrl-C`` shuts
vLLM down cleanly.
"""

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from types import FrameType

from common.config import Settings, get_settings
from common.errors import ServeError
from common.logging import get_logger

log = get_logger(__name__)

_VLLM_ENTRYPOINT = "vllm"


@dataclass(frozen=True, slots=True)
class ServeOverrides:
    """Per-invocation overrides layered on top of :class:`Settings`.

    Any field left as ``None`` falls back to the corresponding ``Settings`` value.
    Keeping overrides in a dataclass (not raw kwargs) keeps :func:`build_serve_command`
    purely data-in / data-out, which is what makes it cheap to test.

    Attributes:
        model: HuggingFace-style model identifier.
        host: Bind host.
        port: Bind port.
        max_model_len: Override model max sequence length.
        dtype: Weight precision (``auto`` / ``float16`` / ``bfloat16`` / ``float32``).
        gpu_memory_utilization: Fraction of free GPU memory vLLM may allocate.
        tensor_parallel_size: Number of GPUs for tensor-parallel sharding.
        max_num_seqs: Max concurrent sequences per step.
        trust_remote_code: Force ``--trust-remote-code`` on.
    """

    model: str | None = None
    host: str | None = None
    port: int | None = None
    max_model_len: int | None = None
    dtype: str | None = None
    gpu_memory_utilization: float | None = None
    tensor_parallel_size: int | None = None
    max_num_seqs: int | None = None
    trust_remote_code: bool | None = None


def build_serve_command(settings: Settings, overrides: ServeOverrides) -> list[str]:
    """Assemble the argv list for ``vllm serve``.

    Args:
        settings: Global application settings (defaults).
        overrides: Per-invocation overrides; non-``None`` fields win over ``settings``.

    Returns:
        Argv list starting with the ``vllm`` executable name.
    """
    model = overrides.model or settings.model_name
    host = overrides.host or settings.vllm_host
    port = overrides.port if overrides.port is not None else settings.vllm_port
    max_model_len = (
        overrides.max_model_len
        if overrides.max_model_len is not None
        else settings.vllm_max_model_len
    )
    dtype = overrides.dtype or settings.vllm_dtype
    gpu_mem = (
        overrides.gpu_memory_utilization
        if overrides.gpu_memory_utilization is not None
        else settings.vllm_gpu_memory_utilization
    )
    tp_size = (
        overrides.tensor_parallel_size
        if overrides.tensor_parallel_size is not None
        else settings.vllm_tensor_parallel_size
    )
    max_num_seqs = (
        overrides.max_num_seqs
        if overrides.max_num_seqs is not None
        else settings.vllm_max_num_seqs
    )
    trust_remote = (
        overrides.trust_remote_code
        if overrides.trust_remote_code is not None
        else settings.vllm_trust_remote_code
    )

    cmd: list[str] = [
        _VLLM_ENTRYPOINT,
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--api-key",
        settings.vllm_api_key,
        "--dtype",
        dtype,
        "--gpu-memory-utilization",
        str(gpu_mem),
        "--tensor-parallel-size",
        str(tp_size),
        "--seed",
        str(settings.seed),
    ]
    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])
    if max_num_seqs is not None:
        cmd.extend(["--max-num-seqs", str(max_num_seqs)])
    if trust_remote:
        cmd.append("--trust-remote-code")
    if settings.vllm_download_dir is not None:
        cmd.extend(["--download-dir", str(settings.vllm_download_dir)])
    return cmd


def _parse_args(argv: list[str] | None = None) -> ServeOverrides:
    """Parse CLI argv into a :class:`ServeOverrides`."""
    p = argparse.ArgumentParser(description="Launch a vLLM OpenAI-compatible server.")
    p.add_argument("--model", default=None, help="HF model id (overrides LLMEVAL_MODEL_NAME)")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=None, dest="max_model_len")
    p.add_argument(
        "--dtype",
        default=None,
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        dest="gpu_memory_utilization",
    )
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        dest="tensor_parallel_size",
    )
    p.add_argument("--max-num-seqs", type=int, default=None, dest="max_num_seqs")
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=None,
        dest="trust_remote_code",
    )
    ns = p.parse_args(argv)
    return ServeOverrides(
        model=ns.model,
        host=ns.host,
        port=ns.port,
        max_model_len=ns.max_model_len,
        dtype=ns.dtype,
        gpu_memory_utilization=ns.gpu_memory_utilization,
        tensor_parallel_size=ns.tensor_parallel_size,
        max_num_seqs=ns.max_num_seqs,
        trust_remote_code=ns.trust_remote_code,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point used by ``python -m serve.serve``.

    Args:
        argv: Optional argv for testing; defaults to ``sys.argv[1:]``.

    Returns:
        The subprocess exit code, or ``0`` if mock backend is enabled.

    Raises:
        ServeError: If the ``vllm`` executable is missing or fails to spawn.
    """
    settings = get_settings()

    if settings.mock_backend:
        log.warning("mock_backend_enabled_skipping_serve")
        sys.stderr.write(
            "LLMEVAL_MOCK_BACKEND=true: not launching vLLM. "
            "Use VLLMClient directly; it will return canned responses.\n"
        )
        return 0

    if shutil.which(_VLLM_ENTRYPOINT) is None:
        raise ServeError(
            f"'{_VLLM_ENTRYPOINT}' not on PATH. Install with `uv sync --extra serve` "
            "and ensure the project venv is active."
        )

    overrides = _parse_args(argv)
    cmd = build_serve_command(settings, overrides)
    log.info(
        "vllm_serve_starting",
        model=cmd[2],
        host=settings.vllm_host,
        port=settings.vllm_port,
        argv=cmd,
    )

    try:
        proc = subprocess.Popen(cmd)
    except OSError as exc:
        raise ServeError(f"failed to spawn vllm serve: {exc}") from exc

    def _forward_signal(signum: int, _frame: FrameType | None) -> None:
        log.info("vllm_serve_forwarding_signal", signum=signum)
        proc.send_signal(signum)

    signal.signal(signal.SIGINT, _forward_signal)
    signal.signal(signal.SIGTERM, _forward_signal)

    returncode = proc.wait()
    log.info("vllm_serve_exited", returncode=returncode)
    return returncode


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["ServeOverrides", "build_serve_command", "main"]
