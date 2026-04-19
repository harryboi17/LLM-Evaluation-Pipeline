"""Tests for ``serve.serve``."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.config import Settings
from common.errors import ServeError
from serve.serve import ServeOverrides, _parse_args, build_serve_command, main


def _make_settings(**overrides: object) -> Settings:
    """Build a ``Settings`` instance bypassing the environment for deterministic tests."""
    defaults: dict[str, object] = {
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "vllm_host": "127.0.0.1",
        "vllm_port": 8000,
        "vllm_api_key": "EMPTY",
        "vllm_dtype": "auto",
        "vllm_gpu_memory_utilization": 0.9,
        "vllm_tensor_parallel_size": 1,
        "vllm_max_model_len": None,
        "vllm_max_num_seqs": None,
        "vllm_trust_remote_code": False,
        "vllm_download_dir": None,
        "seed": 42,
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)  # type: ignore[arg-type]


def test_build_serve_command_uses_settings_defaults() -> None:
    cmd = build_serve_command(_make_settings(), ServeOverrides())

    assert cmd[0] == "vllm"
    assert cmd[1] == "serve"
    assert cmd[2] == "meta-llama/Llama-3.2-1B-Instruct"
    assert "--host" in cmd and "127.0.0.1" in cmd
    assert "--port" in cmd and "8000" in cmd
    assert "--dtype" in cmd and "auto" in cmd
    assert "--gpu-memory-utilization" in cmd and "0.9" in cmd
    assert "--tensor-parallel-size" in cmd and "1" in cmd
    assert "--seed" in cmd and "42" in cmd
    # Optional flags not set at default:
    assert "--max-model-len" not in cmd
    assert "--max-num-seqs" not in cmd
    assert "--trust-remote-code" not in cmd
    assert "--download-dir" not in cmd


def test_build_serve_command_overrides_win() -> None:
    cmd = build_serve_command(
        _make_settings(),
        ServeOverrides(
            model="meta-llama/Llama-3.2-3B-Instruct",
            host="0.0.0.0",
            port=9001,
            max_model_len=4096,
            dtype="bfloat16",
            gpu_memory_utilization=0.75,
            tensor_parallel_size=2,
            max_num_seqs=64,
            trust_remote_code=True,
        ),
    )
    assert cmd[2] == "meta-llama/Llama-3.2-3B-Instruct"
    assert "0.0.0.0" in cmd
    assert "9001" in cmd
    assert cmd.count("--max-model-len") == 1
    assert "4096" in cmd
    assert "bfloat16" in cmd
    assert "0.75" in cmd
    assert "2" in cmd
    assert "64" in cmd
    assert "--trust-remote-code" in cmd


def test_build_serve_command_download_dir_from_settings(tmp_path: Path) -> None:
    cmd = build_serve_command(
        _make_settings(vllm_download_dir=tmp_path),
        ServeOverrides(),
    )
    assert "--download-dir" in cmd
    assert str(tmp_path) in cmd


def test_parse_args_returns_overrides_dataclass() -> None:
    overrides = _parse_args(
        [
            "--model",
            "mistralai/Mistral-7B-v0.1",
            "--port",
            "9999",
            "--dtype",
            "float16",
            "--trust-remote-code",
        ]
    )
    assert overrides.model == "mistralai/Mistral-7B-v0.1"
    assert overrides.port == 9999
    assert overrides.dtype == "float16"
    assert overrides.trust_remote_code is True
    # Unspecified flags come back as None:
    assert overrides.host is None
    assert overrides.max_model_len is None


def test_parse_args_rejects_invalid_dtype() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--dtype", "int8"])


def test_main_in_mock_mode_returns_zero_without_spawning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLMEVAL_MOCK_BACKEND", "true")
    # Sentinel: if we accidentally call shutil.which, this would fail.
    monkeypatch.setattr(
        "serve.serve.shutil.which",
        lambda _name: "/nonexistent/vllm",
    )
    # Belt-and-braces: if we try to Popen, blow up loudly.

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Popen should not be called in mock mode")

    monkeypatch.setattr("serve.serve.subprocess.Popen", _boom)
    rc = main([])
    assert rc == 0


def test_main_raises_when_vllm_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMEVAL_MOCK_BACKEND", "false")
    monkeypatch.setattr("serve.serve.shutil.which", lambda _name: None)
    with pytest.raises(ServeError, match="not on PATH"):
        main([])
