"""Tests for ``improve.infer`` — eval loop against the mock backend."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.config import get_settings


@pytest.fixture
def _mock_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("LLMEVAL_MOCK_BACKEND", "true")
    monkeypatch.setenv("LLMEVAL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LLMEVAL_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("LLMEVAL_MODEL_NAME", "mock/test-model")
    # Don't pollute the real docs/improvement-log.md during tests.
    monkeypatch.setenv("LLMEVAL_IMPROVEMENT_LOG_PATH", str(tmp_path / "improvement-log.md"))
    get_settings.cache_clear()
    return tmp_path


def _sample_rows(n: int = 4) -> list[dict[str, object]]:
    """Build a small HellaSwag-shaped dataset with predictable labels."""
    rows: list[dict[str, object]] = []
    for i in range(n):
        rows.append(
            {
                "ind": f"ex-{i}",
                "ctx": f"[trivia] Physics Story {i}: the ball rolls",
                "activity_label": "physics",
                "endings": [
                    " downhill because gravity pulls it",
                    " uphill because of magic",
                    " sideways because time",
                    " backward because of entropy",
                ],
                "label": i % 4,
            }
        )
    return rows


def _write_split(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_baseline_evaluate_runs_and_returns_one_result_per_example(
    _mock_env: Path,
) -> None:
    import asyncio

    from improve.infer import evaluate

    eval_rows = _sample_rows(3)
    pool_rows = _sample_rows(2)

    results = asyncio.run(
        evaluate(
            variant_name="baseline",
            eval_rows=eval_rows,
            pool_rows=pool_rows,
            max_concurrency=4,
            seed=42,
            model="mock/test-model",
        )
    )
    assert len(results) == 3
    for r in results:
        assert 0 <= r.predicted < 4
        assert len(r.scores) == 4


def test_clean_length_norm_variant_runs(_mock_env: Path) -> None:
    import asyncio

    from improve.infer import evaluate

    eval_rows = _sample_rows(2)
    pool_rows = _sample_rows(0)
    results = asyncio.run(
        evaluate(
            variant_name="clean_length_norm",
            eval_rows=eval_rows,
            pool_rows=pool_rows,
            max_concurrency=2,
            seed=42,
            model="mock/test-model",
        )
    )
    assert len(results) == 2


def test_fewshot_random_variant_runs_without_semantic_retriever(
    _mock_env: Path,
) -> None:
    import asyncio

    from improve.infer import evaluate

    eval_rows = _sample_rows(2)
    pool_rows = _sample_rows(3)
    results = asyncio.run(
        evaluate(
            variant_name="fewshot_random_5",
            eval_rows=eval_rows,
            pool_rows=pool_rows,
            max_concurrency=2,
            seed=42,
            model="mock/test-model",
        )
    )
    assert len(results) == 2


def test_main_end_to_end_baseline_and_variant_logs_result(
    _mock_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from improve.infer import main

    out_dir = _mock_env / "results" / "improve"
    _write_split(out_dir / "hellaswag_eval.jsonl", _sample_rows(3))
    _write_split(out_dir / "hellaswag_fewshot_pool.jsonl", _sample_rows(2))

    # First run: baseline. Should produce baseline.json and a result-log row.
    rc1 = main(["--variant", "baseline", "--n-eval", "3", "--bootstrap-iters", "500"])
    assert rc1 == 0
    assert (out_dir / "baseline.json").exists()
    summary1 = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary1["variant"] == "baseline"
    assert summary1["delta_vs_baseline"] is None  # no prior run to compare to

    # Second run: a variant. Should compute a delta + CI vs baseline.
    rc2 = main(
        [
            "--variant",
            "length_norm",
            "--n-eval",
            "3",
            "--bootstrap-iters",
            "500",
            "--notes",
            "length-normalised scoring",
        ]
    )
    assert rc2 == 0
    summary2 = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary2["variant"] == "length_norm"
    assert summary2["delta_vs_baseline"] is not None
    assert isinstance(summary2["ci95"], list) and len(summary2["ci95"]) == 2
    assert summary2["p_value"] is not None

    # The result log should have two rows at this point.
    from common.result_log import read_results

    rows = read_results()
    assert len(rows) == 2
    assert rows[0]["method"] == "baseline"
    assert rows[1]["method"] == "length_norm"
