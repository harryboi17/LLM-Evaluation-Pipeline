"""Sanity checks for the custom task data file.

These tests run without ``lm-eval`` installed so the JSONL contract is
guaranteed even in the minimal dev install.
"""

from __future__ import annotations

import json
from pathlib import Path

_DATA_FILE = Path(__file__).parents[2] / "eval_runner" / "tasks" / "custom_data.jsonl"


def _load_examples() -> list[dict[str, str]]:
    lines = _DATA_FILE.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_custom_data_file_exists() -> None:
    assert _DATA_FILE.exists(), f"missing {_DATA_FILE}"


def test_custom_data_has_exactly_50_examples() -> None:
    examples = _load_examples()
    assert len(examples) == 50


def test_every_example_has_question_and_answer() -> None:
    for i, ex in enumerate(_load_examples()):
        assert "question" in ex, f"line {i + 1} missing 'question'"
        assert "answer" in ex, f"line {i + 1} missing 'answer'"
        assert ex["question"].strip(), f"line {i + 1} empty question"
        assert ex["answer"].strip(), f"line {i + 1} empty answer"


def test_answers_are_short_and_deterministic() -> None:
    """Keep answers single-token-ish so exact-match scoring is meaningful."""
    for ex in _load_examples():
        # At most 3 whitespace tokens — any longer and exact-match is too brittle.
        assert len(ex["answer"].split()) <= 3, ex
