"""Tests for ``guardrails.validate`` — schema / short-answer validators."""

from __future__ import annotations

import pytest

from common.errors import ValidationError
from guardrails.validate import (
    load_schema,
    validate_batch,
    validate_output,
    validate_short_answer,
)


def test_load_schema_reads_short_answer() -> None:
    schema = load_schema("short_answer")
    assert schema["title"] == "Short Answer"
    assert "pattern" in schema


def test_load_schema_raises_for_missing_name() -> None:
    with pytest.raises(ValidationError, match="no such schema"):
        load_schema("does_not_exist")


@pytest.mark.parametrize(
    "text",
    ["Paris", "56", "Washington DC", "H2O", "New York"],
)
def test_validate_short_answer_accepts_well_formed(text: str) -> None:
    validate_short_answer(text)  # should not raise


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "empty"),
        (" Paris", "whitespace"),
        ("Paris ", "whitespace"),
        ("a b c d", "tokens"),
        ("Paris.", "punctuation"),
        ("What?", "punctuation"),
        ("Café", "non-ASCII"),
    ],
)
def test_validate_short_answer_rejects_bad_input(text: str, reason: str) -> None:
    with pytest.raises(ValidationError):
        validate_short_answer(text)


def test_validate_output_against_mcq_schema_accepts() -> None:
    schema = load_schema("mcq_letter")
    validate_output("A", schema)
    validate_output("C.", schema)


@pytest.mark.parametrize("bad", ["", "a", "AA", "F", "A.."])
def test_validate_output_against_mcq_schema_rejects(bad: str) -> None:
    schema = load_schema("mcq_letter")
    with pytest.raises(ValidationError):
        validate_output(bad, schema)


def test_validate_batch_counts_valid_and_invalid() -> None:
    outputs = ["Paris", "", "a b c d", "Tokyo", "Café"]
    records, summary = validate_batch(outputs, short_answer=True)
    assert summary.total == 5
    assert summary.valid == 2  # Paris, Tokyo
    assert summary.invalid == 3
    assert summary.valid_rate == pytest.approx(2 / 5)
    # Per-record mapping preserves order.
    assert [r.valid for r in records] == [True, False, False, True, False]


def test_validate_batch_requires_exactly_one_validator() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        validate_batch(["A"], schema_name="mcq_letter", short_answer=True)
    with pytest.raises(ValidationError, match="exactly one"):
        validate_batch(["A"])
