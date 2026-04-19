"""Tests for the hellaswag_option_index schema + its use from improve.infer."""

from __future__ import annotations

import pytest

from common.errors import ValidationError
from guardrails.validate import load_schema, validate_output


def test_hellaswag_option_index_accepts_valid() -> None:
    schema = load_schema("hellaswag_option_index")
    for i in (0, 1, 2, 3):
        validate_output(i, schema)  # must not raise


@pytest.mark.parametrize("bad", [-1, 4, 5, 10])
def test_hellaswag_option_index_rejects_out_of_range(bad: int) -> None:
    schema = load_schema("hellaswag_option_index")
    with pytest.raises(ValidationError):
        validate_output(bad, schema)


@pytest.mark.parametrize("bad", ["0", 0.5, None, [0], {"x": 0}])
def test_hellaswag_option_index_rejects_non_integer(bad: object) -> None:
    schema = load_schema("hellaswag_option_index")
    with pytest.raises(ValidationError):
        validate_output(bad, schema)  # type: ignore[arg-type]
