"""Tests for ``common.stats``."""

from __future__ import annotations

import pytest

from common.stats import paired_bootstrap


def test_detects_uniform_improvement() -> None:
    baseline = [0] * 100
    improved = [1] * 100
    result = paired_bootstrap(baseline, improved, resamples=1000, seed=0)
    assert result.diff_acc == 1.0
    assert result.ci_low > 0.9
    assert result.ci_high <= 1.0
    assert result.p_value < 0.05
    assert result.significant


def test_no_difference_yields_nonsignificant_result() -> None:
    # Same correctness pattern: no difference.
    baseline = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    improved = list(baseline)
    result = paired_bootstrap(baseline, improved, resamples=2000, seed=0)
    assert result.diff_acc == 0.0
    assert result.ci_low <= 0 <= result.ci_high
    assert not result.significant


def test_detects_regression() -> None:
    baseline = [1] * 100
    improved = [0] * 100
    result = paired_bootstrap(baseline, improved, resamples=1000, seed=0)
    assert result.diff_acc == -1.0
    assert result.ci_high < -0.9
    assert result.p_value < 0.05
    assert result.significant


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap([1, 0, 1], [1, 0], resamples=10)


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap([], [], resamples=10)


def test_non_binary_values_raise() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap([1, 2], [0, 1], resamples=10)


def test_invalid_ci_raises() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap([1, 0], [1, 1], resamples=10, ci=1.5)


def test_seed_reproducibility() -> None:
    baseline = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    improved = [1, 1, 1, 1, 0, 1, 1, 0, 1, 1]
    a = paired_bootstrap(baseline, improved, resamples=500, seed=123)
    b = paired_bootstrap(baseline, improved, resamples=500, seed=123)
    assert a == b
