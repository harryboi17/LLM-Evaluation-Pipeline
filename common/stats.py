"""Statistical helpers for benchmark comparison.

Part E of the assignment requires that any improvement be **statistically
significant** (p < 0.05). The right tool for per-example accuracy comparisons
(same items, two systems) is a **paired bootstrap** over the difference in
correctness vectors.

Only the standard library is used here, so this module is importable from any
Part without extra deps.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 42
DEFAULT_CI = 0.95


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Result of a paired-bootstrap accuracy comparison.

    Attributes:
        baseline_acc: Point accuracy of the baseline system.
        improved_acc: Point accuracy of the improved system.
        diff_acc: ``improved_acc - baseline_acc``.
        ci_low: Lower bound of the confidence interval on ``diff_acc``.
        ci_high: Upper bound of the confidence interval on ``diff_acc``.
        p_value: Two-sided bootstrap p-value for ``H0: diff_acc == 0``.
        n: Number of paired examples.
        resamples: Number of bootstrap resamples drawn.
    """

    baseline_acc: float
    improved_acc: float
    diff_acc: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    resamples: int

    @property
    def significant(self) -> bool:
        """Return True iff the 95% CI (or equivalent) excludes zero."""
        return (self.ci_low > 0.0) or (self.ci_high < 0.0)


def paired_bootstrap(
    baseline: Sequence[int],
    improved: Sequence[int],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
) -> BootstrapResult:
    """Compute a paired-bootstrap CI and two-sided p-value for the accuracy delta.

    Args:
        baseline: Per-example correctness (``0`` or ``1``) for the baseline run.
        improved: Per-example correctness (``0`` or ``1``) for the improved run.
            Must be the same length as ``baseline`` and aligned element-wise.
        resamples: Number of bootstrap resamples (default ``10_000``).
        seed: RNG seed for reproducibility.
        ci: Confidence level, e.g. ``0.95`` for a 95% CI.

    Returns:
        A :class:`BootstrapResult` describing the comparison.

    Raises:
        ValueError: If inputs are empty, mis-matched in length, or contain values
            other than 0/1.
    """
    if len(baseline) != len(improved):
        raise ValueError(f"length mismatch: baseline={len(baseline)} improved={len(improved)}")
    n = len(baseline)
    if n == 0:
        raise ValueError("paired_bootstrap requires at least one example")
    if not all(v in (0, 1) for v in baseline) or not all(v in (0, 1) for v in improved):
        raise ValueError("correctness vectors must contain only 0 or 1")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    rng = random.Random(seed)
    diffs = [improved[i] - baseline[i] for i in range(n)]
    observed_diff = sum(diffs) / n

    boot_diffs: list[float] = [0.0] * resamples
    for i in range(resamples):
        # Draw n indices with replacement; average the paired differences.
        total = 0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        boot_diffs[i] = total / n
    boot_diffs.sort()

    alpha = (1.0 - ci) / 2.0
    lo_idx = max(0, int(alpha * resamples))
    hi_idx = min(resamples - 1, int((1.0 - alpha) * resamples))
    ci_low = boot_diffs[lo_idx]
    ci_high = boot_diffs[hi_idx]

    # Two-sided p-value: proportion of bootstrap diffs with the opposite sign
    # (or zero), doubled and capped at 1.0.
    if observed_diff >= 0.0:
        tail = sum(1 for d in boot_diffs if d <= 0.0) / resamples
    else:
        tail = sum(1 for d in boot_diffs if d >= 0.0) / resamples
    p_value = min(1.0, 2.0 * tail)

    return BootstrapResult(
        baseline_acc=sum(baseline) / n,
        improved_acc=sum(improved) / n,
        diff_acc=observed_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        n=n,
        resamples=resamples,
    )


__all__ = [
    "DEFAULT_CI",
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "BootstrapResult",
    "paired_bootstrap",
]
