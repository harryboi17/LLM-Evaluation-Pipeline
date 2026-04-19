"""Output validation and determinism checks (Part D).

Two guardrails are provided:

1. **Structural validation** — JSON-Schema + regex validators for task outputs.
   Used both programmatically (``validate_output``) and from the CLI
   (``python -m guardrails.validate``) to score a JSONL of model outputs
   against a named schema.
2. **Determinism verification** — :func:`verify_determinism` fires the same
   prompt ``n`` times through :class:`common.vllm_client.VLLMClient` with
   ``temperature=0``, ``top_p=1``, and a fixed ``seed``; asserts every
   completion is byte-identical.

Both pieces are used as the reliability floor for the rest of the pipeline
(Part B eval, Part E improvement loop): if outputs aren't shaped right or
the server isn't deterministic under the chosen seeds, downstream numbers
are meaningless.

Known residual non-determinism on real hardware — documented in
:file:`guardrails/README.md`:

* Mixed-precision reductions in the vLLM GEMM kernels can flip the last ULP
  of logits on different batch shapes; at ``temperature=0`` this is rare but
  visible on long generations.
* Server-side scheduling orders concurrent requests non-deterministically.
  The same prompt on the same server instance, run alone, is deterministic;
  the same prompt co-scheduled with unrelated traffic may differ in its
  sampled token when temperature > 0.
* CUDA kernel autotuning can pick different implementations on first vs
  subsequent calls; warming up the server removes the one-shot asymmetry.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

from common.config import get_settings
from common.errors import DeterminismError, ValidationError
from common.logging import get_logger
from common.vllm_client import VLLMClient

log = get_logger(__name__)

_SCHEMAS_DIR = Path(__file__).parent / "schemas"


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    """One validation outcome for a single output string.

    Attributes:
        index: Zero-based index into the input sequence.
        output: The raw output string that was validated.
        valid: Whether validation passed.
        error: Error message when ``valid`` is ``False``; empty otherwise.
    """

    index: int
    output: str
    valid: bool
    error: str = ""


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """Aggregate results from a batch validation run."""

    total: int
    valid: int
    invalid: int

    @property
    def valid_rate(self) -> float:
        """Return the fraction of valid outputs (0.0 if the batch was empty)."""
        return self.valid / self.total if self.total else 0.0


def load_schema(name: str) -> dict[str, Any]:
    """Load a JSON schema from ``guardrails/schemas/<name>.json``.

    Args:
        name: Schema basename (without ``.json``).

    Returns:
        The parsed schema dict.

    Raises:
        ValidationError: If the schema file is missing or unparseable.
    """
    path = _SCHEMAS_DIR / f"{name}.json"
    if not path.exists():
        raise ValidationError(f"no such schema: {name} (looked at {path})")
    try:
        schema: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"schema {name} is invalid json: {exc}") from exc
    return schema


def validate_output(output: str, schema: dict[str, Any]) -> None:
    """Validate ``output`` against ``schema``.

    Raises:
        ValidationError: If the output fails the schema.
    """
    try:
        jsonschema.validate(output, schema)
    except jsonschema.ValidationError as exc:
        raise ValidationError(str(exc.message)) from exc


def validate_short_answer(output: str) -> None:
    """Fast-path validator for short-answer outputs (the ``custom_qa`` task).

    Enforces:

    * 1-3 whitespace tokens
    * printable ASCII only
    * no leading / trailing whitespace
    * no trailing ``.`` / ``?`` / ``!`` (we strip these during eval; stripping
      them here too keeps the boundary tight)

    Raises:
        ValidationError: On any violation, with a short explanation.
    """
    if output != output.strip():
        raise ValidationError("output has leading/trailing whitespace")
    if not output:
        raise ValidationError("output is empty")
    if not output.isascii():
        raise ValidationError("output contains non-ASCII characters")
    tokens = output.split()
    if len(tokens) > 3:
        raise ValidationError(f"output has {len(tokens)} tokens, max is 3")
    if re.search(r"[.?!]$", output):
        raise ValidationError("output ends in punctuation (.?!)")


def validate_batch(
    outputs: Iterable[str],
    schema_name: str | None = None,
    *,
    short_answer: bool = False,
) -> tuple[list[ValidationRecord], ValidationSummary]:
    """Validate many outputs, returning per-item records and an aggregate.

    Exactly one of ``schema_name`` or ``short_answer=True`` must be provided.

    Args:
        outputs: Iterable of output strings.
        schema_name: Optional name of a JSON schema under ``schemas/``.
        short_answer: If ``True``, use :func:`validate_short_answer`.

    Returns:
        A tuple of (per-item records, aggregate summary).

    Raises:
        ValidationError: If the arguments are inconsistent.
    """
    if bool(schema_name) == bool(short_answer):
        raise ValidationError(
            "validate_batch: exactly one of schema_name / short_answer must be set"
        )
    schema: dict[str, Any] | None = load_schema(schema_name) if schema_name else None

    records: list[ValidationRecord] = []
    valid_count = 0
    for i, out in enumerate(outputs):
        try:
            if short_answer:
                validate_short_answer(out)
            else:
                assert schema is not None  # narrowing for mypy
                validate_output(out, schema)
            records.append(ValidationRecord(index=i, output=out, valid=True))
            valid_count += 1
        except ValidationError as exc:
            records.append(
                ValidationRecord(index=i, output=out, valid=False, error=str(exc))
            )
    summary = ValidationSummary(
        total=len(records),
        valid=valid_count,
        invalid=len(records) - valid_count,
    )
    log.info(
        "validation_batch_complete",
        total=summary.total,
        valid=summary.valid,
        invalid=summary.invalid,
        valid_rate=round(summary.valid_rate, 4),
    )
    return records, summary


@dataclass(frozen=True, slots=True)
class DeterminismReport:
    """Output of :func:`verify_determinism`.

    Attributes:
        prompt: The prompt used.
        n_runs: Number of repeated generations.
        identical: ``True`` iff all completions matched exactly.
        completions: The list of completion strings in run order.
        first_divergence_at: Index of the first non-matching run, or ``None``
            if all matched. Useful when ``n_runs`` is large.
    """

    prompt: str
    n_runs: int
    identical: bool
    completions: list[str]
    first_divergence_at: int | None


async def verify_determinism(
    prompt: str,
    *,
    n_runs: int = 5,
    max_tokens: int = 64,
    seed: int | None = None,
) -> DeterminismReport:
    """Fire ``prompt`` ``n_runs`` times at ``temperature=0`` and report.

    Args:
        prompt: Prompt text.
        n_runs: Number of repeats. Must be >= 2.
        max_tokens: Completion length cap per run.
        seed: Optional RNG seed forwarded to vLLM. Defaults to ``Settings.seed``.

    Returns:
        A :class:`DeterminismReport` summarising the runs.

    Raises:
        DeterminismError: If ``n_runs < 2``.
    """
    if n_runs < 2:
        raise DeterminismError("verify_determinism requires n_runs >= 2")

    seed_value = seed if seed is not None else get_settings().seed
    completions: list[str] = []
    async with VLLMClient() as client:
        for _ in range(n_runs):
            result = await client.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
                seed=seed_value,
            )
            completions.append(result.text)

    first_div: int | None = None
    reference = completions[0]
    for i, text in enumerate(completions[1:], start=1):
        if text != reference:
            first_div = i
            break

    return DeterminismReport(
        prompt=prompt,
        n_runs=n_runs,
        identical=first_div is None,
        completions=completions,
        first_divergence_at=first_div,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI argv."""
    p = argparse.ArgumentParser(description="Guardrails: validate outputs / verify determinism.")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="Validate a JSONL of outputs against a schema.")
    v.add_argument("input", help="Path to a JSONL with {'output': '...'} per line.")
    v.add_argument("--schema", default=None, help="Schema basename under guardrails/schemas/.")
    v.add_argument(
        "--short-answer",
        action="store_true",
        help="Use the built-in short-answer validator instead of a JSON schema.",
    )
    v.add_argument("--report", default=None, help="Optional output path for a JSON report.")

    d = sub.add_parser("determinism", help="Repeat a prompt and assert identical completions.")
    d.add_argument("prompt")
    d.add_argument("--n-runs", type=int, default=5, dest="n_runs")
    d.add_argument("--max-tokens", type=int, default=64, dest="max_tokens")
    d.add_argument("--seed", type=int, default=None)
    d.add_argument("--report", default=None, help="Optional output path for a JSON report.")

    return p.parse_args(argv)


def _cmd_validate(args: argparse.Namespace) -> int:
    outputs: list[str] = []
    for raw_line in Path(args.input).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"bad JSONL line: {exc}") from exc
        outputs.append(str(row.get("output", "")))
    records, summary = validate_batch(
        outputs,
        schema_name=args.schema,
        short_answer=bool(args.short_answer),
    )
    report = {
        "input": args.input,
        "schema": args.schema,
        "short_answer": bool(args.short_answer),
        "summary": {
            "total": summary.total,
            "valid": summary.valid,
            "invalid": summary.invalid,
            "valid_rate": summary.valid_rate,
        },
        "records": [
            {"index": r.index, "valid": r.valid, "error": r.error, "output": r.output}
            for r in records
            if not r.valid
        ][:20],  # cap invalid rows in the report
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))
    sys.stdout.write(
        json.dumps({"valid": summary.valid, "invalid": summary.invalid}) + "\n"
    )
    return 0 if summary.invalid == 0 else 1


def _cmd_determinism(args: argparse.Namespace) -> int:
    report = asyncio.run(
        verify_determinism(
            args.prompt,
            n_runs=args.n_runs,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
    )
    payload = {
        "prompt": report.prompt,
        "n_runs": report.n_runs,
        "identical": report.identical,
        "first_divergence_at": report.first_divergence_at,
        "completions": report.completions,
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(payload, indent=2))
    sys.stdout.write(
        json.dumps({"identical": report.identical, "first_divergence_at": report.first_divergence_at})
        + "\n"
    )
    return 0 if report.identical else 1


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m guardrails.validate``.

    Returns:
        ``0`` on success, ``1`` if any validation / determinism check failed.
    """
    args = _parse_args(argv)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "determinism":
        return _cmd_determinism(args)
    raise ValidationError(f"unknown command: {args.command}")  # argparse should prevent this


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DeterminismReport",
    "ValidationRecord",
    "ValidationSummary",
    "load_schema",
    "main",
    "validate_batch",
    "validate_output",
    "validate_short_answer",
    "verify_determinism",
]
