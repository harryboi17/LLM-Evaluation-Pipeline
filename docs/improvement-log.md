# Improvement Log

Chronological journal of every Part E change and its measured impact. One
entry per distinct run; each entry is self-contained so future readers
(and future-me) can reconstruct exactly what was tried, why, and what
happened.

Every row should also exist in `results/result-log.csv` with the matching
`run_id`. The result log is the source of truth for numbers; this file
explains the story behind them.

## Template

```
## <run_id> — <short title>
- **Date:** YYYY-MM-DDTHH:MMZ
- **Commit:** <short sha>
- **Task / limit / seed:** <task> / <limit> / <seed>
- **Hypothesis:** what I thought would help, and why
- **Change:** exact flags / files changed
- **Config:** model, decoding params, prompt template version
- **Result:** metric value and (if not baseline) delta vs last baseline
- **95% CI (vs baseline):** from paired bootstrap, 10k resamples
- **Decision:** keep / revert / iterate, with one-line rationale
```

## Entries

_Runs are appended below in chronological order. The first entry is always a
baseline establishing the starting point._

<!-- Part E entries start here; keep this comment so the file stays parseable. -->
