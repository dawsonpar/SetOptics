# Contributing to SetOptics

Thanks for considering a contribution. This is a focused project: rally
detection on volleyball video. Anything that improves detection accuracy,
inference speed, the evaluation framework, or the annotation experience
is in scope.

## In scope

- New rally detectors or improvements to existing ones.
- Speed / accuracy / cost optimizations.
- Improvements to `tools/shared/eval/` (new metric domains, better
  matching, clearer reports).
- Annotation UX (Label Studio integration, prompt tuning, multi-model
  support).
- Documentation improvements, especially fixing inaccuracies.
- Tests, fixtures, smoke tests for the CLI entry points.

## Out of scope

- The hosted product (auth, billing, web UI, Cloud Run deploy).
- Volleyball domain extensions that are not detection-adjacent (e.g.,
  full play-by-play tagging is a separate project).
- Closed-data integrations.

## How to propose a change

1. **Open an issue first** for anything non-trivial. Describe the problem
   and the proposed approach before writing code. This saves rework.
2. **Fork and branch.** Branch names like `feat/new-detector-name` or
   `fix/eval-iou-edge-case` are preferred.
3. **Write or update tests** when changing detection or evaluation logic.
   The eval framework is the trust anchor; treat its tests as load-bearing.
4. **Run the smoke test** before opening a PR (see below).
5. **Open a PR** against `main`. Reference the issue. Describe the change
   in 1-3 sentences and the test strategy in 1-3 more.

## Local setup

```bash
git clone https://github.com/<your-fork>/SetOptics.git
cd SetOptics
./setup.sh
source .venv/bin/activate
```

## Smoke test

Before submitting:

```bash
python -c "import setoptics; print(setoptics.__version__)"
python scripts/signal_rally_detector.py --help
python tools/annotation/annotate_sliding_window.py --help
```

If you touched the eval framework:

```bash
python -m pytest tools/shared/eval/ -v
```

If you touched a detector, run it against a short clip end-to-end and
attach the output JSON (or a screenshot of the summary) to the PR.

## Style

- Python: snake_case, 4-space indent, type hints where they help, docstrings
  on public functions only.
- Files cap at 500 lines. Split if you cross this.
- Commit titles: ultra-concise, under 50 chars, sentence form, no trailing
  period. Body wraps at 72 chars. Example:

  ```
  Tighten signal rally peak threshold

  The default 0.45 produced 8% more false positives than 0.55 on the
  indoor benchmark. No regression in recall.
  ```

- No em-dashes or en-dashes in code, comments, or PR text. Use periods,
  parentheses, or colons.

## License

By contributing, you agree that your contributions are licensed under
Apache 2.0, matching the project license.
