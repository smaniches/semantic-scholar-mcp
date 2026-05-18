<!--
Thanks for contributing! A few asks to keep review fast.

PR title format: Conventional Commits (feat:, fix:, docs:, ci:, refactor:, test:, deps:, perf:).
The title becomes a changelog line via release-please, so make it a sentence
that's useful to someone reading the changelog with no other context.
-->

## Summary

<!-- 1–3 sentences. What problem does this solve, why now? -->

## Changes

<!-- Bullet list of what changed. Reference filenames where useful. -->

## Test plan

<!-- How you verified, including any new tests. -->

- [ ] `ruff check src/ tests/`
- [ ] `ruff format --check src/ tests/`
- [ ] `mypy src/`
- [ ] `pytest -q`

## Risk & rollout

<!--
- Backward compatibility: any breaking changes to the tool surface, env vars,
  or import paths?
- Rollout: anything special needed beyond a normal release?
-->

## Related

<!-- Closes #N, refs #M -->
