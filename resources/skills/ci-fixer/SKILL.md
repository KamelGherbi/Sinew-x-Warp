---
name: ci-fixer
description: Diagnose and fix CI/build/test failures locally and in GitHub workflows.
---

# CI Fixer

Use this for failing builds, tests, lint, typecheck, packaging, or GitHub Actions.

## Workflow

- Identify the exact failing command/job.
- Reproduce locally when possible.
- Separate failures caused by current changes from unrelated failures.
- Fix root cause with minimal changes.
- Re-run the failing command.
- Summarize commands, result, and any remaining CI-only risks.

Never push or rerun remote workflows without user approval.

