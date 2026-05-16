---
name: debug-code
description: Systematically diagnose and fix bugs using reproduction, hypothesis narrowing, instrumentation, and targeted validation.
---

# Debug Code

Use this when behavior is broken or uncertain.

## Workflow

1. Reproduce or define the failure precisely.
2. Inspect logs/errors and relevant code paths.
3. Form hypotheses and eliminate them one by one.
4. Add temporary instrumentation only when useful, then remove it.
5. Fix the smallest root cause.
6. Validate with targeted commands or manual steps.

Do not shotgun changes. Explain the root cause when found.

