---
name: apex
description: Sinew-native APEX workflow for structured implementation: Analyze, Plan, Execute, eXamine, Validate, Verify. Use for non-trivial features, bug fixes, refactors, and changes that benefit from disciplined planning or agent teams.
argument-hint: "[-a] [-x] [-s] [-t] [-v] [-e] [-b] [-pr] [-i] [-k] [-m] [-r <task-id>] <task description>"
---

# APEX for Sinew

APEX means **Analyze → Plan → Execute → eXamine → Validate/Verify**. This version is adapted for Sinew tools and must not assume Claude Code-only tools.

## Tool mapping

- `TaskCreate` / `TaskUpdate` / `TodoWrite` → use `ToDoList`.
- `TeamCreate` / `Task tool` → use `TeamRun`, `TeamStatus`, `TeamStop`.
- `SendMessage` is available inside Sinew agent teams.
- `Edit` / `Write` → use `apply_patch` only for file writes.
- `Read`, `Grep`, `Glob`, `bash`, `WebFetch`, `WebSearch` → use the matching Sinew tools.
- Save optional workflow artifacts under `.sinew/output/apex/`, not `.claude/output/apex/`.

## Flags

- `-a`, `--auto`: proceed with fewer confirmations when the task is clear.
- `-x`, `--examine`: run adversarial review before finalizing.
- `-s`, `--save`: save analysis/plan/results under `.sinew/output/apex/<task-id>/`.
- `-t`, `--test`: include test planning/creation when appropriate.
- `-v`, `--verify`: run app-level or manual verification steps when practical.
- `-e`, `--economy`: avoid agent teams and keep context usage low.
- `-b`, `--branch`: check branch safety before editing.
- `-pr`, `--pull-request`: prepare PR notes; only create/push if the user explicitly approves.
- `-i`, `--interactive`: ask the user to choose workflow options.
- `-k`, `--tasks`: break work into explicit dependency-aware tasks.
- `-m`, `--teams`: use Sinew Agent Team for independent parallel workstreams.
- `-r <task-id>`, `--resume <task-id>`: resume from `.sinew/output/apex/<task-id>/` if present.

## Workflow

### 0. Initialize

1. Parse flags and the task description.
2. If `-i`, use `Question` to confirm options.
3. If `-b` or `-pr`, inspect Git status/branch before editing.
4. If `-s`, create a task id and plan output path `.sinew/output/apex/<task-id>/`.
5. Create a concise `ToDoList` for the workflow.

### 1. Analyze

Build context before deciding. Read relevant files. Use `Glob`/`Grep` before broad `bash`. Identify:

- entry points and nearby patterns;
- data/model/state flow;
- UI and backend boundaries;
- tests/checks/build commands;
- risks, unknowns, and acceptance criteria.

For `-m`, launch an agent team only when analysis can be parallelized. Useful profiles: `code-explorer`, `code-architect`, `websearch`, `frontend-design`, `verifier`.

### 2. Plan

Produce a concrete implementation plan:

- files to create/modify;
- changes per file;
- dependency order;
- migration/backward compatibility concerns;
- validation commands.

For `-k` or `-m`, create a dependency-aware task board. With `TeamRun`, pass `tasks` directly and assign owners where useful.

### 3. Execute

Implement incrementally. Rules:

- use `apply_patch` for all file modifications;
- keep user changes and custom fork behavior intact;
- keep at most one `ToDoList` task in progress;
- do not broaden scope without user approval;
- prefer existing patterns over new abstractions.

For `-m`, coordinate via `TeamRun` and let teammates own independent slices. The main agent remains responsible for final merge, conflict resolution, checks, and user summary.

### 4. Validate

Run the smallest meaningful checks first, then broader checks if needed:

- typecheck/build;
- Rust/cargo checks for Tauri/backend changes;
- targeted tests;
- diff/whitespace checks.

Fix failures caused by your changes. If unrelated failures exist, report them clearly.

### 5. eXamine

If `-x`, perform adversarial review. Use `code-reviewer` or self-review. Look for:

- regressions;
- broken edge cases;
- async/race issues;
- UX inconsistencies;
- security/data-loss risks;
- update/merge safety issues.

Resolve findings or explicitly document why they are accepted.

### 6. Tests and Verify

If `-t`, create or update tests only when they fit the repo. If `-v`, run app-level verification or give exact manual verification steps.

### 7. Finish

Summarize:

- what changed;
- files changed;
- checks run and result;
- remaining risks;
- whether commit/push/release is recommended.

Never push, create PRs, or install updates without explicit user approval.

