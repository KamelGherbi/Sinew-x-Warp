---
name: subagent-creator
description: Create focused Sinew sub-agent profiles with clear responsibilities, boundaries, and reporting expectations.
---

# Subagent Creator

Use this to design sub-agent profiles for Sinew Settings → Subagents.

## Required fields

- `id`: stable kebab-case identifier.
- `name`: readable role name.
- `description`: when to delegate to it.
- `prompt`: role, workflow, constraints, output format.
- `model`: pick a capable configured model.
- `enabled`: true unless experimental.

Avoid tool names that do not exist in Sinew.

