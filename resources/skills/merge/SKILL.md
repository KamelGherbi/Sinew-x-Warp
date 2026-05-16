---
name: merge
description: Merge or sync branches/upstream safely while preserving custom changes.
---

# Merge

Use for updating from upstream, merging branches, or resolving conflicts.

## Safety rules

- Check clean working tree first.
- Create a backup branch before risky syncs.
- Fetch before merging.
- Resolve conflicts by preserving intentional local customization.
- Run build/checks after merge.
- Do not push until user approves.

