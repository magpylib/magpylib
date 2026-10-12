---
name: resolving-merge-conflicts
description: >-
  Resolve an in-progress Git merge, rebase, or cherry-pick conflict by
  preserving the intended behavior of both sides. Use when conflict markers or
  unmerged paths are present.
license: BSD-3-Clause
---

# Resolving Merge Conflicts

Inspect both changes before deciding what the merged file should contain.
Preserve user work and obtain explicit approval before aborting or discarding a
change.

## Process

1. Use `git status` to identify the active operation and every unmerged path.
1. Read the conflicting commits and the surrounding code to understand what each
   change was intended to do.
1. Edit each file to keep one side, the other side, or a combined result. Remove
   all conflict markers.
1. Stage each resolved path with `git add` or `git rm` as appropriate.
1. Review the staged diff and search the repository for remaining markers.
1. Run focused tests for the affected behavior, followed by the relevant
   repository checks.
1. Continue the merge, rebase, or cherry-pick only after validation succeeds.

If the changes are incompatible or the operation targets the wrong base, stop
and explain the available Git options before modifying more files.

## Sources

The procedure follows Git's conflict markers and index workflow as documented by
the public
[GitHub command-line conflict guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/resolving-a-merge-conflict-using-the-command-line)
and
[rebase conflict guide](https://docs.github.com/en/get-started/using-git/resolving-merge-conflicts-after-a-git-rebase).
