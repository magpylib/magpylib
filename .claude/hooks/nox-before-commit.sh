#!/bin/bash
# PreToolUse hook on Bash: before Claude runs `git commit`, run the checks CI
# runs (`uvx nox`: lint, pylint, tests, doctests). A failure blocks the commit
# and is sent back to Claude (exit 2).
command=$(jq -r '.tool_input.command // empty')
grep -qE '(^|[;&|[:space:]])git([[:space:]]+-[^[:space:]]+([[:space:]]+[^-][^[:space:]]*)?)*[[:space:]]+commit([[:space:]]|$)' <<<"$command" || exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0

out=$(uvx nox 2>&1) && exit 0
{
  echo "uvx nox fails, so the commit is blocked:"
  grep -E '^nox > Session .* (failed|was successful)' <<<"$out"
  tail -40 <<<"$out"
} >&2
exit 2
