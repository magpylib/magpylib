#!/bin/bash
# PreToolUse hook on Bash: before Claude runs `git commit`, run the checks CI
# runs (`uvx nox`: lint, pylint, tests, doctests) in the checkout the commit is
# made in. A failure blocks the commit and is sent back to Claude (exit 2).
input=$(cat)
command=$(jq -r '.tool_input.command // empty' <<<"$input")
commit='git([[:space:]]+-[^[:space:]]+([[:space:]]+[^-][^[:space:]]*)?)*[[:space:]]+commit([[:space:]]|$)'
grep -qE "(^|[;&|[:space:]])$commit" <<<"$command" || exit 0

# The commit can be made in another worktree than the session's: with
# `git -C <dir> commit`, or after a `cd <dir>` earlier in the command.
before=$(sed -E "s/$commit.*//" <<<"$command" | tr '\n' ' ')
dir=$(grep -oE "git[[:space:]]+-C[[:space:]]+[^[:space:]]+[[:space:]]+commit" <<<"$command" | head -1 | awk '{print $3}')
[ -n "$dir" ] || dir=$(grep -oE "(^|[;&|[:space:]])cd[[:space:]]+[^[:space:];&|]+" <<<"$before" | tail -1 | awk '{print $NF}')
[ -n "$dir" ] || dir=$(jq -r '.cwd // empty' <<<"$input")
dir=${dir//[\"\']/}
cd "${dir/#\~/$HOME}" 2>/dev/null || cd "$CLAUDE_PROJECT_DIR" || exit 0
cd "$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0

out=$(uvx nox 2>&1) && exit 0
{
  echo "uvx nox fails in $PWD, so the commit is blocked:"
  grep -E '^nox > Session .* (failed|was successful)' <<<"$out"
  tail -40 <<<"$out"
} >&2
exit 2
