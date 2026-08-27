---
name: handoff
description:
  Create a concise handoff so another agent session can continue the current
  work.
argument-hint: Describe the work that should continue after this conversation.
disable-model-invocation: true
license: BSD-3-Clause
---

# Handoff

Write a self-contained Markdown handoff to the operating system's temporary
directory, outside the repository. Tailor it to any focus supplied by the user.

Include the objective, current state, decisions and rationale, unresolved
questions, relevant files and symbols, commands and results, working-tree state,
constraints, and concrete next actions. Add a short suggested-skills section
using skills actually available in the destination workspace.

Reference existing issues, plans, commits, and diffs rather than reproducing
them. Exclude secrets, credentials, personal data, and irrelevant conversation.
Report the resulting file path.
