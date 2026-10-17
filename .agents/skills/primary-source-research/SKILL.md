---
name: primary-source-research
description: >-
  Research technical questions using authoritative primary sources. Use when a
  task depends on current API behavior, specifications, upstream source code,
  release history, or claims that should not rely on memory alone.
license: BSD-3-Clause
---

# Primary-Source Research

Investigate the question against sources that own the facts: official
documentation, specifications, upstream repositories, release notes, and
executable behavior. Use a read-only research subagent for an independent scope
when available, but verify its material claims before relying on them.

## Process

1. Define the exact questions and what evidence would answer each one.
1. Prefer primary sources; use secondary sources only to locate or contrast
   authoritative material.
1. Record versions, dates, and applicability constraints.
1. Trace every non-obvious conclusion to a source or reproducible check.
1. Reconcile contradictions explicitly instead of selecting the convenient
   source.
1. Return a concise synthesis separating established facts, inference, and
   unresolved uncertainty.

Create a repository artifact only when the user requests one or the findings
must support later implementation. In that case, use an agreed project location
and include source links beside the claims they support.
