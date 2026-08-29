---
name: magpylib-code-review
description: >-
  Review Magpylib changes for correctness, regressions, scope, tests, and
  repository conventions. Use when reviewing a branch, pull request, commit, or
  working-tree diff in this repository.
license: BSD-3-Clause
---

# Magpylib Code Review

Review behavior and risk before style. Use `magpylib-development` for the
repository's implementation and validation conventions.

## Establish the review target

1. Resolve the comparison point supplied by the user. For a branch or pull
   request, compare against its merge base; for working-tree changes, include
   staged and unstaged changes.
1. Read the commit list and changed-file summary before individual hunks.
1. Find the originating GitHub issue, pull request description, or stated user
   request. If no specification exists, say so and review observable behavior
   without inventing requirements.

## Review criteria

Inspect the complete diff using parallel read-only agents for independent areas
when that improves coverage:

- Correctness: wrong results, broken invariants, numerical instability,
  singularities, shape errors, unit mistakes, and path or collection semantics.
- Compatibility: public API, typing, warnings, optional dependencies, supported
  Python versions, and serialized or displayed output.
- Requirements: missing behavior, partial acceptance criteria, and scope creep.
- Tests: meaningful regression coverage, independently derived expectations,
  justified tolerances, and important scalar/vectorized or boundary cases.
- Maintainability: unnecessary complexity, duplication, misleading names, and
  changes made outside the abstraction that owns the behavior.
- Documentation: current signatures, SI units, examples, links, and user-facing
  behavior reflected in docs or docstrings.

Treat `CONTRIBUTING.md`, `.github/CONTRIBUTING.md`, `pyproject.toml`, and nearby
tests as authoritative. Tooling enforces formatting; do not report formatter
preferences as review findings.

## Verify findings

Trace each suspected defect through the controlling code and callers. Run the
narrowest test or reproducer that can distinguish a real problem from a concern
when execution is practical. Do not claim a bug from pattern matching alone.

## Report

Lead with findings ordered by severity. For each finding, give a precise file
location, the failure mode, its impact, and the smallest credible correction.
Then list open questions and test gaps. If no defects are found, say so and
state residual risks or validation not performed.

## Sources

This workflow is grounded in Magpylib's contribution guides. Its review criteria
follow the public
[Google Engineering Practices review guide](https://google.github.io/eng-practices/review/).
