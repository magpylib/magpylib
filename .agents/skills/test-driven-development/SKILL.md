---
name: test-driven-development
description: >-
  Apply test-driven development to Magpylib changes. Use when implementing a
  feature or bug fix test-first, adding a regression test, or working through a
  red-green-refactor cycle in this repository.
license: BSD-3-Clause
---

# Test-Driven Development

Develop one observable behavior at a time through a red-green-refactor cycle.
Use this skill together with `magpylib-development` for repository commands and
conventions.

## Choose the test boundary

Test behavior through the narrowest stable interface used by real callers. A
public API is preferred, but a package-internal interface is appropriate when
the task intentionally changes that contract. Avoid private implementation
details, incidental call order, and snapshots that do not express a meaningful
contract.

For numerical behavior, obtain expected values independently: use an analytic
result, symmetry, a trusted reference value, or a cross-interface consistency
relation. Do not reproduce the implementation formula in the test.

## Cycle

1. State one behavior and the interface through which it is observed.
1. Add the smallest test that expresses that behavior.
1. Run only that test and confirm that it fails for the intended reason.
1. Implement only the behavior required to satisfy that test.
1. Rerun the same test until it passes without weakening the assertion.
1. Refactor names, structure, or duplication while keeping the test green.
1. Repeat for the next behavior, then run the containing test module and the
   broader validation required by the changed surface.

If the test unexpectedly passes before the implementation changes, prove that it
can detect the defect before continuing. If it fails because of setup, imports,
or unrelated state, repair the test until the failure demonstrates the missing
behavior.

## Magpylib test design

- Match the organization and assertion style of the nearest tests.
- Include array shape and scalar/vectorized behavior when changing field APIs.
- Cover path and collection semantics when they affect the changed contract.
- Treat warnings as part of the behavior; Pytest runs with warnings as errors.
- Keep tolerances physically and numerically justified. Do not widen them to
  hide an unexplained discrepancy.
- Add optional-dependency cases only where the affected boundary requires them.

## Completion

The focused test has been witnessed failing for the intended reason and passing
after the implementation. The containing module passes, relevant lint checks
pass, and public behavior changes are reflected in docs or docstrings.

## Sources

This workflow follows the public red-green-refactor practice described by Kent
Beck in _Test-Driven Development: By Example_, adapted to Magpylib's Pytest
configuration and the public
[Scientific Python testing guide](https://learn.scientific-python.org/development/guides/pytest/).
