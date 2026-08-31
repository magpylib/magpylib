---
name: codebase-design
description: >-
  Design or improve module interfaces and architecture. Use when deciding where
  behavior belongs, reducing a public surface, choosing dependency boundaries,
  improving testability, or comparing alternative designs before implementation.
license: BSD-3-Clause
---

# Codebase Design

Design modules that hide substantial complexity behind a small, coherent
interface. A module may be a function, class, package, or subsystem; judge it by
what callers must understand, not by its physical size.

## Design workflow

1. Identify callers, required capabilities, invariants, failure modes, and
   performance constraints.
1. Locate the behavior and data that change together. Prefer one owning module
   over policies repeated across callers.
1. Sketch at least two materially different interfaces before committing to a
   difficult-to-reverse design.
1. Compare alternatives by caller complexity, hidden implementation detail,
   dependency direction, testability, compatibility, and likely change patterns.
1. Select the smallest interface that represents current requirements without
   speculative extension points.
1. Name the observable tests that can validate the interface independently of
   its implementation.

## Review questions

- Does removing the module make its complexity reappear across callers? If not,
  it may only be forwarding calls.
- Are callers required to coordinate steps or know ordering that the module
  could own?
- Do parameters repeatedly travel together as one domain concept?
- Does a proposed abstraction have more than one real implementation or source
  of variation?
- Can tests use the same interface as production callers?
- Is compatibility preserved, or is the migration explicit and justified?

Prefer dependency injection at genuine variation points, returned results over
hidden side effects, and composition over inheritance that exposes irrelevant
behavior. Do not optimize architectural purity at the expense of numerical
clarity or a stable scientific API.

## Sources

The emphasis on deep modules follows John Ousterhout's _A Philosophy of Software
Design_. The use of seams for testability follows Michael Feathers' _Working
Effectively with Legacy Code_.
