---
name: magpylib-array-api-development
description:
  Implement or review Magpylib Array API namespace dispatch, field-kernel ports,
  capability declarations, and backend tests.
license: BSD-3-Clause
---

# Magpylib Array API Development

Add backend portability without changing established NumPy results, numerical
accuracy, or performance unintentionally. Use `magpylib-development` for the
general repository workflow and `primary-source-research` when behavior depends
on a current standard or backend release.

## Establish the contract

Before editing the first kernel, establish these repository-level decisions:

1. Target version: initially use Array API 2024.12, the newest version currently
   implemented by `array-api-compat`. Recheck this when the compatibility
   dependency or supported backend minimums change.
1. Compatibility delivery: add `array-api-compat` as a runtime dependency unless
   maintainers explicitly choose its documented vendoring option. Do not create
   an ad hoc fallback. Add `array-api-strict` only to test dependencies, and add
   `array-api-extra` only when accepted implementation or testing helpers
   require it.
1. Rollout: decide whether dispatch is experimental opt-in or always active and
   identify the stable public boundary that owns that decision.
1. Mixed inputs: choose rejection or document one owning input whose namespace
   and device other inputs follow.
1. Capability matrix: name every initially supported backend, device, dtype,
   JIT, lazy, and autodiff combination and its CI coverage.

Record these decisions in the implementing issue or pull request and user-facing
documentation. Do not begin parallel kernel ports while they remain unresolved.

For each change, state which capabilities it intends to support. Treat each of
these as an independent claim:

- use of standard Array API operations;
- execution with each named array library;
- preservation of a non-CPU device;
- JAX execution under `jax.jit`;
- lazy execution without materialization;
- automatic differentiation;
- numerical accuracy for each supported dtype.

Do not infer one claim from another. Record unsupported combinations explicitly.
Read [references/official-guidance.md](references/official-guidance.md) when
choosing dependencies, dispatch semantics, or a backend test matrix. Consult
[references/magpylib-array-api-work.md](references/magpylib-array-api-work.md)
before reusing or superseding earlier Magpylib work.

## Design the boundary

1. Resolve the namespace once at the public or core computation boundary from
   every relevant array input with `array_api_compat.array_namespace`. The
   compatibility layer normalizes supported NumPy, CuPy, PyTorch, Dask, JAX, and
   other arrays; it is not only a fallback for older NumPy.
1. Define mixed-input behavior deliberately. Reject mixed non-NumPy namespaces
   unless the API has a documented owning input whose namespace and device the
   other inputs follow.
1. Centralize namespace resolution, coercion, dtype promotion, device handling,
   backend detection, and test assertions. Do not add per-kernel variants of the
   same compatibility logic.
1. Pass `xp` into internal kernels when dispatch has already occurred. Do not
   repeatedly rediscover it or add an independent backend selector.
1. Preserve ordinary NumPy behavior whenever all inputs are NumPy arrays. If an
   opt-in gate is selected, test both enabled and disabled behavior.

## Port a computation

1. Capture the NumPy baseline with focused accuracy, shape, dtype, warning, and
   timing checks.
1. Add a failing test for the smallest capability being introduced.
1. Replace array operations with operations from `xp`; retain `np` only at a
   named and tested NumPy-only boundary.
1. Use explicit `dtype=` and, where supported, `device=` for created arrays. Do
   not rely on NumPy's default `float64` or integer width.
1. Follow standard promotion rules. Avoid Python scalars changing promotion or
   forcing values from device arrays.
1. Avoid mutation, conversion to Python scalars, and value-dependent Python
   control flow. These commonly break JAX JIT and lazy arrays even when eager
   execution passes.
1. Preserve the input namespace and device in array outputs. Never perform an
   implicit GPU-to-CPU transfer.
1. Isolate unavoidable compiled NumPy or SciPy calls behind one explicit
   conversion boundary. Declare the resulting function CPU-only unless tested
   native delegation provides wider support.
1. Keep singularity handling, tolerances, broadcasting, and physical field
   conventions identical unless a separately documented change is required.

Start with a simple, purely array-based kernel. Treat complete elliptic
integrals and Cylinder Segment control flow as dedicated algorithmic work rather
than using them as the first migration slice.

## Test each advertised capability

- Run existing NumPy tests first and compare against independently established
  results.
- Use `array-api-strict` to detect non-standard operations and simulated device
  assumptions. It is a test dependency, not a production backend.
- Isolate the backend under test to the function being converted; compute
  references with NumPy and compare through namespace-aware assertions.
- Test JAX support inside `jax.jit`. An eager JAX pass does not establish JIT
  support or differentiability.
- When lazy support is claimed, ensure the test prohibits hidden computation or
  persistence. Otherwise declare lazy execution unsupported.
- Test real libraries and hardware for every documented backend/device pair;
  strict conformance alone cannot establish GPU behavior.
- Add explicit gradient tests before claiming autodiff support.
- Exercise deliberate `float32` and `float64` cases. Do not let backend default
  dtype settings stand in for precision testing.
- Benchmark NumPy before and after each kernel conversion. Investigate material
  regressions rather than accepting portability as justification.

Run the focused backend test immediately after the first implementation edit,
then the corresponding NumPy kernel tests. Widen to the full suite and lint only
after the focused behavior is stable.

## Document and finish

Document capabilities per public function or coherent API group, including
backend, CPU/GPU, dtype, JIT, lazy, and autodiff limitations. A capability must
not be advertised as supported unless an automated test exercises it in CI on
the relevant backend and device. Record manual hardware results as provisional
evidence only; they do not establish published support.

A completed slice has focused portable tests, unchanged NumPy accuracy,
namespace and device assertions, an explicit capability declaration, and a
recorded NumPy benchmark. Do not merge stale Array API branches wholesale;
reapply their useful ideas to current kernels and tests.

## Sources

This workflow is derived solely from public specifications, official project
documentation, public Magpylib issues and pull requests, and the repository's
current source. Detailed links and applicability notes are in the bundled
references.
