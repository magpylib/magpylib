---
name: magpylib-array-api-development
description: >-
  Implement or review Magpylib Array API namespace dispatch, field-kernel ports,
  capability declarations, and backend tests. Use when converting a core field
  kernel to an `xp` namespace, choosing an Array API dependency or standard
  version, adding backend-parametrized tests, or reviewing an Array API pull
  request in this repository.
license: BSD-3-Clause
---

# Magpylib Array API Development

Add backend portability without changing established NumPy results, numerical
accuracy, or performance unintentionally. Use `magpylib-development` for the
general repository workflow and `primary-source-research` when behavior depends
on a current standard or backend release.

## Establish the contract

Before editing the first kernel, establish these repository-level decisions:

1. Target version: choose the standard version to implement. Current tooling
   implements 2025.12: `array-api-compat` 1.14.0 sets `__array_api_version__` to
   `2025.12` for every wrapped namespace, `array-api-strict` 2.6 defaults to it,
   and NumPy 2.5 reports it. Take this from the compatibility changelog and the
   installed `__array_api_version__`; project landing pages lag by months.
   Implementing an older revision is a legitimate conservative choice, but then
   pin it in tests with
   `array_api_strict.set_array_api_strict_flags(api_version=...)`, because the
   default follows the newest revision.
1. Compatibility delivery: choose whether `array-api-compat` is a runtime
   dependency or vendored through its documented option; a runtime dependency is
   the recommended default. Do not create an ad hoc fallback. Keep
   `array-api-strict` to test dependencies only. `array-api-extra` is not
   optional under this workflow: its `testing` assertions, `at`, and
   `apply_where` are required below, so add it as a test and runtime dependency
   or replace those rules deliberately.
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
before reusing or superseding earlier Magpylib work; issue #981 is the current
design source for measured cost, scope, and sequencing in this repository.

## Design the boundary

Current state: the core functions accept a foreign array, coerce it to host, and
return NumPy with no error or warning, and an `array_api_strict` input fails on
array iteration. Raising an explicit error for non-NumPy input is an improvement
worth making before, and independently of, any kernel port.

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
   `_cel_iter_scalarvector(..., xp=, any_=)` in
   `src/magpylib/_src/fields/special_cel.py` is the established form of this
   seam in this repository; follow it rather than inventing a second one.
1. Preserve ordinary NumPy behavior whenever all inputs are NumPy arrays. If an
   opt-in gate is selected, test both enabled and disabled behavior.

## Port a computation

1. Capture the NumPy baseline with focused accuracy, shape, dtype, warning, and
   timing checks.
1. Add a failing test for the smallest capability being introduced.
1. Replace array operations with operations from `xp`; retain `np` only at a
   named and tested NumPy-only boundary.
1. Convert indexed assignment with `array_api_extra.at`, as
   `xpx.at(arr)[idx].set(value)`. This is the most common single operation in
   the port: `_src/fields/` holds 389 indexed assignments, of which 160 are
   plain slice or integer targets that convert mechanically and 229 are mask or
   fancy-index targets. Use `array_api_extra.apply_where` for value-dependent
   branches, and `array_api_extra.lazy_apply` only where a Python callback is
   unavoidable.
1. Use explicit `dtype=` and, where supported, `device=` for created arrays. Do
   not rely on NumPy's default `float64` or integer width.
1. Follow standard promotion rules. Avoid Python scalars changing promotion or
   forcing values from device arrays.
1. Avoid mutation, conversion to Python scalars, and value-dependent Python
   control flow. These commonly break JAX JIT and lazy arrays even when eager
   execution passes.
1. Where an iteration count is data-dependent, keep the eager early exit and
   degrade only lazy backends to a fixed trip count. The Bulirsch `cel` loop
   converges in two to five iterations depending on geometry, so a worst-case
   fixed count costs six to sixteen times more work on the elliptic kernels, and
   is worst in the far field that users sweep over.
1. Preserve the input namespace and device in array outputs. Never perform an
   implicit GPU-to-CPU transfer.
1. Isolate unavoidable compiled NumPy or SciPy calls behind one explicit
   conversion boundary. Declare the resulting function CPU-only unless tested
   native delegation provides wider support.
1. Keep singularity handling, tolerances, broadcasting, and physical field
   conventions identical unless a separately documented change is required.

Start with a simple, purely array-based kernel. Sequence the rest by the kind of
indexed assignment rather than by raw count: `field_BH_cylinder_segment.py` has
the largest total but converts mostly mechanically, while `special_el3.py` is
almost entirely data-dependent mask assignment and is the genuinely hard file.
Treat it and the complete elliptic integrals as dedicated algorithmic work
rather than as an early migration slice.

## Test each advertised capability

- Run existing NumPy tests first and compare against independently established
  results.
- Use `array-api-strict` to detect non-standard operations and simulated device
  assumptions. Disable its `boolean_indexing` and `data_dependent_shapes` flags
  to surface the mask-indexing patterns that will not port. It is a test
  dependency, not a production backend.
- Isolate the backend under test to the function being converted; compute
  references with NumPy, then convert them into the namespace under test before
  comparing. Use `array_api_extra.testing` (`assert_close`, `assert_close_nulp`,
  `assert_equal`, `assert_less`) rather than `numpy.testing`, which forces the
  comparison onto NumPy and defeats the purpose of the conversion. These
  assertions check namespace, dtype, shape, and device by default, so a raw
  NumPy reference passed as `desired` fails on namespace mismatch rather than on
  values.
- Test JAX support inside `jax.jit`. An eager JAX pass does not establish JIT
  support or differentiability.
- When lazy support is claimed, tag the function with
  `array_api_extra.testing.lazy_xp_function` and call `patch_lazy_xp_functions`
  from the test or a fixture; the tag does nothing without it. It jits the
  function under JAX and raises when Dask materializes the graph. Otherwise
  declare lazy execution unsupported.
- Test real libraries and hardware for every documented backend/device pair;
  strict conformance alone cannot establish GPU behavior.
- Add explicit gradient tests before claiming autodiff support.
- Exercise deliberate `float32` and `float64` cases. Do not let backend default
  dtype settings stand in for precision testing.
- Benchmark NumPy before and after each kernel conversion. Investigate material
  regressions rather than accepting portability as justification. Rewriting
  masks as `apply_where` or `at` can regress the default NumPy users, who are
  the majority; a NumPy-path performance check in CI is worth adding before the
  first kernel PR.

Run the focused backend test immediately after the first implementation edit,
then the corresponding NumPy kernel tests. Widen to the full suite and lint only
after the focused behavior is stable.

## Document and finish

Document capabilities per public function or coherent API group, including
backend, CPU/GPU, dtype, JIT, lazy, and autodiff limitations. A capability must
not be advertised as supported unless an automated test exercises it in CI on
the relevant backend and device. CI currently runs CPU only, on Ubuntu, Windows,
and macOS, so GPU support cannot be advertised until a GPU runner exists. Manual
hardware results belong in a separate provisional tier that is labeled as such
in the documentation; they do not establish published support, and they are not
a reason to weaken the CI rule.

A completed slice has focused portable tests, unchanged NumPy accuracy,
namespace and device assertions, an explicit capability declaration, and a
recorded NumPy benchmark. Do not merge stale Array API branches wholesale;
reapply their useful ideas to current kernels and tests.

## Sources

This workflow is derived solely from public specifications, official project
documentation, public Magpylib issues and pull requests, and the repository's
current source. Detailed links and applicability notes are in the bundled
references.
