# Magpylib Array API Work Inventory

Public GitHub state checked 2026-08-27. Re-query GitHub before acting on status,
mergeability, checks, or branch divergence. Old pull requests are design and
test evidence, not authoritative versions of current kernels.

## Direct work

| Item                                                                                    | State at audit      | Scope                                                     |
| --------------------------------------------------------------------------------------- | ------------------- | --------------------------------------------------------- |
| [#792](https://github.com/magpylib/magpylib/issues/792)                                 | Open issue          | Original standard-support request and umbrella discussion |
| [#844](https://github.com/magpylib/magpylib/pull/844)                                   | Open, conflicted PR | Initial aggregate core conversion                         |
| [#849](https://github.com/magpylib/magpylib/pull/849)                                   | Open, conflicted PR | Circle core kernel                                        |
| [#850](https://github.com/magpylib/magpylib/pull/850)                                   | Open, conflicted PR | Triangular Mesh core kernel                               |
| [#851](https://github.com/magpylib/magpylib/pull/851)                                   | Open, conflicted PR | Sphere core kernel                                        |
| [#852](https://github.com/magpylib/magpylib/pull/852)                                   | Open, conflicted PR | Triangle core kernel                                      |
| [#853](https://github.com/magpylib/magpylib/pull/853)                                   | Open, conflicted PR | Cylinder Segment core kernel                              |
| [#854](https://github.com/magpylib/magpylib/pull/854)                                   | Open, conflicted PR | Cuboid core kernel                                        |
| [#855](https://github.com/magpylib/magpylib/pull/855)                                   | Open, conflicted PR | Tetrahedron core kernel                                   |
| [#856](https://github.com/magpylib/magpylib/pull/856)                                   | Open, conflicted PR | Polyline core kernel                                      |
| [#857](https://github.com/magpylib/magpylib/pull/857)                                   | Open, conflicted PR | Dipole core kernel                                        |
| [#858](https://github.com/magpylib/magpylib/pull/858)                                   | Open, conflicted PR | Cylinder core kernel                                      |
| [#859](https://github.com/magpylib/magpylib/pull/859)                                   | Closed PR           | Initial core-test submission, superseded by #860          |
| [#860](https://github.com/magpylib/magpylib/pull/860)                                   | Open, conflicted PR | Backend-parametrized core tests                           |
| [#861](https://github.com/magpylib/magpylib/pull/861)                                   | Open, conflicted PR | Combined utilities, kernels, and tests; stale at audit    |
| [#866](https://github.com/magpylib/magpylib/issues/866)                                 | Closed issue        | NumFOCUS grant discussion                                 |
| [NumFOCUS #46](https://github.com/numfocus/small-development-grant-proposals/issues/46) | Not selected        | Public four-stage implementation proposal                 |
| [#981](https://github.com/magpylib/magpylib/issues/981)                                 | Open issue          | Performance and Array API implementation scope            |

At the audit date, all thirteen open Array API pull requests reported
`mergeable_state=dirty`, required review, and had zero checks. The combined
prototype in #861 introduced compatibility helpers, strict/JAX/Dask tests,
promotion logic, and lazy control-flow experiments. It also retained NumPy
coupling, contained unfinished or duplicated Cylinder Segment work, and
substantially predated current `main`.

## Enabling and constraining work

- [#726](https://github.com/magpylib/magpylib/issues/726): dtype and precision
  policy directly controls portable creation, promotion, and tolerances.
- [#883](https://github.com/magpylib/magpylib/issues/883): arbitrary batching
  and broadcasting affect the computation boundary.
- [#704](https://github.com/magpylib/magpylib/issues/704): functional-interface
  design affects where namespace dispatch belongs.
- [#916](https://github.com/magpylib/magpylib/pull/916): merged path-property
  architecture deferred pure broadcasting and preserved the functional interface
  partly for future JAX use.
- [#937](https://github.com/magpylib/magpylib/pull/937): merged SciPy 1.17
  Rotation adaptation changes an integration surface touched by old branches.
- [#941](https://github.com/magpylib/magpylib/pull/941) and
  [#945](https://github.com/magpylib/magpylib/pull/945): merged complete
  elliptic-integral changes supersede overlapping code in old prototypes.
- [#893](https://github.com/magpylib/magpylib/issues/893): proposed Dipole
  far-field fallback introduces value-dependent control-flow and accuracy
  concerns relevant to JIT and lazy execution.
- [#795](https://github.com/magpylib/magpylib/pull/795): merged NumPy 2.0
  support is relevant dependency history, not Array API dispatch support.

## Motivation and exclusions

- [#835](https://github.com/magpylib/magpylib/issues/835),
  [#827](https://github.com/magpylib/magpylib/issues/827), and
  [#579](https://github.com/magpylib/magpylib/issues/579) concern performance or
  alternative compiled cores. They motivate backend work but do not implement
  Array API dispatch.
- [#867](https://github.com/magpylib/magpylib/issues/867) concerns rigid
  transformations and may matter to a later object-level phase.
- [#692](https://github.com/magpylib/magpylib/issues/692) concerns unit-array
  interoperability, a related duck-array problem with a distinct contract.
- [#732](https://github.com/magpylib/magpylib/issues/732) requested finite
  element core tests and was closed as unnecessary. It is test-history context.
- [#974](https://github.com/magpylib/magpylib/issues/974) uses “backend” for
  display testing and is unrelated to numerical Array API backends.
- Historical item `#668`, referenced from #579, returned HTTP 404 during the
  audit. Do not infer its title, type, or outcome.

## Recheck procedure

Search the public repository across issues and pull requests for `array api`,
`array-api`, `array_namespace`, `array-api-compat`, `array-api-strict`, `JAX`,
`CuPy`, `PyTorch`, `Dask`, `GPU`, `backend`, `autodiff`, `device`, `dtype`,
`broadcast`, and `batch`. Inspect comments, timelines, linked pull requests, and
cross-references, then classify new results as direct, enabling, constraining,
motivating, tangential, or unrelated.
