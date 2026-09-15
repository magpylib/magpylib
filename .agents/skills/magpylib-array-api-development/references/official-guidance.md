# Official Array API Guidance

Checked 2026-08-27; version claims re-verified 2026-08-31 against installed
packages and upstream changelogs. Recheck version-sensitive claims when a new
standard is published, when `array-api-compat` or a supported backend minimum
changes, and before changing dependency minimums or advertised capabilities.
Verify a version from a changelog or an installed `__array_api_version__`, not
from a project landing page: several landing pages were months out of date at
the checked date.

## Normative standard

- [Python Array API standard 2025.12](https://data-apis.org/array-api/latest/)
  is the latest published specification at the checked date. The written
  specification is authoritative; its verification suite does not yet cover
  every requirement.
- [Future API evolution](https://data-apis.org/array-api/latest/future_API_evolution.html)
  defines date-based versions and requires a compliant namespace to expose
  `__array_api_version__`.
- Arrays expose their namespace through `__array_namespace__(api_version=...)`.
  Consumers should negotiate a version they support instead of assuming the
  newest standard is implemented by every backend.
- 2025.12 is the revision that current tooling implements. `array-api-compat`
  1.14.0 (2026-02-26) targets it and sets `__array_api_version__` to `2025.12`
  for every wrapped namespace, `array-api-strict` 2.6 defaults to it, and NumPy
  2.5 reports it. Implementing an older revision is a maintainer decision,
  recorded in the implementing issue and pinned in tests through
  `array_api_strict.set_array_api_strict_flags(api_version=...)`.
- [Lazy versus eager execution](https://data-apis.org/array-api/latest/design_topics/lazy_eager.html)
  explains why Python truth testing and scalar conversion can force execution or
  fail for lazy implementations.
- [Extension namespaces](https://data-apis.org/array-api/latest/extensions/index.html)
  such as `linalg` and `fft` are optional coherent units. Feature-detect the
  namespace before relying on one.

## Consumer compatibility tools

- [array-api-compat](https://data-apis.org/array-api-compat/) provides namespace
  discovery and compatibility wrappers for NumPy, CuPy, PyTorch, Dask, JAX, and
  other established libraries. It is suitable for runtime use by an
  array-consuming library and officially supports installation as a dependency
  or vendoring. Its wrappers implement Array API 2025.12 from 1.14.0 onward; the
  project landing page still claims 2024.12 and is stale.
- [array-api-strict](https://data-apis.org/array-api-strict/) deliberately
  exposes non-portable assumptions during testing. Do not require users to adopt
  its array objects. Its `boolean_indexing` and `data_dependent_shapes` flags,
  and its `api_version` flag, are set through `set_array_api_strict_flags(...)`.
- [array-api-extra](https://data-apis.org/array-api-extra/) provides portable
  operations outside the standard and testing support for JAX JIT and lazy
  execution. `at` supplies out-of-place indexed assignment, `apply_where`
  supplies value-dependent branches that survive the JIT, and `lazy_apply` wraps
  an unavoidable Python callback. Its
  [`lazy_xp_function`](https://data-apis.org/array-api-extra/generated/array_api_extra.testing.lazy_xp_function.html)
  test helper exercises JAX under JIT and guards Dask against unintended
  computation, and takes effect only when the test or a fixture calls
  `patch_lazy_xp_functions`. Its
  [`testing`](https://data-apis.org/array-api-extra/api-testing.html) module
  provides `assert_close`, `assert_close_nulp`, `assert_equal` and `assert_less`
  as namespace-aware replacements for the `numpy.testing` assertions; they check
  namespace, dtype, shape and device by default, so both arguments must already
  be in the namespace under test.

## Downstream implementation precedents

- [SciPy's Array API developer guide](https://scipy.github.io/devdocs/dev/api-dev/array_api.html)
  is the most complete public procedure for a scientific array consumer. It
  resolves `xp` from all inputs, validates and coerces through centralized
  helpers, converts around compiled calls explicitly, declares per-function
  capabilities, and requires tests for every advertised capability.
- SciPy tests NumPy, strict, PyTorch, JAX, and Dask on CPU and CuPy, PyTorch,
  and JAX on available GPUs. JAX functions are tested under `jax.jit`; lazy Dask
  tests prevent materialization. SciPy currently permits Dask to be declared
  unsupported where structural barriers make support disproportionately costly.
- SciPy has merged Array API support for `scipy.spatial.transform.Rotation`
  ([scipy#23249](https://github.com/scipy/scipy/pull/23249)), which removes the
  object-level blocker recorded in Magpylib #792. It landed in SciPy 1.17, which
  is this repository's declared minimum, so `Rotation` is no longer
  automatically a NumPy-only boundary.
- [NumPy Array API compatibility](https://numpy.org/doc/stable/reference/array_api.html)
  documents support in NumPy's main namespace. The separate `numpy.array_api`
  module was removed in NumPy 2.0. At the checked date NumPy 2.5 reports
  `__array_api_version__` of `2025.12`, while that documentation page still
  states 2024.12.
- [NEP 56](https://numpy.org/neps/nep-0056-array-api-main-namespace.html)
  explains the main-namespace design, predictable promotion, device-aware array
  creation, and why a strict standalone array type is better suited to testing
  than normal downstream use.
- [Scikit-learn's Array API guide](https://scikit-learn.org/stable/modules/array_api.html)
  demonstrates experimental opt-in dispatch, owning-input conversion policies,
  namespace/device-preserving outputs, strict checks, and real GPU validation.

## Implications for Magpylib

1. Before porting kernels, choose and document the standard version, dependency
   or vendoring strategy, rollout gate, mixed-input semantics, and CI-backed
   capability matrix.
1. Namespace dispatch belongs at a stable computation boundary, not in a global
   mutable backend setting.
1. If array arguments already identify namespace and device, separate `xp=` or
   `device=` public keywords add ambiguity and should normally be avoided.
1. Creation without an array input may require explicit keyword-only namespace,
   device, and dtype parameters, but this is a separate API design decision.
1. Calls into NumPy/SciPy compiled implementations are CPU boundaries. Device
   transfer must fail clearly rather than happen silently. Check each SciPy
   surface individually: `Rotation` is no longer one of them.
1. Array API syntax is only the common vocabulary. Backend availability,
   accelerator execution, JIT, lazy evaluation, and autodiff require separate
   implementation and evidence.
