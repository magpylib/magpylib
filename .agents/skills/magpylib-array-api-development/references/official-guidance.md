# Official Array API Guidance

Checked 2026-08-27. Recheck version-sensitive claims when a new standard is
published, when `array-api-compat` or a supported backend minimum changes, and
before changing dependency minimums or advertised capabilities.

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
- Magpylib's initial implementation target is 2024.12 because that is the newest
  version implemented by `array-api-compat` at the checked date. The newer
  specification remains useful for forward planning, not as the implementation
  contract.
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
  or vendoring. Its wrappers implement Array API 2024.12 at the checked date.
- [array-api-strict](https://data-apis.org/array-api-strict/) deliberately
  exposes non-portable assumptions during testing. Do not require users to adopt
  its array objects.
- [array-api-extra](https://data-apis.org/array-api-extra/) provides portable
  operations outside the standard and testing support for JAX JIT and lazy
  execution. Each helper remains an additional dependency and capability
  decision. Its
  [`lazy_xp_function`](https://data-apis.org/array-api-extra/generated/array_api_extra.testing.lazy_xp_function.html)
  test helper exercises JAX under JIT and guards Dask against unintended
  computation.

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
- [NumPy Array API compatibility](https://numpy.org/doc/stable/reference/array_api.html)
  documents support in NumPy's main namespace. The separate `numpy.array_api`
  module was removed in NumPy 2.0. At the checked date, NumPy 2.3 documents
  compatibility with standard version 2024.12.
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
   transfer must fail clearly rather than happen silently.
1. Array API syntax is only the common vocabulary. Backend availability,
   accelerator execution, JIT, lazy evaluation, and autodiff require separate
   implementation and evidence.
