# Agent instructions

Magpylib computes static magnetic fields of magnets, currents and other sources
from analytical expressions. Follow [CONTRIBUTING.md](CONTRIBUTING.md) for
coding, docstring, changelog and pull request rules; this file adds what an
agent needs on top.

## Layout

- `src/magpylib/` re-exports the public API; the implementation lives in
  `src/magpylib/_src/`.
- `tests/` holds the pytest suite.
- `docs/_pages/` holds the MyST Markdown docs. `docs/_autogen/` is build output;
  never edit it.

## Checks

CI runs all of these; run them before pushing:

- `uvx nox -s tests`, and `uvx nox -s doctests` for the docstring examples.
- `uvx nox -s pylint`, which must rate 10/10. Pylint is not in the pre-commit
  hooks, so nothing else catches it before CI.
- `uvx nox -s lint`: the pre-commit hooks (ruff check and format, prettier for
  Markdown and YAML, codespell, zizmor).

Pytest runs with `filterwarnings = error`: a warning raised in a test fails it.

## Conventions

- Units are SI since v5: metres, tesla, amperes. Magnets take `polarization` in
  T; `magnetization` is in A/m.
- Index names follow SPOTIN (s sources, p path, o observers, t targets, i
  instances), see CONTRIBUTING.md.
