# Agent instructions

Magpylib computes static magnetic fields of magnets, currents and other sources
from analytical expressions. Follow [CONTRIBUTING.md](CONTRIBUTING.md) for
coding, docstring, changelog and pull request rules, and
[.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) to set up the environment and
the pre-commit hooks; this file adds what an agent needs on top.

## Layout

- `src/magpylib/` re-exports the public API; the implementation lives in
  `src/magpylib/_src/`.
- `tests/` holds the pytest suite. A pre-commit hook requires its modules,
  except `conftest.py`, to be named `test_*.py`.
- `docs/` holds the Sphinx docs: `index.md`, `conf.py`, and the MyST Markdown
  pages in `docs/_pages/`. `docs/_autogen/` and `docs/_build/` are generated;
  never edit them.

## Checks

Run `uvx nox` before pushing. Its default sessions are the checks CI runs:

- `lint`: the hooks in `.pre-commit-config.yaml`. Prettier skips `docs/`, where
  it would break MyST colon fences; don't run it there by hand.
- `pylint`, which must rate 10/10. Ruff covers part of Pylint's rules, but
  Pylint itself is not a pre-commit hook.
- `tests`, and `doctests` for the docstring examples.

Pytest runs with `filterwarnings = error`: a warning raised in a test fails it.

## Conventions

- Units are SI since v5: metres, tesla, amperes. Magnets take `polarization` in
  T or `magnetization` in A/m; examples prefer `polarization`.
- Index names follow SPOTIN (s sources, p path, o observers, t targets, i
  instances), see CONTRIBUTING.md.
