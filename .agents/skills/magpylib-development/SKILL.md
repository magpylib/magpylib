---
name: magpylib-development
description: >-
  Develop and maintain the Magpylib repository. Use when changing Python source,
  tests, packaging, CI, or contributor tooling in this repository, or when
  choosing the correct local validation command.
license: BSD-3-Clause
---

# Magpylib Development

Follow the repository's established Scientific Python workflow. Read the code
nearest to the requested behavior and its focused tests before changing it.

## Repository map

- Public package: `src/magpylib/`
- Internal implementation: `src/magpylib/_src/`
- Tests: `tests/`
- Sphinx and MyST documentation: `docs/`
- Development sessions: `noxfile.py`
- Tool configuration and dependency groups: `pyproject.toml`

Keep public APIs in the public package namespaces. Preserve the existing split
between public classes and helpers in `_src`, and update type information when a
public signature changes.

## Implementation workflow

1. Identify the public or package-internal behavior that owns the change.
1. Find the closest existing test and state the behavior it observes.
1. For a bug fix or behavior change, add a focused regression test and witness
   the intended failure before editing the implementation when practical.
1. Make the smallest coherent change at the owning abstraction.
1. Run the focused test immediately, then widen validation according to risk.
1. Update user documentation and docstrings when the public contract changes.

Do not rewrite unrelated code or weaken warnings, numerical tolerances, or test
assertions merely to obtain a passing run. For numerical work, derive expected
values independently from the implementation under test and include boundary,
inside/outside, path, and array-shape cases where relevant.

## Project conventions

- Python support starts at 3.11.
- Public code is typed; MyPy is strict for `magpylib.*`.
- Public docstrings use NumPy style.
- Source and test formatting is governed by the checked-in Ruff, Pylint, and
  prek configuration.
- Tests run with strict Pytest configuration and warnings treated as errors.
- Follow the SPOTIN axis vocabulary documented in `CONTRIBUTING.md`.
- Use SI units for Magpylib v5 code and examples.

## Validation

Start narrow and broaden only as the changed surface requires:

```console
uv run pytest tests/test_relevant_file.py -k relevant_behavior
uv run pytest
uvx nox -s lint
uvx nox -s pylint
uvx nox -s docs --non-interactive
uvx nox -s build
```

Use `uvx nox` when Nox is not installed. A source change normally requires its
focused tests and linting. A public API change also requires the relevant docs
build. Packaging changes require the build session.

## Sources

This project workflow is derived from `.github/CONTRIBUTING.md`,
`CONTRIBUTING.md`, `noxfile.py`, and `pyproject.toml`. Its general packaging and
tooling practices follow the public
[Scientific Python Development Guide](https://learn.scientific-python.org/development/)
and the [scientific-python/cookie](https://github.com/scientific-python/cookie)
template used to generate this repository.
