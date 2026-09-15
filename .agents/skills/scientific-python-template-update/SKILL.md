---
name: scientific-python-template-update
description: >-
  Update Magpylib from the scientific-python/cookie Copier template. Use when
  running or reviewing a Copier update, changing .copier-answers.yml, or
  reconciling generated template changes with repository customizations.
license: BSD-3-Clause
---

# Scientific Python Template Update

Magpylib tracks `gh:scientific-python/cookie` in `.copier-answers.yml`. Template
updates are integration work: preserve intentional Magpylib customizations while
accepting applicable upstream improvements.

## Prepare

1. Read `.copier-answers.yml` and the public template release notes.
1. Record the current and target template revisions.
1. Review the working tree and preserve any unrelated user changes before the
   update begins.
1. Require a clean working tree and use a dedicated update branch. Get explicit
   approval before stashing, committing, switching branches, or running Copier
   when the user has not already requested those operations.

## Apply and reconcile

Run `copier update` with the requested template revision and reuse the recorded
answers unless the update intentionally changes them. Do not edit
`.copier-answers.yml` manually; Copier documents that this makes later smart
updates unreliable.

Review every generated change and resolve inline conflict markers or `.rej`
files according to the selected conflict mode. Compare the result with the
pre-update commit to confirm that project metadata, dependencies, automation,
documentation configuration, and Magpylib-specific behavior remain intentional.
Use repository history when the purpose of an existing customization is unclear.

## Validate

Run the checks affected by the update, followed by the full repository gates:

```console
uvx nox -s lint
uvx nox -s pylint
uvx nox -s tests
uvx nox -s docs --non-interactive
uvx nox -s build
```

Review the final diff against the pre-update commit. Report accepted upstream
changes, retained project customizations, deliberate departures, unresolved
conflicts, and any validation that could not run.

## Sources

This workflow follows public
[Copier update behavior](https://copier.readthedocs.io/en/stable/updating/) and
Magpylib's recorded use of the `scientific-python/cookie` template.
