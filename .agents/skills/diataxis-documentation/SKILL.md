---
name: diataxis-documentation
description: >-
  Write or review Magpylib documentation using the Diataxis framework. Use when
  changing tutorials, how-to guides, explanations, API reference, docstrings,
  examples, or documentation navigation in this repository.
license: BSD-3-Clause
---

# Diataxis Documentation

Give each page one primary reader need. Use this skill together with
`magpylib-development` for repository commands and local writing conventions.

## Classify before writing

- Tutorial: help a learner gain confidence by completing a guided experience.
- How-to guide: help a competent user accomplish a specific practical task.
- Reference: provide accurate, complete facts for lookup.
- Explanation: build understanding of concepts, reasons, and trade-offs.

Do not combine quadrants merely because their material concerns the same API.
Put each fact in one canonical location, summarize it briefly elsewhere, and
link to it.

## Magpylib documentation map

- Handwritten MyST pages live in `docs/_pages/`.
- API pages are generated from NumPy-style public docstrings.
- Images, videos, data, and web assets live in `docs/_static/`.
- Navigation begins in `docs/index.md` and the pages it references.
- Generated files in `docs/_autogen/` are build output; change their source
  docstrings or Sphinx configuration instead of editing them directly.

## Workflow

1. Identify the target reader, their goal, and the Diataxis quadrant.
1. Inspect the implementation, existing canonical docs, and neighboring pages.
1. Choose the canonical location and update navigation when adding a page.
1. Draft for one reader need, using runnable SI-unit examples where useful.
1. Verify API names, defaults, shapes, units, and output against current code.
1. Build the docs in nitpicky mode and repair relevant warnings or broken links.

For tutorials, keep a meaningful result visible at each stage. For how-to
guides, lead with the task and prerequisites. Keep reference neutral and
complete. Use explanation for physical concepts, design rationale, and
trade-offs rather than step-by-step instructions.

## Local writing rules

- Use MyST Markdown conventions in documentation and NumPy style in docstrings.
- Use the SPOTIN vocabulary from `CONTRIBUTING.md` for axes and shapes.
- Express examples in Magpylib v5 SI units.
- Include only the code needed to demonstrate verified behavior.
- Preserve one canonical home for tables, equations, and behavioral facts.
- Use descriptive link text and connect new pages to existing navigation.

Validate documentation changes with:

```console
uvx nox -s docs --non-interactive
```

## Sources

The four-quadrant method is adapted from the public
[Diataxis documentation framework](https://diataxis.fr/). Repository placement,
formatting, and validation come from `docs/README.md`, `CONTRIBUTING.md`, and
`noxfile.py`.
