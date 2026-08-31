---
name: magpylib-skill-maintenance
description: >-
  Create, review, or update Magpylib's package-distributed Agent Skill. Use when
  changing src/magpylib/.agents/skills, adding skill references, validating
  Agent Skills metadata, or checking that the skill ships in distributions.
license: BSD-3-Clause
---

# Magpylib Skill Maintenance

The user-facing `magpylib` skill is package data under
`src/magpylib/.agents/skills/magpylib/`. It is versioned with the API it
describes and discovered from an installed distribution by `library-skills`.

Do not confuse it with repository development skills under `.agents/skills/`.
Those guide contributors in this checkout and are not part of the installed
Magpylib package.

## Ground the change

When the request names a specific correction, investigate that assertion and the
neighboring guidance it affects. When the whole skill may be stale, build a
complete checklist of its API names, parameters, defaults, units, shapes,
examples, caveats, and file references. Independent read-only agents may verify
separate checklist sections when available.

Every technical claim must be supported by current package code, an executable
check, or canonical documentation. Prefer executable checks for output shapes,
path behavior, unit-sensitive examples, warnings, and exceptions.

## Authoring rules

- Keep `name: magpylib`; the directory and frontmatter name must match.
- Make the description specific enough to trigger before stale pre-v5 knowledge
  is used, especially for units and `polarization` versus `magnetization`.
- Keep universally needed guidance in `SKILL.md`; place branch-specific detail
  in `references/` and link it relatively.
- Use paths valid in the installed package. Do not point users to
  repository-only source, tests, or docs.
- Keep examples minimal, executable, and in SI units.
- Update or remove contradicted claims instead of preserving historical advice.

Author the skill directly at its package path. Do not run `library-skills`
installation from the repository root: a generated symlink can alter Hatch's
sdist collection and omit the real packaged skill.

## Validate

Build both distribution formats first, then run the focused packaging tests:

```console
uvx nox -s build
uv run pytest tests/test_package.py
```

The order matters. `test_agent_skill_ships_in_distributions` opens the wheel and
sdist left in `dist/` and asserts that `SKILL.md` and every linked reference are
present at `magpylib/.agents/skills/magpylib/`; with no artifacts to inspect it
skips. Every other test in that file resolves through `magpylib.__file__`, which
is the source tree under an editable install, so it checks the frontmatter and
relative links but cannot show that package data ships.

## Sources

The format follows the public
[Agent Skills specification](https://agentskills.io) and
[library-skills](https://library-skills.io/) discovery convention. Package
placement and the symlink caveat are grounded in this branch's introducing
commit and `tests/test_package.py`.
