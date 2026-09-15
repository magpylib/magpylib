---
name: agent-skill-authoring
description: >-
  Design, write, or review repository Agent Skills and their bundled references.
  Use when adding or revising SKILL.md files, choosing invocation behavior, or
  improving skill discovery and progressive disclosure.
license: BSD-3-Clause
---

# Agent Skill Authoring

Create skills that change agent behavior predictably without loading irrelevant
material into every request. Use `magpylib-skill-maintenance` as well when the
target is Magpylib's package-distributed user skill.

## Design

1. Define the task branch the skill owns and a checkable completion condition.
1. Confirm a skill is the right primitive: use always-on instructions for rules
   that apply broadly, and a prompt for a single parameterized action.
1. Choose model invocation only when autonomous discovery or cross-skill use is
   valuable. Use `disable-model-invocation: true` for explicit utilities whose
   invocation should remain a human choice.
1. Write a concise description naming capabilities and genuine trigger branches.
   The description is the discovery interface, not a summary of every section.
1. Put ordered actions and universally needed rules in `SKILL.md`. Move
   branch-specific reference, examples, scripts, and templates into nearby files
   and link them only where needed.

## Authoring rules

- Match the frontmatter `name` to the containing directory.
- Keep instructions executable, positive, and specific about observable done
  conditions.
- Prefer repository configuration and canonical docs over duplicated facts that
  will become stale.
- State tool or environment assumptions and provide a viable fallback.
- Avoid hidden dependencies on unavailable skills, trackers, agents, or private
  services.
- Include provenance and licensing appropriate for shared repository content.

## Validate

Parse the YAML frontmatter, verify required fields and relative links, run the
repository's formatting and spelling hooks, and test triggering with requests
that should and should not activate the skill.

## Sources

The format and progressive-disclosure model follow the public
[Agent Skills specification](https://agentskills.io). Invocation behavior is
adapted for clients that support explicit model-invocation controls.
