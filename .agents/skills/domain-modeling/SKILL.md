---
name: domain-modeling
description: >-
  Clarify domain concepts, terminology, invariants, and architectural decisions.
  Use when names are overloaded, code and documentation disagree about a
  concept, or a consequential design decision needs durable rationale.
license: BSD-3-Clause
---

# Domain Modeling

Make the project's language precise enough that code, tests, documentation, and
discussion refer to the same concepts. In Magpylib, preserve established physics
terminology and distinguish physical quantities from implementation
representations.

## Process

1. Collect the terms, relationships, invariants, and concrete scenarios
   involved.
1. Compare their use across public APIs, implementation, tests, documentation,
   issues, and relevant scientific sources.
1. Surface overloaded or contradictory meanings and propose precise
   alternatives.
1. Stress-test the model with boundary cases and counterexamples.
1. Confirm canonical terms and definitions with the user.
1. Update the smallest existing canonical documentation location. Create a new
   glossary only with user agreement and place it within the established docs
   structure.

Offer an architecture decision record when a design choice significantly affects
the system and future contributors will need its rationale. Use an existing
project convention when present; otherwise ask before introducing one. Record
one decision per document together with its context, considered options,
consequences, and status.

## Boundaries

A domain glossary describes concepts and language, not file layout or current
implementation detail. A specification describes required behavior. An ADR
records why a durable technical choice was made. Assign each fact to one of
these artifact types and use cross-references where readers need the connection.

## Sources

The ubiquitous-language practice follows Eric Evans' _Domain-Driven Design_.
Decision-record guidance follows the public
[Architecture Decision Record collection](https://github.com/architecture-decision-record/architecture-decision-record).
