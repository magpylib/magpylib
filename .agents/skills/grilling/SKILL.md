---
name: grilling
description: >-
  Interview the user to stress-test and clarify a plan, design, or decision. Use
  when the user asks to be grilled or when consequential requirements remain
  ambiguous and implementation should wait for explicit decisions.
license: BSD-3-Clause
---

# Grilling

Use disciplined Socratic questioning to expose assumptions and make the user's
reasoning explicit. Research factual questions from the repository or primary
sources; reserve questions for choices that require human judgment.

## Process

1. State the proposal, desired outcome, and known constraints.
1. Ask the user to explain the evidence, assumptions, definitions, and expected
   consequences behind the proposal.
1. Test the reasoning with counterexamples, boundary cases, and credible
   alternatives.
1. Address one coherent topic at a time so answers can inform later questions.
1. Explain why each question matters and state a recommendation when the
   available evidence supports one.
1. Revisit answers that conflict with established constraints or one another.
1. Summarize decisions, assumptions, rejected alternatives, and remaining open
   questions for confirmation.

Do not ask the user for facts that can be established from code, documentation,
tools, or authoritative external sources. Do not begin implementation until the
user confirms the resulting understanding or explicitly asks to proceed despite
listed uncertainties.

## Result

The user receives a confirmed account of the proposal, supporting reasons,
assumptions, trade-offs, and unresolved questions.

## Sources

This method is based on the public tradition of Socratic questioning and
critical-thinking prompts about clarification, evidence, assumptions,
alternatives, and consequences.
