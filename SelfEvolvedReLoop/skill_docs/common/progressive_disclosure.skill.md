## Purpose
Enforce progressive disclosure: load only minimal necessary docs for each LLM call.

## When to use
- Every LLM skill before assembling prompts.

## Inputs (state fields)
- skill id, archetype_id, scenario text, budget.

## Outputs (state fields)
- prompt_injections[skill_id] metadata (files, sections, hashes, used_tokens_est).

## Hard constraints
- Load only listed docs for the skill and current archetype.
- Trim injected text to budget_tokens (default 3000).
- Record injected files and sha256 hashes in state.

## Failure modes & recovery
- Over budget → trim deterministically.
- Missing doc file → skip with warning.

## Examples
- route skill injects route.skill.md + archetype doc only.

## Audit expectations
- Run snapshot must show prompt_injections per LLM call.

## Memory impact
- prompt_injections stored in run record for audit/replay.
