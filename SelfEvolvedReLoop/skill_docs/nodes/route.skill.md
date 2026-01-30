## Purpose
Select the correct archetype for a scenario.

## When to use
- Always as the first step; LLM only if rule-based confidence < 0.7.

## Inputs (state fields)
- input_text

## Outputs (state fields)
- route (archetype_id, confidence, rationale, missing_inputs)

## Hard constraints
- JSON output only with fields: archetype_id, confidence (0-1), rationale, missing_inputs (array).

## Failure modes & recovery
- Ambiguous → lower confidence, include missing_inputs suggestion.

## Examples
- {"archetype_id": "transport_flow", "confidence": 0.74, "rationale": "mentions shipping from warehouses", "missing_inputs":[]}

## Audit expectations
- Routing rationale stored for transparency.

## Memory impact
- Prompt injection metadata captured.
