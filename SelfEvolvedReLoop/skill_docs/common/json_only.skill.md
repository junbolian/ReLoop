## Purpose
Enforce strict JSON-only outputs for LLM steps.

## When to use
Before any LLM that must return machine-parsable JSON (route, extract, model_plan, diagnose, fix_propose).

## Inputs (state fields)
- scenario text or relevant state fields for the node.

## Outputs (state fields)
- LLM response parsed as JSON into node-specific output.

## Hard constraints
- Respond with JSON only; no prose, no code fences.
- Include all required fields; set unknown numeric fields to 0 and unknown strings to "".
- Provide `missing_fields` array when information is absent.

## Failure modes & recovery
- Malformed JSON → retry with strict prompt.
- Missing fields → populate `missing_fields` and keep zeros.

## Examples
- Extraction returns `{"items": [], "requirements": {}, "missing_fields": ["items"]}`.

## Audit expectations
- Downstream parser must succeed without manual fixes.

## Memory impact
- None directly; enables storing deterministic JSON in run snapshots.
