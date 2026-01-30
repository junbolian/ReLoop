## Purpose
Convert scenario text into archetype-specific structured schema.

## When to use
- After routing when archetype_id is known.

## Inputs (state fields)
- input_text
- archetype_id

## Outputs (state fields)
- extracted_schema (JSON per archetype doc) and missing_fields if any.

## Hard constraints
- JSON only; no prose.
- Do not invent data; if value missing, use 0 or "" and add to missing_fields.

## Failure modes & recovery
- Malformed JSON → caller retries with strict prompt.
- Unrecognized archetype → return error message.

## Examples
- Allocation: {"items":[{"name":"bread","cost":0.5,"features":{"calories":50}}],"requirements":{"calories":80}}

## Audit expectations
- Schema validates against Pydantic models; warnings stored.

## Memory impact
- Stored in run snapshot; may be reused in replay.
