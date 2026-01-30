## Purpose
Guide LLM to handle numeric values robustly (units, conversions, defaults).

## When to use
- Any extraction/model_planning step where numbers appear in text.

## Inputs (state fields)
- scenario text.

## Outputs (state fields)
- numeric fields in JSON outputs.

## Hard constraints
- Parse numbers as floats.
- If units are ambiguous, leave as-is and add to `missing_fields`.
- Do not invent precision; keep source significant figures.

## Failure modes & recovery
- Non-numeric tokens → set 0 and list in `missing_fields`.

## Examples
- "cost $3.5" → 3.5

## Audit expectations
- Downstream canonicalizer clips negatives to zero.

## Memory impact
- None.
