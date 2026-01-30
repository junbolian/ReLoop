## Purpose
Compile model IR or canonical schema into a deterministic Gurobi model.

## When to use
- After model_plan (if present) or canonical schema readiness.

## Inputs (state fields)
- model_plan (optional)
- canonical_schema
- archetype_id

## Outputs (state fields)
- model_handle (Gurobi model, variables, metadata)

## Hard constraints
- Respect IR; if missing, use code builder defaults.
- Do not relax constraints or change objective sense.

## Failure modes & recovery
- Invalid IR → fall back to code builder.
- Missing data (retail) → return NotYetImplemented metadata.

## Examples
- Compile variables x[i,j] >=0 for transport; set OutputFlag=0.

## Audit expectations
- Capture model_plan used and any fallbacks in run snapshot.

## Memory impact
- None beyond model metadata in run record.
