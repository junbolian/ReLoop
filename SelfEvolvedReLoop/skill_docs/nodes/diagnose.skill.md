## Purpose
Explain solver outcomes and suggest symptom tags / safe tweaks.

## When to use
- When audit fails or solver status not optimal.

## Inputs (state fields)
- solve_result
- audit_result
- symptoms
- archetype_id

## Outputs (state fields)
- diagnosis (summary, symptom_tags, recommended_params)

## Hard constraints
- Do not invent data; only suggest param tweaks or tightening checks.
- JSON output with fields: summary, symptom_tags, recommended_params, missing_fields.

## Failure modes & recovery
- Insufficient info → add to missing_fields.

## Examples
- {"summary":"infeasible","symptom_tags":["SOLVER_INFEASIBLE"],"recommended_params":{"Method":1}}

## Audit expectations
- Tags feed fix proposal; IIS if available handled by code path.

## Memory impact
- diagnosis stored in run snapshot.
