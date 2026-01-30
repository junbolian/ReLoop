## Skill Card: allocation_lp
## Purpose
Skill Card: allocation_lp. Allocation LP (diet-style) minimizing cost while meeting feature requirements.

## Canonical JSON schema
```json
{
  "items": [{"name": "string", "cost": 0.0, "features": {"feature_name": 0.0}}],
  "requirements": {"feature_name": 0.0},
  "missing_fields": []
}
```

## Modeling outline
- Vars: x[item] >= 0
- Obj: minimize sum cost[item]*x[item]
- Constraints: sum feature[item, f] * x[item] >= requirements[f] for all f

## Common pitfalls
- Missing features → fill 0 and list in missing_fields.
- Negative numbers → clip to 0.

## Hard constraints
- Objective sense: minimize
- No relaxation of requirements.

## Coverage level
- implemented

## Inputs
- text

## Outputs
- allocation schema, model plan, solution

## Examples
- requirements calories>=2000, protein>=50 with foods.

## Failure modes
- Infeasible requirements; detection via audit/diagnose.

## Memory impact
- Stored schema and plan for replay.
