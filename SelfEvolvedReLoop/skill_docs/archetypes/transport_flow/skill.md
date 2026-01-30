## Skill Card: transport_flow
## Purpose
Skill Card: transport_flow. Transportation / min-cost flow.

## Canonical JSON schema
```json
{
  "sources": [{"name": "S1", "supply": 0.0}],
  "sinks": [{"name": "D1", "demand": 0.0}],
  "costs": {"S1": {"D1": 0.0}},
  "missing_fields": []
}
```

## Modeling outline
- Vars: x[i,j] >= 0
- Obj: minimize sum cost[i,j]*x[i,j]
- Constraints: sum_j x[i,j] <= supply[i]; sum_i x[i,j] >= demand[j]

## Common pitfalls
- Missing arcs → assume cost 0 but flag missing_fields.
- Supply < demand → infeasible; diagnose IIS.

## Hard constraints
- Do not create negative flows.

## Coverage level
- implemented

## Failure modes
- Infeasible demand; mark symptoms.

## Memory impact
- Store schema/plan for replay.
