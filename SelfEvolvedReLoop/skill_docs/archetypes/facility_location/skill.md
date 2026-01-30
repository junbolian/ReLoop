## Skill Card: facility_location
## Purpose
Uncapacitated (with optional capacity) facility location with opening + shipping costs.

## Canonical JSON schema
```json
{
  "facilities": [{"name":"F1","open_cost":0.0,"capacity": "inf"}],
  "customers": ["C1"],
  "ship_cost": {"F1": {"C1": 0.0}},
  "demand": {"C1": 1.0},
  "missing_fields": []
}
```

## Modeling outline
- Vars: open[f] binary; ship[f,c] >=0
- Obj: minimize sum open_cost*open + sum ship_cost*ship
- Constraints: sum_f ship[f,c] == demand[c]; ship[f,c] <= open[f]; optional capacity sum_c ship[f,c] <= cap[f]

## Common pitfalls
- Missing demand → default 1 and list missing_fields.
- Capacity set to inf if absent.

## Coverage level
- implemented

## Failure modes
- Demand not served; linking violation.

## Memory impact
- Store schema/plan for replay.
