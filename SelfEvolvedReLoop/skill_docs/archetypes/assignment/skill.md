## Skill Card: assignment
## Purpose
Bipartite assignment (workers to tasks) minimizing cost.

## Canonical JSON schema
```json
{
  "workers": ["w1"],
  "tasks": ["t1"],
  "costs": {"w1": {"t1": 0.0}},
  "missing_fields": []
}
```

## Modeling outline
- Vars: y[w,t] binary
- Obj: minimize sum cost[w,t]*y[w,t]
- Constraints: sum_t y[w,t]=1 for each worker; sum_w y[w,t]=1 for each task

## Common pitfalls
- Unequal counts → infeasible; highlight in missing_fields/symptoms.

## Coverage level
- implemented

## Failure modes
- Infeasible assignment; diagnose.

## Memory impact
- Store schema/plan for replay.
