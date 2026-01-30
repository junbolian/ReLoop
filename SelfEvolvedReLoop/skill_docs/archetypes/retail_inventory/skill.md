## Skill Card: retail_inventory
## Purpose
Retail inventory planning with external data dict (Regime B).

## Required inputs
- text (scenario description)
- data_dict provided via CLI --data-json or programmatic input_data

## Coverage level
- baseline_no_aging (no perishable aging queue, no waste by age)

## Canonical JSON schema
```json
{
  "data": {"products": [], "locations": [], "periods": [], "...": "..."},
  "missing_inputs": [],
  "missing_capabilities": ["perishable_aging_queue","waste_accounting_by_age"]
}
```

## Modeling outline (baseline)
- Vars: prod[p,l,t] >=0, inv[p,l,t] >=0, sales[p,l,t] >=0, lost[p,l,t] >=0
- Demand: sales + lost == demand[p,t]*demand_share[l]
- Balance: inv_t = inv_{t-1} + prod - sales
- Prod cap: sum_l prod <= production_cap[p][t]
- Cold capacity: sum_p cold_usage[p]*inv <= cold_capacity[l]
- Obj: min purchase_cost*prod + hold_cost*inv + lost_penalty*lost

## Common pitfalls
- Missing data_dict → return NotYetImplemented with missing_inputs ["data_dict"]
- No aging modeled; document missing_capabilities.

## Failure modes
- Infeasible capacity; diagnose.

## Memory impact
- data and coverage_level stored for replay.
