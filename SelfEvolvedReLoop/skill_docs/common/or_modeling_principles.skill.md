## Purpose
Provide safe OR modeling principles for LP/MIP construction.

## When to use
- Model planning/building for any archetype.

## Inputs (state fields)
- Canonical schema (sets/params).

## Outputs (state fields)
- Model plan / IR that can compile to Gurobi.

## Hard constraints
- Objectives must match problem intent (usually cost minimization).
- Do not relax constraints; only add tightening cuts if justified.
- Variable domains must reflect problem (nonnegative or binary).

## Failure modes & recovery
- Infeasible plan → mark `missing_fields` and prefer conservative bounds.

## Examples
- Diet/allocation: x[item] >=0, meet requirements >=.

## Audit expectations
- Feasible solutions satisfy all constraints within tolerance 1e-6.

## Memory impact
- Plans stored in run record for replay.
