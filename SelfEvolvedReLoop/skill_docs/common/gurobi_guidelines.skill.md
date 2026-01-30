## Purpose
Safe defaults for Gurobi modeling/solving.

## When to use
- Any model build or plan step targeting Gurobi.

## Inputs (state fields)
- Model IR / plan.

## Outputs (state fields)
- Gurobi model/params.

## Hard constraints
- Set `OutputFlag=0` for reproducibility.
- Avoid non-determinism; no random seeds.
- Respect variable types; avoid large-M unless justified.

## Failure modes & recovery
- Infeasible → compute IIS when available.
- Slow solves → suggest Method=1 or tighter tolerances.

## Examples
- Linear constraints only for IR compilation.

## Audit expectations
- Capture status, objective, runtime, mipgap in solve_result.

## Memory impact
- None directly.
