## Purpose
Produce a JSON IR (intermediate representation) of the optimization model before building.

## When to use
- After canonicalization and trusted fix retrieval, before build_model.

## Inputs (state fields)
- canonical_schema
- archetype_id

## Outputs (state fields)
- model_plan (IR JSON)

## Hard constraints
- JSON only; conform to model_ir.py schema (sets, parameters, variables, constraints, objective).
- Do not change objective sense; use "minimize" for cost problems.
- Include missing_fields array if information is absent.

## Failure modes & recovery
- Invalid IR → builder falls back to code templates.
- Over budget → trim docs but keep schema and rules.

## Examples
- Objective: {"sense":"minimize","expr":[["cost","x[item]"]]}

## Audit expectations
- IR stored in run snapshot; replay can recompile without LLM.

## Memory impact
- model_plan persisted; prompt injection metadata recorded.
