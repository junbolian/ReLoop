## Purpose
Persist run snapshot, mark status, and surface NYI or correctness info.

## When to use
- End of pipeline.

## Inputs (state fields)
- run_id, solve_result, audit_result, candidate_fix, applied_fixes, not_implemented, prompt_injections.

## Outputs (state fields)
- RunRecord JSON persisted to memory/runs/<run_id>.json

## Hard constraints
- Append-only memory; do not mutate previous entries.
- Preserve backward compatibility of fields.

## Failure modes & recovery
- File write errors → surface warning.

## Examples
- status "success" with is_correct flag when expected_obj provided.

## Audit expectations
- prompt_injections must be stored for every LLM call.

## Memory impact
- RunRecord and fix jsonl updates.
