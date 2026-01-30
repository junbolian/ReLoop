## Purpose
Ensure proposed fixes are safe, auditable, and do not alter problem semantics.

## When to use
- Any fix proposal or application step.

## Inputs (state fields)
- archetype_id, symptoms, diagnosis, candidate fix.

## Outputs (state fields)
- Safe FixRecord or rejection.

## Hard constraints
- Do NOT relax constraints or change objective sense.
- Do NOT fabricate data; extraction fixes may only adjust prompts or normalization.
- Candidate fixes auto-apply only if sandbox improves audit and objective.

## Failure modes & recovery
- Unsafe patch detected → reject and log reason.
- Missing preconditions (e.g., no LLM) → mark non-applicable.

## Examples
- Prompt tightening for extraction; canonicalization tweak.

## Audit expectations
- Sandbox compares audit_pass, objective, status before applying.

## Memory impact
- Candidate/trusted/negative jsonl entries; append-only.
