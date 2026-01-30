## Purpose
Propose a single safe candidate fix for observed symptoms.

## When to use
- After diagnosis when status is not success.

## Inputs (state fields)
- archetype_id
- symptoms
- diagnosis

## Outputs (state fields)
- candidate_fix (FixRecord JSON)

## Hard constraints
- No constraint relaxation or objective changes.
- Allowed patch_types: prompt_patch, canonicalize_patch.
- JSON only with fields: fix_id, patch_type, patch_payload, symptom_tags, preconditions, expected_effect.

## Failure modes & recovery
- No applicable fix → return null.
- Unsafe suggestion → reject.

## Examples
- {"fix_id":"run123-prompt-strict","patch_type":"prompt_patch","patch_payload":{"prompt_version":"v2_strict","strict_mode":true},"symptom_tags":["AUDIT_VIOLATION"],"preconditions":["LLM_available"],"expected_effect":"tighter extraction JSON"}

## Audit expectations
- Sandbox must evaluate before application unless trusted.

## Memory impact
- Candidate fix stored append-only; may be promoted on human feedback.
