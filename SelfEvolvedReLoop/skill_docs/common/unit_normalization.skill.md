## Purpose
Normalize units (e.g., grams vs kilograms) during extraction and planning.

## When to use
- Allocation or transport problems where units may vary.

## Inputs (state fields)
- scenario text with quantities.

## Outputs (state fields)
- normalized numeric coefficients.

## Hard constraints
- If unsure about units, do not convert; flag in `warnings` or `missing_fields`.
- Do not mix units without explicit conversion factors.

## Failure modes & recovery
- Conflicting units -> prefer leaving raw values and listing conflict.

## Examples
- "5 kg" -> 5000 g if explicitly requested; otherwise leave 5 and warn.

## Audit expectations
- Canonicalizer may clip negatives but not rescale.

## Memory impact
- None.
