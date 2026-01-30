# SelfEvolvedReLoop

Publication-grade OR agent with self-evolving fixes, deterministic memory, and LangGraph control flow. Supports NL-only datasets (Regime A, e.g., NL4OPT/MAMO) and preloaded-data scenarios (Regime B, e.g., retail_inventory). Claude-style skills are defined via a manifest + skill cards with progressive disclosure and prompt injection logging.

## Architecture
- **Controller** (`controller.py`): LangGraph pipeline from route → extract → canonicalize → retrieve_trusted_fixes → build_model → solve → audit → (diagnose → propose_fix → sandbox_test → apply_fix) → finalize. Routing covers allocation_lp (diet alias), transport_flow, assignment, facility_location, retail_inventory.
- **State** (`state.py`): Pydantic models for run state, schemas, fixes, audits, and run records.
- **Skills** (`skills/`): Modular, typed skills for routing, extraction, canonicalization, modeling, solving, auditing, diagnosis, fix lifecycle, and replay validation.
- **Memory** (`memory/`): Append-only JSONL stores for candidates/trusted/negative fixes and per-run JSON snapshots for reproducibility.
- **Prompts** (`prompts/`): Baseline和strict allocation extraction prompts demanding JSON-only output (dynamic feature keys).
- **Skills & Progressive Disclosure**: `skills_manifest.yaml` + `skill_docs/` (common, nodes, archetypes) define every skill and archetype. Each LLM call loads only the minimal docs within a budget and records `prompt_injections` (files, hashes, budgets) in the run snapshot.

## Running
- Solve a scenario:
  ```bash
  python -m SelfEvolvedReLoop.cli solve --text "transport problem {...json...}" \
    --data-json '<json dict for retail>' \
    --expected-obj 123.4 --tolerance 1e-3 \
    --disclosure-budget-tokens 2500 --llm-model gpt-4o-mini \
    --auto-apply-trusted true|false --suggest-candidate true|false --allow-candidate-auto-apply true|false
  ```
  - `--data-json` supplies the preloaded data dict for retail_inventory (Regime B); if omitted, the run finalizes as NotYetImplemented with missing_inputs=["data_dict"].
  - `--expected-obj`, `--tolerance` tag correctness when objective within tolerance.
  - `--disclosure-budget-tokens` overrides doc budget; `--llm-model` overrides default model.
- Record human feedback:
  ```bash
  python -m SelfEvolvedReLoop.cli feedback --run <run_id> --label correct|wrong|skip [--notes "..."]
  ```
  - `correct` promotes the candidate fix to trusted.
  - `wrong` stores it as negative.
  - `skip` records label without promotion.
- Replay regression validation:
  ```bash
  python -m SelfEvolvedReLoop.cli replay --archetype diet_lp --max-runs 20
  ```
- Smoke LLM (requires OR_LLM_API_KEY):
  ```bash
  python -m SelfEvolvedReLoop.cli smoke-llm
  ```
  Missing key -> exits 2 with a clear message.

## Fix Lifecycle
- **candidate_fix**: proposed per run; append-only in `memory/candidates.jsonl`.
- **trusted_fix**: promoted only after human `correct` feedback; auto-eligible for future runs.
- **negative_fix**: stored on `wrong` feedback; used to down-rank retrieval.
- **skipped_fix**: feedback `skip`; stays candidate.

## Determinism & Auditability
- Run IDs: timestamp + hash of input text.
- Each run snapshot saved under `memory/runs/<run_id>.json`.
- Optional env vars for LLM: `OR_LLM_BASE_URL`, `OR_LLM_API_KEY` (required for LLM), `OR_LLM_MODEL` (optional). Compatible with `OPENAI_API_KEY`/`OPENAI_MODEL`. Keys are never written to snapshots. If no key, extraction falls back to deterministic parsing with warnings.
- Replay validation avoids LLM calls by default, reusing stored schemas when present.
- NotYetImplemented is explicit: includes `missing_inputs`, `missing_capabilities`, and `coverage_level` (e.g., retail baseline_no_aging).

## Extending Archetypes
- Add routing keywords in `skills/router.py`.
- Implement extractor/canonicalizer/model builder for the new archetype (dispatch inside existing skills; no controller edge changes needed).
- Keep append-only memory formats; add fields rather than changing schemas.
- For Regime B-style problems, accept data via `--data-json` or programmatic `input_data` parameter.
