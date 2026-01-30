# SelfEvolvedReLoop Technical Report (v2026-01-26)

This document summarizes the design and implementation of **SelfEvolvedReLoop**, a publication-grade Operations Research (OR) agent with progressive-disclosure skills, patch-based self-evolution, and deterministic auditing.

## 1. System Overview
- **Goal:** Given a natural-language scenario (and optional data dict), produce a solved OR result with full audit trail and fix lifecycle.
- **Pipeline (LangGraph):** `route → extract → canonicalize → retrieve_trusted_fixes → model_plan → build_model → solve → audit → (diagnose → propose_fix → sandbox_test → apply_fix) → finalize`.
- **Reproducibility:** Every run is snapshotted to `memory/runs/<run_id>.json` (run_id = UTC timestamp + hash(input_text)). Trusted/candidate/negative fixes are append-only JSONL.

## 2. Skills and Manifest
- Skills are first-class units (router, extract, canonicalize, model_plan, model_build, solve, audit, diagnose, fix_propose, sandbox_test, apply_fix, finalize).
- Machine-readable manifest: `skills_manifest.yaml` lists id, type (llm/code/hybrid), triggers, doc paths, IO schema, safety tags, budgets.
- Human-readable skill cards under `skill_docs/` (common, nodes, archetypes/*) with Purpose/When/Inputs/Outputs/Constraints/Failure modes/Audit/Memory.
- Archetype docs include canonical JSON schemas and modeling outlines.

## 3. Progressive Disclosure (Claude-style)
- Before every LLM call, **PromptAssembler** loads only the minimal required docs (common + node + archetype) and trims to a token budget (default 3000, overridable via `--disclosure-budget-tokens`).
- Injected docs are hashed (sha256) and recorded in `state.prompt_injections[skill_id]` and persisted in run snapshots for audit.
- No secrets or full prompts are persisted.

## 4. LLM Runtime
- LLM access via `llm.py` (`get_llm_client`), using env vars `OR_LLM_BASE_URL`, `OR_LLM_API_KEY`, optional `OR_LLM_MODEL`; temperature 0 for extraction/model_plan; retries disabled unless configured by LangChain defaults.
- All LLM calls go through **SkillRuntime**, which:
  - Assembles prompts via PromptAssembler.
  - Logs injection metadata.
  - Enforces JSON-only outputs per prompt templates (`prompts/*.txt`).

## 5. Model IR and Planning
- **model_ir.py:** JSON-serializable IR with sets, parameters, variables, linear constraints, objective, provenance.
- **model_plan** node (LLM): emits IR; **build_model** compiles IR to Gurobi if valid; otherwise falls back to code builders.
- `model_plan_hash` stored in runs for integrity tracking.

## 6. Archetypes and Formulations
- **allocation_lp** (alias `diet_lp`): Minimize cost, x[i] ≥ 0, feature matrix satisfies ≥ requirements. Dynamic features (no hardcoded nutrient names).
- **transport_flow:** Min-cost flow; supply ≤, demand ≥.
- **assignment:** Binary y[w,t]; each worker/task exactly one; min cost.
- **facility_location (UFL baseline):** Binary open[f], ship[f,c] ≥ 0, demand served, linking ship ≤ open; min open_cost + ship_cost.
- **retail_inventory (Regime B, baseline_no_aging):**
  - Inputs only via `input_data` (never sent to LLM). Required keys: products, locations, periods, demand[p][t]; optional demand_share, cold_capacity, cold_usage, production_cap, purchase_cost, hold_cost, lost_penalty, coverage_level.
  - Vars: prod, inv, sales, lost (all ≥ 0). Constraints: demand satisfaction per loc-period, inventory balance, cold capacity, production caps. Objective: purchase + holding + lost penalty.
  - If data missing: returns NotYetImplemented with `missing_inputs=["data_dict"]`.

## 7. Extraction and Canonicalization
- Router: rule-based first; LLM fallback if confidence < 0.7.
- Extraction: LLM JSON-only when available; deterministic fallback for allocation_lp (NL4OPT-style bullet parser). Retail directly wraps provided data dict.
- Canonicalization: unify feature keys, fill zeros, enforce nonneg floats, ordered items/features; warnings recorded.

## 8. Solve, Audit, Diagnose
- Solve: Gurobi (OutputFlag=0); captures status, objective, runtime, mipgap.
- Audit: archetype-specific residual checks (nutrient slacks, flow balances, assignment sums, FL linking, retail demand/balance/capacity).
- Diagnose: IIS or param hints; symptom taxonomy includes EXTRACT_PARSE_ERROR, CANONICALIZE_INCOMPLETE, SOLVER_INFEASIBLE, SOLVER_UNBOUNDED, AUDIT_VIOLATION, SOLVER_SLOW, MISSING_INPUT_DATA. May call LLM (progressive disclosure) to enrich tags.

## 9. Fix Lifecycle (Patch-Based Self-Evolution)
- Fix types: `prompt_patch` (use stricter extraction prompt), `canonicalize_patch` (extra normalization). Never changes objectives or relaxes constraints.
- Lifecycle: candidate (per-run) → trusted (only after human `correct` feedback) → negative (`wrong`) → skipped (`skip` keeps as candidate).
- Retrieval (`FixMemoryRetrieveSkill`): only trusted fixes, filtered by archetype, scored by symptom_tag overlap, top-k (default 3). Candidate/negative never auto-applied.
- Sandbox test: applies candidate on cloned state, reruns extract→audit, compares audit_pass/objective/runtime; recommends apply/reject/inconclusive.
- Auto-apply policy: trusted only if `--auto-apply-trusted true`; candidates only if `--allow-candidate-auto-apply true` and sandbox improves without objective regression.

## 10. Memory & Persistence
- Run snapshots: `memory/runs/<run_id>.json` include input_text, archetype_id, schemas, model_plan/hash, solve/audit/diagnosis, symptoms, fixes, prompt_injections, warnings, expected_obj/tolerance, correctness flag, not_yet_implemented (if any).
- Fix stores: `memory/candidates.jsonl`, `memory/trusted.jsonl`, `memory/negative.jsonl` (append-only).
- Atomic writes via `atomic_append_jsonl` and `write_json`; no secrets persisted.

## 11. CLI Surface
- `solve --text "..."`
  - Flags: `--auto-apply-trusted`, `--suggest-candidate`, `--allow-candidate-auto-apply`, `--data-json`, `--expected-obj`, `--tolerance`, `--disclosure-budget-tokens`, `--llm-model`, `--llm true|false`.
  - Prints run_id, archetype, status/objective, audit result, correctness (if expected), prompt injection hashes.
- `feedback --run <run_id> --label correct|wrong|skip [--notes "..."]`: promotes/demotes fixes.
- `replay --archetype <id> --max-runs N`: LLM-free regression using stored schemas/model_plan.
- `smoke-llm`: quick E2E with/without creds (exit 2 if missing).
- `evaluate --dataset <path> --format nl4opt [--llm true|false] [--tolerance] [--max-examples] [--out-dir]`: offline/online batch evaluation; writes `outputs_eval/...`.

## 12. Evaluation (NL4OPT 3-example)
- Dataset: `samples/nl4opt_3.jsonl`.
- Runner: `eval_min/runner.py` -> per-case solve, abs_error vs expected, correctness flag (needs audit_pass & optimal & |obj-exp|≤tol).
- Outputs: `outputs_eval/nl4opt_3/<timestamp>/{results.jsonl,summary.json,report.md}`.

## 13. Reproducibility & Safety
- Deterministic seeds (temperature 0); model IDs configurable; LLM optional (fallbacks for allocation_lp).
- No external web calls; only OpenAI-compatible endpoint if configured.
- Run records exclude API keys/base URLs; tests assert no secret leakage.
- Fix safety validator forbids objective changes and constraint relaxation.

## 14. Limitations
- Retail coverage_level is `baseline_no_aging` (no perishable aging queue or substitution edges).
- Model IR presently linear-only; nonlinear constraints unsupported.
- Routing and extraction quality improve with LLM; offline fallbacks may be approximate.

## 15. How to Extend
- Add archetype doc under `skill_docs/archetypes/<id>/skill.md`, update manifest triggers.
- Implement extraction/canonicalize/model builder/audit dispatch for the new archetype in `skills/*.py`.
- Update prompts if LLM nodes need schema-specific guidance.
- Add regression tests (LLM-free where possible) and replay coverage.

## 16. Key Paths (clickable)
- Code: `controller.py`, `skill_runtime.py`, `prompt_assembler.py`, `model_ir.py`, `skills/`.
- Docs: `skill_docs/`, `prompts/`.
- Memory: `memory/runs/`, `memory/*.jsonl`.
- Eval: `eval_min/`, `samples/nl4opt_3.jsonl`.

## 17. Quick Commands
- Unit tests: `python -m pytest -q`
- Offline solve (allocation):  
  `python -m SelfEvolvedReLoop.cli solve --text "...JSON schema..." --llm false`
- Retail solve with data dict:  
  `python -m SelfEvolvedReLoop.cli solve --text "retail test" --data-json '<json>' --llm false`
- Evaluate offline NL4OPT:  
  `python -m SelfEvolvedReLoop.cli evaluate --dataset SelfEvolvedReLoop/samples/nl4opt_3.jsonl --format nl4opt --llm false --tolerance 1e-6`

---
Prepared for top-tier OR/AI reproducibility: progressive disclosure, append-only memory, IR-backed modeling, and human-in-the-loop fix promotion.
