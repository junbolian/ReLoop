# Agents (Step0–Step6)

This directory contains a clean LangGraph-based orchestrator that runs a strict six-step workflow for MILP code generation on retail scenarios.

## State machine
- Nodes: profile_data -> step0 (contract) -> step1 (tag sentences) -> step2 (spec sheet) -> step3 (constraint templates) -> step4 (sanity checks) -> step5 (codegen) -> static_audit -> step6 (run/diagnose).
- Conditional edges: failed sanity -> step2; failed static audit -> step5; repair briefs loop back to step2 (SPEC) or step5 (CODEGEN). A hard repair cap stops after repeated failures.
- State fields (validated by Pydantic v2): run_id, scenario_id, base_prompt_hash, data_profile, step0_contract, step1_tags, step2_spec_sheet, step3_templates, step4_sanity_report, code_versions, static_audit_reports, solve_reports, iis_reports, repair_briefs, repair_count, last_error, conversation_log, base_prompt, scenario_text, data.

## Prompt stacking (immutable base)
- The base prompt comes from `scenarios/prompts/<scenario>.txt` (or `.user.txt`). It is never modified and its SHA-256 hash is stored.
- Every LLM call uses messages in this exact order:
  1) `reloop/agents/step_prompts/00_global_guardrails.txt`
  2) BASE PROMPT (immutable)
  3) step prompt for the current step (01...09 as mapped in `prompt_stack.py`)
  4) runtime context (scenario narrative, data_profile summary, previous step outputs)
- Format repair: if JSON parsing fails in Steps0–4/6, re-call with `07_format_repair_json.txt`; if code formatting fails in Step5, re-call with `08_format_repair_code.txt`.
- All messages and raw responses are logged per turn in `artifacts/<run_id>/turns/<k>/messages.json`.

## Persistence layout
For each run_id under the chosen output root (default `artifacts/`):
- meta.json (run metadata, base prompt hash) and events.jsonl (append-only events)
- turns/<k>/messages.json (prompts + raw responses)
- turns/<k>/step_outputs.json (structured outputs for that step)
- turns/<k>/code.py (latest generated script)
- turns/<k>/static_audit.json
- turns/<k>/stdout.txt and stderr.txt (runtime logs)
- turns/<k>/solve.json (solver status and metrics)
- turns/<k>/iis.json (optional IIS summary)
- turns/<k>/sanity.json (non-solver sanity results)
- Atomic writes are used for every file.

## Tools
- data_profiler: summarizes types/indexing only (no numeric values).
- sanity_checker: six logic-only checks catch missing lost-sales slack, missing shelf-life indexing, and missing substitution coupling.
- static_auditor: blocks forbidden I/O, enforces solver params/printing/naming hints.
- script_runner: executes generated code in a subprocess with `data` preloaded, captures stdout/stderr, and computes IIS when feasible.
- iis_tools: groups IIS constraint names by prefix.
- persistence: atomic artifact writer + JSONL event log.

## CLI
- Run one scenario: `python -m reloop.agents.cli.run_one --scenario <id-or-json> --base-prompt <path-or-text> --out artifacts`
- Run a suite: `python -m reloop.agents.cli.run_benchmark --suite <suite.txt> --out artifacts`
- Shims keep `python -m reloop.agents.run_one` and `python -m reloop.agents.run_benchmark` working but emit a deprecation warning.
