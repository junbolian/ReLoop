# ReLoop Agents (Step1–Step5)

This directory contains a LangGraph-based orchestrator that runs a 5-step workflow for MILP code generation on retail scenarios, with **Semantic Probe Verification** for detecting silent failures.

## State Machine

```
profile_data → step1 → step2 → step3 → sanity → step4_codegen → static_audit → semantic_probe → step5_run
               contract  spec  templates check    codegen                       NEW!           run/repair
```

### Nodes:
- `profile_data`: Summarize data structure (types/indexing only)
- `step1`: Lock task contract (optimize, controls, constraints)
- `step2`: Build spec sheet (sets, decisions, objective, constraint families)
- `step3`: Map to constraint templates (mathematical formulas)
- `sanity_check`: Code-based validation (no LLM)
- `step4_codegen`: Generate Gurobi Python script
- `static_audit`: Check for forbidden patterns
- `semantic_probe`: **NEW** - Run 8 boundary tests to detect constraint errors
- `step5_run`: Execute code, diagnose errors, repair if needed

### Conditional Edges:
- Failed sanity → `step2` (revise spec)
- Failed static audit → `step4_codegen` (retry)
- Failed semantic probe → `step4_codegen` (repair with diagnosis)
- Repair brief target=SPEC → `step2`
- Repair brief target=CODEGEN → `step4_codegen`
- Hard cap: `repair_limit=5`, `max_turns=12`

### State Fields (Pydantic v2):
```python
run_id, scenario_id, base_prompt_hash, data, scenario_text, base_prompt,
data_profile, step1_contract, step2_spec_sheet, step3_templates,
sanity_report, code_versions, static_audit_reports, semantic_probe_reports,
solve_reports, repair_briefs, repair_count, last_error, conversation_log, turn_index
```

---

## Prompt Stacking

The base prompt comes from `scenarios/prompts/<scenario>.txt`. It is never modified; its SHA-256 hash is stored.

Every LLM call uses messages in this order:
1. `00_global_guardrails.txt` - Code standards, naming contract, substitution semantics
2. Step prompt (01-07) with base prompt prefixed
3. Runtime context (scenario narrative, data_profile, previous outputs)
4. Probe diagnosis (if available from failed probes)

### Step Prompt Mapping:
| File | Step | Purpose |
|------|------|---------|
| `01_step1_contract.txt` | Step 1 | Contract extraction |
| `02_step2_spec_sheet.txt` | Step 2 | Spec sheet (detailed schema) |
| `03_step3_constraint_templates.txt` | Step 3 | Mathematical templates |
| `04_step4_codegen.txt` | Step 4 | Code generation (rules only, no template) |
| `05_format_repair_json.txt` | Repair | JSON format fix |
| `06_format_repair_code.txt` | Repair | Code format fix |
| `07_step5_repair_brief.txt` | Step 5 | Repair diagnosis with probe info |

---

## Semantic Probes (NEW)

8 boundary test probes detect constraint errors **without parsing code**:

| Probe | Tests | Detects |
|-------|-------|---------|
| `substitution_basic` | S variable implementation | Wrong substitution direction |
| `demand_route_constraint` | S_out ≤ demand | UNBOUNDED model |
| `no_substitution` | Empty sub_edges handling | Spurious S variables |
| `production_capacity` | Prod cap enforcement | Missing constraint |
| `storage_capacity` | Storage cap enforcement | Wrong coefficients |
| `aging_dynamics` | Shelf-life aging | Missing waste calculation |
| `lost_sales_slack` | L variable presence | INFEASIBLE when demand > supply |
| `inventory_nonnegativity` | I ≥ 0 | Negative inventory allowed |

**Flow:**
```
static_audit passes → semantic_probe runs
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
            All PASS              Any FAIL/CRASH
                 │                     │
                 ▼                     ▼
         step5_run            repair_codegen
                              (with diagnosis)
```

---

## Persistence Layout

For each `run_id` under output root (default `artifacts/`):
```
artifacts/<run_id>/
├── meta.json                    # Run metadata, base prompt hash
├── events.jsonl                 # Append-only event log
└── turns/<k>/
    ├── messages.json            # Prompts + raw responses
    ├── step_outputs.json        # Structured outputs
    ├── code.py                  # Generated script
    ├── static_audit.json        # Audit results
    ├── semantic_probe.json      # Probe results (NEW)
    ├── stdout.txt / stderr.txt  # Runtime logs
    └── solve.json               # Solver status
```

---

## Tools

| Tool | Purpose |
|------|---------|
| `data_profiler` | Summarize types/indexing (no numeric values) |
| `sanity_checker` | 6 logic checks (lost-sales slack, shelf-life, substitution) |
| `static_auditor` | Block forbidden I/O, enforce solver params |
| `script_runner` | Execute code with `data` preloaded |
| `semantic_probes` | **NEW** - 8 boundary test probes |
| `persistence` | Atomic artifact writer |

---

## CLI

### Run one scenario:
```bash
python -m reloop.agents.cli.run_one \
  --scenario retail_f1_52_weeks_v0 \
  --out artifacts \
  --repair-limit 5 \
  --max-turns 12
```

### Run with probes disabled:
```bash
python -m reloop.agents.cli.run_one \
  --scenario retail_f1_52_weeks_v0 \
  --no-probes
```

### Run a suite:
```bash
python -m reloop.agents.cli.run_benchmark \
  --suite suite.txt \
  --out artifacts
```

---

## Key Design Decisions

1. **No code template in Step 4**: LLM must translate math → code (not copy-paste)
2. **Semantic probes before run**: Catch constraint errors early
3. **Probe-guided repair**: Diagnosis fed back to LLM
4. **Substitution semantics enforced**: Edge [A,B] = A's demand served by B