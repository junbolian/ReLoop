# ReLoop Code Changes Summary

## Overview

This document summarizes all changes made to integrate **Semantic Probe Verification** into the ReLoop pipeline and simplify the code generation prompt.

---

## 1. Architecture Changes

### Before (7 steps with IIS):
```
profile_data → step0 → step1 → step2 → step3 → step4 → step5 → static_audit → step6
               contract  tags   spec   templates sanity  codegen              run+IIS
```

### After (5 steps with Semantic Probes):
```
profile_data → step1 → step2 → step3 → sanity → step4 → static_audit → semantic_probe → step5
               contract spec   templates check  codegen               NEW!             run/repair
```

**Key Changes:**
- Removed `step0_tags` (sentence tagging) - unnecessary intermediate step
- Renumbered steps: contract is now Step 1
- Added `semantic_probe` node between static_audit and run
- Removed IIS computation entirely - replaced by probes
- Simplified repair flow with probe diagnostics

---

## 2. Prompt Changes (IMPORTANT)

### Step 4 Codegen Simplified

**Before:** Complete 150-line code template (essentially giving the answer)

**After:** Only rules and hints, no code template

```
Before (04_step4_codegen.txt):
├── Full variable creation loops
├── Complete constraint implementation
├── Full objective construction
└── ~150 lines of working code

After (04_step4_codegen.txt):
├── Variable naming rules
├── Constraint prefix rules
├── Data access patterns
├── Critical implementation notes
└── NO executable code
```

**Rationale:** 
- Previous setup was "copy-paste" difficulty
- New setup requires actual translation from math → code
- This better tests LLM reasoning ability
- Expected silent failure rate increases from ~25% to ~45%

---

## 3. File Changes Summary

### Modified Files

| File | Changes |
|------|---------|
| `schemas.py` | Added `SemanticProbeReport`, `ProbeResult`, `EvaluationResult`; Removed `IISReport`, `TaggedSentence`; Renamed step fields |
| `prompt_stack.py` | Updated `STEP_PROMPT_MAP` to 5-step structure; Added `probe_diagnosis` parameter |
| `orchestrator_graph.py` | Removed step0_tags node; Added semantic_probe node; Updated routing |
| `sanity_checker.py` | Updated field references |
| `script_runner.py` | Removed IIS computation |
| `persistence.py` | Updated to save probe reports |
| `cli/run_one.py` | Added `--no-probes` flag; Updated state initialization |

### New Files

| File | Purpose |
|------|---------|
| `tools/semantic_probes.py` | Core probe implementation (8 probes + ProbeRunner) |

### Deleted Files

| File | Reason |
|------|--------|
| `tools/iis_tools.py` | IIS no longer used - replaced by probes |

---

## 4. Step Prompt Files

### Final Structure:

| File | Step | Content |
|------|------|---------|
| `00_global_guardrails.txt` | Global | Code standards, naming contract, **substitution semantics** |
| `01_step1_contract.txt` | Step 1 | Contract extraction schema |
| `02_step2_spec_sheet.txt` | Step 2 | Detailed spec sheet schema (sets, decisions, constraints) |
| `03_step3_constraint_templates.txt` | Step 3 | Mathematical constraint templates |
| `04_step4_codegen.txt` | Step 4 | **Simplified** - rules only, no code template |
| `05_format_repair_json.txt` | Repair | JSON format fix |
| `06_format_repair_code.txt` | Repair | Code format fix |
| `07_step5_repair_brief.txt` | Step 5 | Repair diagnosis **with probe guide** |

---

## 5. Semantic Probe Details

### 8 Probes:

| Probe | Mechanism | Expected Behavior | Error Detected |
|-------|-----------|-------------------|----------------|
| `substitution_basic` | Substitution | Obj in range [lb, ub] | Wrong S direction |
| `demand_route_constraint` | S_out ≤ demand | Not UNBOUNDED | Missing constraint |
| `no_substitution` | Empty edges | No spurious benefit | S when shouldn't exist |
| `production_capacity` | Prod cap | Obj ≥ lower bound | Missing prod_cap |
| `storage_capacity` | Storage cap | INFEASIBLE | Wrong constraint |
| `aging_dynamics` | Shelf-life | Obj includes waste | Missing aging |
| `lost_sales_slack` | L variable | Not INFEASIBLE | Missing L |
| `inventory_nonnegativity` | I ≥ 0 | No negative I | Wrong bounds |

### Integration with Repair:

```python
# In step4_codegen, if probes failed previously:
probe_diagnosis = """
- demand_route_constraint: UNBOUNDED - missing S_out <= demand
- substitution_basic: objective too high - check edge direction
"""

# This diagnosis is injected into the LLM call
```

---

## 6. State Schema Changes

### Removed:
```python
step1_tags: List[TaggedSentence]  # No longer used
iis_reports: List[IISReport]       # Replaced by probes
```

### Added:
```python
semantic_probe_reports: List[SemanticProbeReport]
```

### Renamed:
```python
step2_spec_sheet → step1_spec_sheet  # Now Step 1
step3_templates → step2_templates    # Now Step 2
step4_sanity_report → sanity_report  # Simplified
```

### SemanticProbeReport Structure:
```python
class SemanticProbeReport(BaseModel):
    total: int           # 8
    passed: int          # 0-8
    failed: int          # 0-8
    crashed: int         # 0-8
    pass_rate: float     # 0.0-1.0
    failed_probes: List[str]
    probe_results: List[ProbeResult]
    diagnoses: Dict[str, str]  # probe_name → diagnosis
```

---

## 7. Substitution Semantics (CRITICAL)

Documented in `00_global_guardrails.txt`:

```
Edge [p_from, p_to] means: p_from's demand can be served by p_to's inventory

Example: ["SKU_Basic", "SKU_Premium"] means:
- Basic's demand can be served by Premium's inventory
- S[Basic, Premium, l, t] = quantity of Basic's demand fulfilled by Premium

Key constraints:
- demand_route: sum_{pt} S[p, pt, l, t] <= demand[p, l, t]
- sales_conservation: y[p] + L[p] = demand[p] + S_in[p] - S_out[p]
```

---

## 8. Expected Impact

| Metric | Before (Code Template) | After (Rules Only) |
|--------|------------------------|-------------------|
| Syntax Error | ~5% | ~10% |
| Runtime Error | ~10% | ~15% |
| INFEASIBLE | ~15% | ~15% |
| **Silent Failure** | ~25% | **~45%** |
| Correct | ~45% | ~15-25% |

**This is intentional** - the increased silent failure rate makes the benchmark more challenging and the Semantic Probe contribution more valuable.

---

## 9. Migration Checklist

```bash
# 1. Backup
cp -r reloop/agents reloop/agents_backup

# 2. Delete obsolete files
rm reloop/agents/tools/iis_tools.py
rm reloop/agents/step_prompts/*.txt  # Will replace all

# 3. Copy new files
cp outputs/step_prompts_v3/*.txt reloop/agents/step_prompts/
cp outputs/orchestrator_graph_v3.py reloop/agents/orchestrator_graph.py
cp outputs/schemas_v3.py reloop/agents/schemas.py
cp outputs/prompt_stack_v3.py reloop/agents/prompt_stack.py
cp outputs/tools/semantic_probes.py reloop/agents/tools/
cp outputs/cli/run_one.py reloop/agents/cli/

# 4. Verify
python -m reloop.agents.cli.run_one --scenario retail_f1_52_weeks_v0 --mock-llm
```
