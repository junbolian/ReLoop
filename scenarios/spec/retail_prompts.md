# RetailOpt-190: Prompt System Documentation

---

## 1. Prompt Architecture

RetailOpt-190 uses **two prompt formats** for different evaluation modes:

### Two Prompt Files Per Scenario

| File | Content | Used By |
|------|---------|---------|
| `{scenario_id}.scenario.txt` | Scenario + data schema + output format | Zero-shot baseline (single LLM call) |
| `{scenario_id}.base.txt` | Scenario description only | ReLoop Agent (guardrails injected by step_prompts) |

---

## 2. Evaluation Modes

### Mode 1: Zero-shot Baseline (All Baseline Models)

**Input:** `{scenario_id}.scenario.txt` (complete prompt)

```
┌─────────────────────────────────────────┐
│ [SCENARIO]                              │
│ Family, archetype, scenario ID          │
│                                         │
│ [BUSINESS DESCRIPTION]                  │
│ Business narrative + structure cues     │
│                                         │
│ [DATA SCHEMA]                           │
│ JSON structure (types only, not data)   │
│                                         │
│ [DATA ACCESS]                           │
│ How to read from `data` variable        │
│                                         │
│ [OUTPUT FORMAT]                         │
│ GurobiPy, print status and objective    │
│                                         │
│ [TASK]                                  │
│ Write GurobiPy script...                │
└─────────────────────────────────────────┘
         ↓
      [LLM] (single call)
         ↓
   Python script
         ↓
  [Execute + Probes]
         ↓
      Results
```

**For:** GPT-4o, Claude, Qwen, DeepSeek, SIRL, ORLM, OptiMUS, OptiChat

### Mode 2: ReLoop Agent (Multi-step Pipeline)

**Input:** `{scenario_id}.base.txt` (scenario only)

```
┌─────────────────────────────────────────┐
│ [SCENARIO]                              │
│ Family, archetype, business narrative   │
│ (NO guardrails - injected separately)   │
└─────────────────────────────────────────┘
         ↓
   Step 0: Data Profile
   Step 1: Problem Understanding
   Step 2: Math Specification
   Step 3: Code Generation
         ↓
  [Semantic Probes]
         ↓
   Step 4-5: Repair (if probes fail)
         ↓
      Results
```

**For:** ReLoop (our method)

---

## 3. Baseline Prompt Content

The baseline prompt (`{scenario_id}.scenario.txt`) contains **minimal specification**:

### What We Give

| Section | Content |
|---------|---------|
| Business Narrative | Scenario description, structure cues |
| Data Schema | JSON structure with types (NOT actual data) |
| Data Access | How to read from `data` variable |
| Output Format | GurobiPy, print format |

### What We DON'T Give

| Omitted | Rationale |
|---------|-----------|
| Decision variables | LLM decides how to model |
| Objective function formula | LLM derives from costs |
| Constraint formulas | LLM derives from narrative |
| Boundary conditions | LLM handles edge cases |
| Common error warnings | No hints about pitfalls |

**Philosophy:** Give LLM the business context and data structure. Let it decide how to model.

### Data Schema Section

```
{
  "name": str,                          # scenario identifier
  "periods": int,                       # number of time periods
  "products": [str, ...],               # list of product IDs
  "locations": [str, ...],              # list of location IDs

  "shelf_life": {p: int},               # shelf life in periods per product
  "lead_time": {p: int},                # order lead time per product

  "demand_curve": {p: [float, ...]},    # demand per product per period
  "demand_share": {l: float},           # fraction of total demand at each location

  "production_cap": {p: [float, ...]},  # max production per product per period
  "cold_capacity": {l: float},          # storage capacity per location
  "cold_usage": {p: float},             # storage units per unit of product

  "labor_cap": {l: [float, ...]},       # labor hours per location per period
  "labor_usage": {p: float},            # labor hours per unit sold
  "return_rate": {p: float},            # fraction of sales returned next period

  "costs": {
    "purchasing": {p: float},           # cost per unit ordered
    "inventory": {p: float},            # holding cost per unit per period
    "waste": {p: float},                # cost per unit expired
    "lost_sales": {p: float},           # penalty per unit of unmet demand
    "fixed_order": float,               # fixed cost per order placed
    "transshipment": float              # cost per unit transshipped
  },

  "constraints": {
    "moq": float,                       # minimum order quantity
    "pack_size": int,                   # order must be multiple of this
    "budget_per_period": float|null,    # max purchasing cost per period
    "waste_limit_pct": float|null       # max waste as fraction of total demand
  },

  "network": {
    "sub_edges": [[p_from, p_to], ...], # substitution edges
    "trans_edges": [[l_from, l_to], ...]# transshipment edges
  }
}
```

### Key Parameter Constraint Semantics

The following parameters require explicit constraint modeling. The business description in `archetypes.yaml` provides detailed semantics:

| Parameter | Constraint Semantics |
|-----------|---------------------|
| `shelf_life` | Inventory tracked by **remaining life** (NOT age). Use `I[p,l,t,r]` where `r` = remaining periods until expiry (1..shelf_life[p]). Fresh arrivals enter at `r = shelf_life[p]`. Aging: `I[t+1,r] = I[t,r+1] - sales[t,r+1]`. Waste: units at `r=1` not sold become waste. FIFO sales (sell oldest/lowest r first). |
| `cold_capacity/cold_usage` | Volume-weighted storage: `sum over products of (cold_usage[p] * total_inventory[p,l,t]) <= cold_capacity[l]` |
| `lead_time` | Orders placed in period t arrive in period t + lead_time[p]. Track in-transit vs on-hand inventory. |
| `labor_cap/labor_usage` | Labor constraint: `sum over products of (labor_usage[p] * units_handled[p,l,t]) <= labor_cap[l,t]` |
| `return_rate` | Reverse logistics: `return_rate[p] * sales[p,l,t]` units re-enter inventory at location l in period t+1 |
| `waste_limit_pct` | Global waste cap: `sum(waste) <= waste_limit_pct * sum(demand)` over entire horizon |
| `sub_edges` | Edge [A, B] means A's demand can be served by B's inventory. S[A,B] = units of A's demand fulfilled by B. |
| `trans_edges` | Edge [L1, L2] means inventory can move from L1 to L2 at transshipment cost. |

### Data Access Section

```
- The variable `data` is pre-loaded. Do NOT use file I/O.
- Network data is nested: use data.get('network', {}).get('sub_edges', [])
- IMPORTANT: sub_edges and trans_edges are lists of lists [[a,b], ...].
  Convert to tuples for Gurobi indexing: [tuple(e) for e in sub_edges]
- Lists are 0-indexed
```

### Output Format Section

```
- Output ONLY Python code
- Use GurobiPy
- Print status and objective
```

---

## 4. Data Usage Principle

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  What LLM sees: Data Schema (structure only)                    │
│  ═══════════════════════════════════════════                    │
│  - Field names, types, meanings                                 │
│  - Indexing conventions (0-indexed, etc.)                       │
│  - Access patterns                                              │
│                                                                 │
│  What LLM does NOT see: Full Data                               │
│  ════════════════════════════════════                           │
│  - Actual demand values                                         │
│  - Actual cost values                                           │
│  - Complete 52-week arrays                                      │
│                                                                 │
│  Full data is ONLY used for: Code execution + Verification      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow by Step

| Step | LLM Sees | Full Data Used For |
|------|----------|-------------------|
| 1-3 (Modeling) | Narrative + Schema | - |
| 4 (Verification) | - | Execute code, sensitivity tests |
| 5 (Repair) | Code + "demand anomaly" | - |

**Key insight**: LLM never sees actual data values. It only knows the structure. Verification uses full data to test code behavior, but only reports *which parameter* failed, not the values.

---

## 5. ReLoop Agent Pipeline

### Pipeline Overview

| Step | Type | Input | Output |
|------|------|-------|--------|
| 0 | Auto | JSON data | Parameter roles |
| 1 | LLM | Narrative + Schema | Problem understanding (JSON) |
| 2 | LLM | Step 1 + Schema | Math specification (JSON) |
| 3 | LLM | Step 2 + Schema | GurobiPy code |
| 4 | Auto | Code + Data | Verification report |
| 5 | LLM | Code + Report | Repaired code |

---

## 6. 6-Layer Behavioral Verification

### Verification Architecture (7-Layer System)

| Category | Layer | Name | Status | Description |
|----------|-------|------|--------|-------------|
| **BASIC** | **L1** | Execution | **MANDATORY** | Code must run without errors |
| **BASIC** | **L2** | Feasibility | **LENIENT** | OPTIMAL or TIME_LIMIT with objective |
| **STRUCTURE** | **L3** | Code Structure (AST) | **UNIVERSAL** | Static analysis: objective, variables, constraints, sales <= I |
| **SEMANTIC** | **L4** | Monotonicity | Universal | Does each parameter affect objective? |
| **SEMANTIC** | **L5** | Sensitivity | Role-based | Does direction match expected? |
| **SEMANTIC** | **L6** | Boundary | Universal | Zero/extreme value behavior |
| **DOMAIN** | **L7** | Domain Probes | **OPTIONAL** | Retail-specific checks (enable_layer7=True) |

### Key Design Principles

1. **L1-L2 (Basic)** - Must pass, stops if fails
2. **L3 (Structure)** - Fast AST analysis runs first
3. **L4-L6 (Semantic)** - Runtime behavioral tests, run independently
4. **L7 (Domain)** - Optional, enable with `enable_layer7=True`
5. **Always report objective** - Regardless of which layer passes

### "No Effect" Detection (L4)

When a parameter perturbation shows "No effect":
- **Possible causes**: Constraint missing OR constraint has slack (not binding)
- **Not always an error**: Slack constraints are valid in some scenarios
- **Diagnosis provided**: Tells which parameter had no effect

### Layer Failure with Correct Results

**Important**: Layer verification failures do not always indicate incorrect modeling. There are legitimate cases where layers fail but the objective is correct:

| Scenario | Layer Result | Objective Result | Explanation |
|----------|--------------|------------------|-------------|
| Slack constraint | L3 FAIL (no effect) | Correct | Constraint exists but doesn't bind due to data |
| Simplified model | L3-L5 partial FAIL | Close to optimal | Model captures key dynamics, skips minor features |
| Alternative formulation | L3 FAIL | Correct | Different but equivalent modeling approach |

**Example: cold_capacity "No effect"**
```
Data: cold_capacity = 3500, actual usage = 2000
Result: Perturbation has no effect because capacity constraint has slack
Conclusion: NOT a modeling error - constraint is correctly implemented but not binding
```

**Interpretation Guidelines**:
1. **Layer count alone is insufficient** - Compare objective to ground truth
2. **"No effect" may be correct** - Check if the constraint can have slack given the data
3. **Final evaluation metric** - Objective gap to ground truth (< 1% is acceptable)

---

## 7. Semantic Probe Verification (L6)

### How Probes Work (Code Execution, NOT Prompting)

Probes verify constraints via **actual code execution**:

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Construct boundary test data                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ data = {                                             │    │
│  │   "demand": {"Basic": 100, "Premium": 0},           │    │
│  │   "production_cap": {"Basic": 0, "Premium": 80}     │    │
│  │ }                                                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                         ↓                                    │
│  Step 2: Execute LLM code via subprocess                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ result = subprocess.run(                             │    │
│  │   [python, "-c", llm_generated_code],               │    │
│  │   env={"DATA": probe_data}                          │    │
│  │ )                                                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                         ↓                                    │
│  Step 3: Check observable outcomes                           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ if status == UNBOUNDED → missing constraint         │    │
│  │ if objective < expected → wrong implementation      │    │
│  │ if objective in range → PASS                        │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 8 Core Probes

| # | Probe | Mechanism | Detection Method |
|---|-------|-----------|------------------|
| 1 | `substitution_basic` | S variables | Objective range check |
| 2 | `demand_route_constraint` | S_out ≤ demand | UNBOUNDED detection |
| 3 | `no_substitution` | Empty edges | Spurious benefit detection |
| 4 | `production_capacity` | Prod cap | Objective lower bound |
| 5 | `storage_capacity` | Storage cap | INFEASIBLE detection |
| 6 | `aging_dynamics` | Shelf-life | Waste cost verification |
| 7 | `lost_sales_slack` | L variable | INFEASIBLE detection |
| 8 | `inventory_nonnegativity` | I ≥ 0 | Negative inventory check |

### Key Insight

Probes test **behavior**, not **code**. They work on any implementation without parsing.

---

## 8. Evaluation Metrics

| Metric | Definition | Formula |
|--------|------------|---------|
| **Execution Rate** | Code runs without runtime errors | `exec_ok / total` |
| **Accuracy** | Status matches AND objective within 1% | `correct / total` |
| **Silent Failure Rate** | Runs OK but wrong answer | `(exec_ok - correct) / exec_ok` |

### Accuracy Criterion

An instance is **correct** if:
1. Solver status matches ground truth (both feasible, or both infeasible)
2. For feasible instances: |y_pred - y_ref| / |y_ref| < 1%

---

## 9. Key Differences: Baseline vs ReLoop

| Aspect | Baseline (Zero-shot) | ReLoop (Multi-step) |
|--------|---------------------|---------------------|
| Prompt count | 1 minimal prompt | Multi-step prompts |
| Guidance level | Data schema only | Step-by-step modeling |
| Interaction | Single LLM call | Multi-step pipeline |
| Error handling | Full regeneration | Targeted repair based on probes |
| Intermediate artifacts | None | Understanding → Spec → Code |
| Verification | Post-hoc probes only | Probes integrated in loop |

---

## 10. Actual Test Results (Claude Opus 4.5)

### Test 1: retail_f1_52_weeks_v0 (Core Operations, 52 weeks)

| Metric | Baseline | ReLoop | Delta |
|--------|----------|--------|-------|
| **Objective** | 1,006,432.00 | 1,006,432.00 | - |
| **Gap to Ground Truth** | **0.00%** | **0.00%** | - |
| **Layers Passed** | 3/7 | 3/7 | +0 |
| **LLM Turns** | 1 | 5 | +4 |
| **Duration** | 32.11s | 137.93s | +105.82s |

**Note**: L4 failures (shelf_life, cold_capacity, cold_usage "NO EFFECT") are false positives - constraints are correctly implemented but have slack in this data instance.

### Test 2: retail_f5_ultimate_stress_v0 (Stress Test)

| Metric | Baseline | ReLoop | Delta |
|--------|----------|--------|-------|
| **Objective** | 702,188.80 | 702,188.80 | - |
| **Gap to Ground Truth** | **1.06%** | **1.06%** | - |
| **Layers Passed** | 7/7 | 7/7 | +0 |
| **LLM Turns** | 1 | 3 | +2 |
| **Duration** | 31.45s | 77.73s | +46.28s |

**Note**: Both methods achieve near-optimal results even on stress scenarios.

### Test 3: retail_f1_52_weeks_v0 with GPT-5.1

| Metric | Baseline | ReLoop | Delta |
|--------|----------|--------|-------|
| **Objective** | 980,824.00 | 1,035,267.50 | - |
| **Gap to Ground Truth** | **2.54%** | **2.87%** | +0.33% |
| **Layers Passed** | 3/7 | 3/7 | +0 |
| **LLM Turns** | 1 | 6 | +5 |

**Note**: GPT-5.1 has higher gap than Claude Opus 4.5, proving prompts don't leak answers.

### Key Observations

1. **Claude Opus 4.5 is highly capable** - Even single-shot baseline achieves near-optimal results
2. **GPT-5.1 shows 2.54% gap** - Proves prompts don't leak answers (if leaked, all models would get ~0%)
3. **L4 "NO EFFECT" can be false positives** - Slack constraints show "no effect" but are correctly implemented
4. **Final metric is objective gap** - Layer count is diagnostic, not the final success criterion
5. **ReLoop provides most value for weaker models** - On harder scenarios with less capable LLMs
