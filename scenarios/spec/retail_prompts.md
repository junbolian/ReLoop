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
│ Family, archetype, business narrative   │
│                                         │
│ [DATA SCHEMA]                           │
│ JSON structure (types only, not data)   │
│                                         │
│ [OUTPUT FORMAT]                         │
│ GurobiPy, print status and objective    │
│                                         │
│ [INSTRUCTION]                           │
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

**For:** GPT-4o, Claude, Qwen2.5-Coder, SIRL, ORLM, OptiMUS, OptiChat

### Mode 2: ReLoop Agent (Multi-step Pipeline)

**Input:** `{scenario_id}.base.txt` (scenario only)

```
┌─────────────────────────────────────────┐
│ [SCENARIO]                              │
│ Family, archetype, business narrative   │
│ (NO guardrails - injected separately)   │
└─────────────────────────────────────────┘
         ↓
   Step 0: Global Guardrails
   Step 1: Task Contract
   Step 2: Model Specification
   Step 3: Constraint Templates
   Step 4: Code Generation
         ↓
  [Semantic Probes]
         ↓
   Step 5: Repair (if probes fail)
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

### Output Format Section

```
- Output ONLY Python code, no markdown, no explanations
- Use GurobiPy (import gurobipy as gp)
- Set Gurobi params: OutputFlag=0, Threads=1, Seed=0
- Print results:
  print(f"status: {m.Status}")
  if m.Status == 2:  # GRB.OPTIMAL
      print(f"objective: {m.ObjVal}")
```

---

## 4. ReLoop Agent Pipeline

### Pipeline Overview (8 Prompt Modules)

| File | Step | Purpose |
|------|------|---------|
| 00 | Global Guardrails | Core rules, data format, variable naming, substitution semantics |
| 01 | Task Contract | Lock optimization goal, controls, hard/soft constraints |
| 02 | Model Spec Sheet | Define sets, decisions, objective terms, constraint families |
| 03 | Constraint Templates | Derive mathematical LHS/RHS formulas |
| 04 | Code Generation | Translate templates to GurobiPy code |
| 05 | Format Repair (JSON) | Fix malformed JSON output |
| 06 | Format Repair (Code) | Fix code that has markdown or syntax issues |
| 07 | Repair Brief | Diagnose errors and suggest fixes based on probe results |

---

## 5. Semantic Probe Verification

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

### 14 Probes (8 Core + 6 Extended)

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
| 9 | `initialization` | t=1 init (I=0 for a<SL) | Objective = 0 detection |
| 10 | `lead_time` | Lead time handling | Delivery timing |
| 11 | `moq` | Minimum order quantity | MOQ enforcement |
| 12 | `transshipment` | Network flows | Trans constraint |
| 13 | `labor_capacity` | Labor constraints | Capacity check |
| 14 | `holding_cost` | (I-y) vs I | Objective range check |

### Key Insight

Probes test **behavior**, not **code**. They work on any implementation without parsing.

---

## 6. Evaluation Metrics

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

## 7. Key Differences: Baseline vs ReLoop

| Aspect | Baseline (Zero-shot) | ReLoop (Multi-step) |
|--------|---------------------|---------------------|
| Prompt count | 1 minimal prompt | 8 modular prompts |
| Guidance level | Data schema only | Full modeling guidance |
| Interaction | Single LLM call | Multi-step pipeline |
| Error handling | Full regeneration | Targeted repair based on probes |
| Intermediate artifacts | None | Contract → Spec → Templates → Code |
| Verification | Post-hoc probes only | Probes integrated in loop |
