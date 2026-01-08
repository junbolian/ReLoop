# RetailOpt-190: Prompt System Documentation

---

## 1. Prompt Architecture

RetailOpt-190 uses **two prompt formats** for different evaluation modes:

### Two Prompt Files Per Scenario

| File | Content | Used By |
|------|---------|---------|
| `{scenario_id}.scenario.txt` | Scenario + guardrails + instructions | Zero-shot baseline (single LLM call) |
| `{scenario_id}.base.txt` | Scenario description only | ReLoop Agent (guardrails injected by step_prompts) |

---

## 2. Evaluation Modes

### Mode 1: Zero-shot Baseline (All Baseline Models)

**Input:** `{scenario_id}.scenario.txt` (complete prompt)

```
┌─────────────────────────────────────────┐
│ [SCENARIO]                              │
│ Family, archetype, business narrative   │
│ Structure cues                          │
│                                         │
│ [MODELING GUIDELINES]                   │
│ Core rules, data format, variables      │
│ Objective function, substitution        │
│ Constraints, boundary conditions        │
│                                         │
│ [DATA ACCESS]                           │
│ Key fields documentation                │
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

### Section 1: Scenario Description

```
[SCENARIO]
Family: {family_id} ({family_name})
Archetype: {archetype_id}
Scenario ID: {scenario_id}

Business narrative:
{business_narrative}

Structure cues:
{structure_cues}
```

### Section 2: Modeling Guidelines

#### Core Rules

```
[CORE RULES]
- `data` is a pre-loaded Python dict. Do not modify it.
- No file I/O. Never invent missing data.
- Never hard-code numeric values.
- Output must be plain Python code. No prose, no markdown, no comments.
```

#### Data Format

```
[DATA FORMAT]
- Network data is NESTED - use safe access:
  sub_edges = data.get('network', {}).get('sub_edges', [])
  trans_edges = data.get('network', {}).get('trans_edges', [])
- DO NOT use data['sub_edges'] directly - this will cause KeyError!
- demand_share: {location: scalar}, NOT nested by product
- demand[p,l,t] = demand_curve[p][t-1] * demand_share[l]
- Time indexing: 1-based (t = 1, 2, ..., T)
- production_cap[p] is a list (0-indexed), access with [t-1]
```

#### Decision Variables

```
[DECISION VARIABLES]
- I[p,l,t,a]: inventory by product, location, period, remaining life bucket
- y[p,l,t,a]: sales from each life bucket
- W[p,l,t]: waste (expired inventory from bucket a=1)
- Q[p,l,t]: orders/production quantity
- L[p,l,t]: lost sales (slack variable for unmet demand)
- S[p_from,p_to,l,t]: substitution flow (only if sub_edges nonempty)
- X[p,l_from,l_to,t]: transshipment flow (only if trans_edges nonempty)
- z[p,l,t]: binary order indicator (only if moq > 0 or fixed_order > 0)
- n[p,l,t]: integer pack count (only if pack_size > 1)
```

#### Objective Function

```
[OBJECTIVE FUNCTION]
Minimize total cost including:
1. PURCHASING COST: costs["purchasing"][p] * Q[p,l,t]
2. INVENTORY HOLDING: costs["inventory"][p] * (I[p,l,t,a] - y[p,l,t,a])
   - Apply to END-OF-PERIOD inventory (I - y), NOT just I
   - Apply ONLY to buckets a >= 2 (a=1 expires, not held overnight)
3. WASTE COST: costs["waste"][p] * W[p,l,t]
4. LOST SALES: costs["lost_sales"][p] * L[p,l,t]
5. TRANSSHIPMENT (if trans_edges): costs["transshipment"] * X
6. FIXED ORDER (if z exists): costs["fixed_order"] * z
```

### Section 3: Substitution Semantics (CRITICAL)

```
SUBSTITUTION SEMANTICS (CRITICAL)

Edge [p_from, p_to] means: p_from's demand can be served by p_to's inventory.
This is "upward substitution" - premium product serves basic product's demand.

S[p_from, p_to, l, t] = quantity of p_from's demand fulfilled by p_to

Example: sub_edges = [["Basic", "Premium"]]
- Basic can have its demand met by Premium's inventory
- Premium CANNOT have its demand met by Basic (no reverse edge)

For each product p, compute:
- outbound: total substitution flow where p is the source (p sends demand out)
- inbound: total substitution flow where p is the target (p receives demand in)

Two key constraints involving substitution:
1. demand_route: Cannot substitute more demand than the product actually has
2. sales_conservation: Total sales + lost = demand + inbound - outbound
```

### Section 4: Boundary Conditions

```
BOUNDARY CONDITIONS SUMMARY

- t = 1: Initialize I[p,l,1,a] = 0 for a < shelf_life[p]
  (CRITICAL - without this, model exploits "free" inventory → objective = 0)
  
- t = T: No aging constraints (would reference t+1)

- t <= lead_time: Fresh inflow = 0 (orders haven't arrived yet)
  NEVER access Q[p,l,0] or negative time indices

- Empty sub_edges: No S variables, no substitution constraints

- Empty trans_edges: No X variables, no transshipment constraints
```

### Section 5: Instruction

```
[INSTRUCTION]
Write a complete GurobiPy script that:
1) Imports gurobipy (import gurobipy as gp; from gurobipy import GRB)
2) Reads all parameters from `data` (already loaded)
3) Creates all necessary decision variables with correct indices
4) Sets objective function with all applicable cost terms
5) Adds all constraints respecting boundary conditions
6) Handles optional constraints based on what exists in data
7) Sets Gurobi params: OutputFlag=0, Threads=1, Seed=0
8) Prints status and objective (if OPTIMAL)

Return ONLY Python code. No markdown, no comments, no explanations.
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

### Step 0: Global Guardrails

Same content as baseline [CORE RULES], [DATA FORMAT], and [SUBSTITUTION SEMANTICS].

### Step 1: Task Contract

```json
{
  "optimize": "minimize total cost over horizon",
  "controls": ["order quantities", "inventory allocation", "substitution flows"],
  "hard_constraints": ["inventory flow", "capacity limits", "perishability"],
  "soft_violations": [{"name": "lost_sales", "penalty": "costs.lost_sales"}],
  "summary": "one sentence"
}
```

### Step 2: Model Specification

```json
{
  "sets": [{"name": "P", "description": "products", "source": "data.products"}],
  "decisions": [{"name": "I", "type": "continuous", "indices": ["p","l","t","a"], 
                 "meaning": "start-of-period inventory by remaining life"}],
  "objective_terms": [{"name": "holding_cost", 
                       "expression": "cost of END-OF-PERIOD inventory (I-y) for a>=2"}],
  "constraint_families": [{"prefix": "sales_conservation", 
                           "meaning": "sales + lost = effective demand"}]
}
```

### Step 3: Constraint Templates

```json
{
  "prefix": "sales_conservation",
  "template_type": "BALANCE",
  "indices": ["p","l","t"],
  "equations": [{"lhs": "sum_a(y[p,l,t,a]) + L[p,l,t]", 
                 "rhs": "D[p,l,t] + S_in[p,l,t] - S_out[p,l,t]", 
                 "sense": "="}],
  "notes": ["L must be included", "D = demand_curve[p][t-1] * demand_share[l]"]
}
```

### Step 4: Code Generation

Translates constraint templates into executable GurobiPy code. Emphasizes:
- Safe data access patterns (nested network data)
- Boundary condition handling (t=1 init, t=T no aging, lead time)
- Optional feature detection (MOQ, pack size, transshipment, etc.)

### Step 5: Repair Brief

When errors occur, diagnose and suggest fixes:

```json
{
  "target": "SPEC|CODEGEN",
  "error_type": "INFEASIBLE|UNBOUNDED|RUNTIME|PROBE_FAILURE|WRONG_ANSWER",
  "likely_cause": "one sentence",
  "fix": "one sentence",
  "failed_probes": ["initialization", "aging_dynamics"]
}
```

---

## 5. Semantic Probe Verification

### How Probes Work (Code Execution, NOT Prompting)

Probes verify constraints via **actual code execution**:

```
┌─────────────────────────────────────────────────────────┐
│  Step 1: Construct boundary test data                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ data = {                                         │   │
│  │   "demand": {"Basic": 100, "Premium": 0},       │   │
│  │   "production_cap": {"Basic": 0, "Premium": 80} │   │
│  │ }                                                │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  Step 2: Execute LLM code via subprocess                │
│  ┌─────────────────────────────────────────────────┐   │
│  │ result = subprocess.run(                         │   │
│  │   [python, "-c", llm_generated_code],           │   │
│  │   env={"DATA": probe_data}                      │   │
│  │ )                                                │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  Step 3: Check observable outcomes                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │ if status == UNBOUNDED → missing constraint     │   │
│  │ if objective < expected → wrong implementation  │   │
│  │ if objective in range → PASS                    │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
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
| 14 | `waste_limit` | Global waste cap | Waste cap enforcement |

### Critical Probes (Silent Failure Detection)

| Probe | Common Error | Symptom |
|-------|--------------|---------|
| `initialization` | Missing `I[p,l,1,a]=0` for a < shelf_life | Objective ≈ 0 (free inventory) |
| `holding_cost` | Using `I` instead of `I-y` | Objective 60% too low |

### Key Insight

Probes test **behavior**, not **code**. They work on any implementation without parsing.

---

## 6. Common Error Patterns and Fixes

### Error Diagnosis Guide

| Error Type | Symptom | Likely Cause | Fix |
|------------|---------|--------------|-----|
| INFEASIBLE | Solver status | Missing L in sales_conservation | Add L variable with lost_sales penalty |
| UNBOUNDED | Solver status | Missing demand_route constraint | Add outbound ≤ demand constraint |
| Objective = 0 | Multiple probes fail | Missing t=1 initialization | Add I[p,l,1,a]=0 for a < shelf_life |
| Objective too low | holding_cost probe fails | Using I instead of (I-y) | Apply holding to end-of-period inventory |
| Wrong answer | Objective differs >1% | Substitution direction wrong | Check edge [A,B] means A→B, not B→A |

### Objective = 0 Diagnosis (CRITICAL)

If multiple probes show objective = 0, check in order:

1. **Missing initialization at t=1?**
   - Without I[p,l,1,a]=0 for a<shelf_life, model gets "free" inventory
   - This is the MOST COMMON cause of obj=0

2. **Missing m.setObjective() call?**
   - Objective must be set before optimize()

3. **Empty objective expression?**
   - Must accumulate costs in a variable before setting objective

4. **Costs not read from data?**
   - Use data["costs"]["lost_sales"][p], etc.

---

## 7. Evaluation Metrics

| Metric | Definition | Formula |
|--------|------------|---------|
| **Execution Rate** | Code runs without runtime errors | `exec_ok / total` |
| **Accuracy** | Status matches AND objective within 1% | `correct / total` |
| **Silent Failure Rate** | Runs OK but wrong answer | `(exec_ok - correct) / exec_ok` |

### Accuracy Criterion

An instance is **correct** if:
1. Solver status matches ground truth (both feasible, or both infeasible)
2. For feasible instances: |y_pred - y_ref| / |y_ref| < 1%

The 1% tolerance accounts for MIP solver behavior on complex instances (F6/F7 families).

---

## 8. Key Differences: Baseline vs ReLoop

| Aspect | Baseline (Zero-shot) | ReLoop (Multi-step) |
|--------|---------------------|---------------------|
| Prompt count | 1 complete prompt | 8 modular prompts |
| Interaction | Single LLM call | Multi-step pipeline |
| Error handling | Full regeneration | Targeted repair based on probes |
| Intermediate artifacts | None | Contract → Spec → Templates → Code |
| Verification | Post-hoc probes only | Probes integrated in loop |

### ReLoop Advantages

1. **Structured reasoning**: Each step focuses on specific aspect (semantics → math → code)
2. **Verifiable artifacts**: Intermediate outputs can be checked before proceeding
3. **Targeted repair**: Failed probes identify WHICH constraint is wrong, enabling precise fixes

```