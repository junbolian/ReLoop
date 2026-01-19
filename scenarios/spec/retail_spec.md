# RetailOpt-190: Comprehensive Retail Supply Chain Benchmark

---

## 1. Overview

RetailOpt-190 evaluates **text-to-optimization agents** on **38 retail operations archetypes**, expanded into **190 solver-validated instances** via controlled perturbations.

### Key Features

| Feature | Description |
|---------|-------------|
| **Instances** | 38 archetypes × 5 variants = 190 |
| **Ground Truth** | Universal Retail Solver (URS) |
| **Semantic Probes** | 14 boundary tests via code execution |
| **Compositional Focus** | Tests constraint interactions, not just isolated mechanisms |

All instances share a **single JSON schema** and are solved by a **single universal MILP formulation**.

### Comparison with Existing Benchmarks

| Benchmark | Scenarios | Multi-period | Domain | Compositional | Semantic Probes |
|-----------|-----------|--------------|--------|---------------|-----------------|
| NL4Opt | ~100 | Few | General OR | ✗ | ✗ |
| MAMO | ~800 | Some | General OR | ✗ | ✗ |
| IndustryOR | 100 | Some | Industrial | ✗ | ✗ |
| **RetailOpt-190** | 190 | All | Supply Chain | ✓ | ✓ |

---

## 2. Prompt System

### Two Evaluation Modes

| Mode | Prompt | Used By | Description |
|------|--------|---------|-------------|
| Zero-shot Baseline | `{id}.scenario.txt` | GPT-4o, Claude, Qwen, SIRL, ORLM | Single comprehensive prompt |
| ReLoop Agent | `{id}.base.txt` + step_prompts | ReLoop (our method) | Multi-step pipeline with repair |

### Zero-shot Baseline Prompt Structure

```
[SCENARIO]
├── Family/Archetype/Scenario ID
├── Business narrative
└── Structure cues

[MODELING GUIDELINES]
├── Core rules (data access, no file I/O)
├── Data format (nested network, demand_share)
├── Decision variables (I, y, W, Q, L, S, X, z, n)
├── Objective function (6 cost terms)
├── Substitution semantics (CRITICAL)
├── Core constraints (9 types)
├── Optional constraints (7 types)
└── Boundary conditions (t=1, t=T, lead_time)

[INSTRUCTION]
└── Write complete GurobiPy script...
```

### ReLoop Agent Pipeline

```
Input: {scenario_id}.base.txt (scenario only)
       ↓
Step 0: Global Guardrails
Step 1: Task Contract      → JSON: optimize, controls, constraints
Step 2: Model Specification → JSON: sets, decisions, objective_terms
Step 3: Constraint Templates → JSON: LHS/RHS formulas
Step 4: Code Generation     → Python code
       ↓
[Semantic Probes] → Step 5: Repair (if failed)
       ↓
   Results
```

---

## 3. Evaluation Metrics

| Metric | Definition | Formula |
|--------|------------|---------|
| **Execution Rate** | Code runs without runtime errors | `exec_ok / total` |
| **Accuracy** | Status matches AND objective within 1% | `correct / total` |
| **Silent Failure Rate** | Runs OK but wrong answer | `(exec_ok - correct) / exec_ok` |

### Accuracy Criterion

An instance is **correct** if:
1. Solver status matches ground truth (both feasible, or both infeasible)
2. For feasible instances: $|y_{pred} - y_{ref}| / |y_{ref}| < 1\%$

The 1% tolerance accounts for MIP solver behavior on complex instances (F6/F7), where the 60-second time limit may yield near-optimal solutions.

---

## 4. Silent Failure Problem

### Definition

**Silent Failure:** Code that executes successfully and returns OPTIMAL status, but produces incorrect solutions due to constraint semantic errors.

### Why It Matters

| Failure Type | Detection | Impact |
|--------------|-----------|--------|
| Syntax Error | Automatic | Easy to fix |
| Runtime Error | Automatic | Easy to fix |
| INFEASIBLE | Solver reports | Debuggable |
| **Silent Failure** | **Requires verification** | **Undetectable without probes** |

### Common Causes

| Error Type | Frequency | Example |
|------------|-----------|---------|
| Substitution direction | ~35% | Edge [A,B] misread |
| Missing constraint | ~20% | No demand_route → UNBOUNDED |
| Wrong holding cost | ~20% | Using I instead of I-y |
| Missing initialization | ~15% | No I[p,l,1,a]=0 → obj=0 |
| Boundary errors | ~10% | t=1/t=T edge cases |

---

## 5. Semantic Probes

### How Probes Work (Code Execution, NOT Prompting)

Probes verify constraints by **running the generated code** with specially constructed test data:

```
┌─────────────────────────────────────────────────────────┐
│  1. Construct boundary test data                        │
│     → e.g., zero production for Basic, 80 for Premium   │
│                                                         │
│  2. Execute LLM code via subprocess                     │
│     → subprocess.run([python, "-c", code], env={data})  │
│                                                         │
│  3. Check observable outcomes                           │
│     → UNBOUNDED = missing demand_route constraint       │
│     → Objective too low = wrong substitution direction  │
│     → INFEASIBLE = missing slack variable               │
└─────────────────────────────────────────────────────────┘
```

### 14 Probes (8 Core + 6 Extended)

#### Core Probes

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

#### Extended Probes

| # | Probe | Mechanism | Detection Method |
|---|-------|-----------|------------------|
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

## 6. Substitution Semantics (CRITICAL)

This is the #1 source of silent failures.

### Edge Direction

```
sub_edges: [["SKU_Basic", "SKU_Premium"]]

Meaning: SKU_Basic's demand can be served by SKU_Premium's inventory
         (Premium serves Basic, NOT Basic serves Premium)

Variable: S[Basic, Premium, l, t] = units of Basic's demand fulfilled by Premium
```

### Edge Mappings (Build BEFORE Constraints)

```python
outgoing_edges = {p: [] for p in products}  # Products that can serve MY demand
incoming_edges = {p: [] for p in products}  # Products whose demand I serve
for p_from, p_to in sub_edges:
    outgoing_edges[p_from].append(p_to)  # p_from sends demand OUT to p_to
    incoming_edges[p_to].append(p_from)  # p_to receives requests IN from p_from
```

### Key Constraints

```python
# demand_route: can't substitute more than own demand
for p in products:
    if outgoing_edges[p]:  # Only if p has outgoing edges
        outbound = sum(S[p, pt, l, t] for pt in outgoing_edges[p])
        m.addConstr(outbound <= demand[p, l, t])

# sales_conservation: balance equation
for p in products:
    inbound = sum(S[pf, p, l, t] for pf in incoming_edges[p])
    outbound = sum(S[p, pt, l, t] for pt in outgoing_edges[p])
    m.addConstr(sum_a(y[p,l,t,a]) + L[p,l,t] == demand[p,l,t] + inbound - outbound)
```

---

## 7. Holding Cost (CRITICAL)

### The Bug

LLMs often use `I` (start-of-period inventory) instead of `I-y` (end-of-period inventory):

```python
# WRONG - causes objective to be 60% too low
obj += cost * I[p,l,t,a]

# CORRECT - use end-of-period inventory
obj += cost * (I[p,l,t,a] - y[p,l,t,a])
```

### Why This Matters

- `I[p,l,t,a]` = inventory at START of period (before sales)
- `I[p,l,t,a] - y[p,l,t,a]` = inventory at END of period (after sales)
- Holding cost is charged on what remains overnight, not what you started with
- Only apply to buckets a ≥ 2 (bucket a=1 expires, not held overnight)

---

## 8. Boundary Conditions (CRITICAL)

### Initialization at t=1

```python
# All non-fresh inventory buckets must start empty
for p in products:
    for l in locations:
        for a in range(1, shelf_life[p]):  # a < shelf_life[p]
            m.addConstr(I[p,l,1,a] == 0)
```

**Without this, the model exploits "free" phantom inventory → objective = 0**

### Aging at t=T

```python
# Do NOT add aging constraints for t = T (would reference T+1)
for t in range(1, T):  # t < T, NOT t <= T
    for a in range(1, shelf_life[p]):  # a < shelf_life[p]
        m.addConstr(I[p,l,t+1,a] == I[p,l,t,a+1] - y[p,l,t,a+1])
```

### Fresh Inflow with Lead Time

```python
for t in range(1, T+1):
    if t > lead_time[p]:
        # Fresh = order from lead_time periods ago
        m.addConstr(I[p,l,t,shelf_life[p]] == Q[p,l,t-lead_time[p]])
    else:
        # No fresh inventory yet (orders haven't arrived)
        m.addConstr(I[p,l,t,shelf_life[p]] == 0)
# NEVER access Q[p,l,0] or negative time indices
```

---

## 9. Scenario Families

| Family | Name | Archetypes | Key Mechanisms |
|--------|------|------------|----------------|
| F1 | Core Operations | 4 | Multi-period inventory, seasonal demand, lost sales |
| F2 | Assortment | 6 | Substitution, promotions, ultra-short shelf life |
| F3 | Resources | 4 | Storage bottleneck, production capacity coupling |
| F4 | Dynamics | 6 | Supply disruptions, demand volatility |
| F5 | Feasibility | 4 | Stress tests requiring lost-sales slack |
| F6 | Logistics | 4 | Lead time, MOQ, pack size, fixed order cost |
| F7 | Network & Multi-Echelon | 6 | Transshipment, hub-spoke, three-echelon |
| F8 | Omni-channel | 4 | Returns, labor constraints, sustainability |

**Total:** 38 archetypes × 5 variants = **190 instances**

### Difficulty Progression

- **F1-F4**: Single-mechanism scenarios (baseline difficulty)
- **F5-F8**: Multi-constraint interactions (compositional difficulty)

Expected pattern: All methods similar on F1-F4; ReLoop leads on F5-F8.

---

## 10. Ablation Study Design

### Three Ablation Dimensions

| Dimension | Options |
|-----------|---------|
| **STEP** (Pipeline Depth) | Zero-shot, 5-step |
| **PROBE** (Verification) | No probe, Probe + Diagnosis |
| **REPAIR** (Iteration) | No repair, Guided repair |

### Ablation Configurations

| Config | Steps | Probes | Repair | Description |
|--------|-------|--------|--------|-------------|
| A1 | Zero-shot | No | No | Baseline: single LLM call |
| A2 | 5-step | No | No | Pipeline only |
| A3 | 5-step | Yes | No | + Probe verification |
| A4 | 5-step | Yes | Yes | Full ReLoop |

### Research Questions

| RQ | Question | Ablation |
|----|----------|----------|
| RQ1 | Does step-by-step pipeline improve accuracy? | A1 vs A2 |
| RQ2 | Do probes detect silent failures? | A2 vs A3 |
| RQ3 | Does probe-guided repair improve accuracy? | A3 vs A4 |

---

## 11. JSON Schema

Each instance has this structure:

```json
{
  "name": "retail_f1_base_v0",
  "periods": 20,
  "products": ["SKU_Basic", "SKU_Premium", "SKU_ShortLife"],
  "locations": ["DC1", "DC2", "DC3", "DC4", "DC5"],

  "shelf_life": {"SKU_Basic": 10, "SKU_Premium": 8, "SKU_ShortLife": 4},
  "lead_time": {"SKU_Basic": 0, "SKU_Premium": 0, "SKU_ShortLife": 0},

  "demand_curve": {
    "SKU_Basic": [303, 311, 328, ...],
    "SKU_Premium": [151, 155, 164, ...]
  },
  "demand_share": {"DC1": 0.25, "DC2": 0.2, "DC3": 0.2, "DC4": 0.2, "DC5": 0.15},

  "production_cap": {"SKU_Basic": [800, 800, ...], ...},
  "cold_capacity": {"DC1": 4000, "DC2": 3500, ...},
  "cold_usage": {"SKU_Basic": 1.0, "SKU_Premium": 3.0, ...},

  "labor_cap": {"DC1": [99999.0, ...], ...},
  "labor_usage": {"SKU_Basic": 0.0, ...},
  "return_rate": {"SKU_Basic": 0.0, ...},

  "costs": {
    "purchasing": {"SKU_Basic": 10.0, ...},
    "inventory": {"SKU_Basic": 1.0, ...},
    "waste": {"SKU_Basic": 2.0, ...},
    "lost_sales": {"SKU_Basic": 50.0, ...},
    "fixed_order": 0.0,
    "transshipment": 0.5
  },

  "constraints": {
    "moq": 0,
    "pack_size": 1,
    "budget_per_period": null,
    "waste_limit_pct": null
  },

  "network": {
    "sub_edges": [["SKU_Basic", "SKU_Premium"]],
    "trans_edges": []
  }
}
```

### Data Access Patterns

```python
# Network data is NESTED - use safe access
sub_edges = data.get('network', {}).get('sub_edges', [])
trans_edges = data.get('network', {}).get('trans_edges', [])
# DO NOT use data['sub_edges'] directly - causes KeyError!

# demand_share is location-only (NOT nested by product)
demand[p,l,t] = data['demand_curve'][p][t-1] * data['demand_share'][l]

# production_cap is 0-indexed list
prod_cap = data['production_cap'][p][t-1]  # Access with [t-1]

# Optional fields - use .get() with defaults
moq = data.get('constraints', {}).get('moq', 0)
fixed_order = data.get('costs', {}).get('fixed_order', 0)
```

---

## 12. Solver Settings

| Parameter | Value | Purpose |
|-----------|-------|---------|
| TimeLimit | 60s | Prevent stalling on complex MIPs |
| MIPGap | 1% | Tolerance for near-optimal solutions |
| OutputFlag | 0 | Suppress solver logs |
| Threads | 1 | Reproducibility |
| Seed | 0 | Reproducibility |

### Status Mapping

| Status | Meaning | Has Solution |
|--------|---------|--------------|
| OPTIMAL | Proven optimal | ✓ |
| TIME_LIMIT | Time limit with solution | ✓ (near-optimal) |
| INFEASIBLE | No feasible solution | ✗ |

```