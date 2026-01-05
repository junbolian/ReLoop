# RetailOpt-190: Comprehensive Retail Supply Chain Benchmark

> File: `reloop/scenarios/spec/retail_spec.md`
> Depends on:
>
> * `reloop/solvers/universal_retail_solver.py`
> * `reloop/tools/retail_benchmark_generator.py`
> * `reloop/agents/` (ReLoop Agent)
> * `reloop/eval/run_benchmark.py`

---

## 1. Overview

RetailOpt-190 evaluates **text-to-optimization agents** on **38 retail operations archetypes**, expanded into **190 solver-validated instances** via controlled perturbations.

### Key Features

| Feature | Description |
|---------|-------------|
| **Instances** | 38 archetypes × 5 variants = 190 |
| **Ground Truth** | Universal Retail Solver (URS) |
| **Semantic Probes** | 8 boundary tests via code execution |
| **No Code Template** | Tests actual math→code translation |

All instances share a **single JSON schema** and are solved by a **single universal MILP formulation**.

### Recommended Model

**Primary:** Qwen-Max (via DashScope API)
- Strong reasoning + code generation
- Best balance of capability and cost

**Alternatives:** Qwen2.5-Coder-32B, GPT-4o, DeepSeek-V3

---

## 2. Prompt System

### Two Files Per Scenario

| File | Content | Used By |
|------|---------|---------|
| `{id}.base.txt` | Scenario only | ReLoop Agent |
| `{id}.scenario.txt` | Scenario + guardrails + instructions | Zero-shot baseline |

### Generation

```bash
python -m reloop.tools.generate_prompts
# Output: scenarios/prompts/{scenario_id}.base.txt
#         scenarios/prompts/{scenario_id}.scenario.txt
```

---

## 3. Evaluation Modes

### Mode 1: Zero-shot Baseline

```
Input: {scenario_id}.scenario.txt (complete prompt)
       ↓
    [LLM] (single call)
       ↓
  Python script
       ↓
[Execute + Semantic Probes]
       ↓
   Results
```

**For:** All baseline models (GPT-4, Claude, Qwen, DeepSeek)

### Mode 2: ReLoop Agent (Multi-step with Repair)

```
Input: {scenario_id}.base.txt (scenario only)
       ↓
profile_data → step1 → step2 → step3 → sanity → step4 → audit → probe → run
               contract spec   templates        codegen        verify
                                                   ↑______________|
                                                   (repair loop)
```

**For:** Qwen-Max primary experiments

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
| Substitution direction | ~45% | Edge [A,B] misread |
| Missing constraint | ~25% | No demand_route |
| Wrong indexing | ~20% | Cost on wrong buckets |
| Boundary errors | ~10% | t=1 initialization |

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

### 8 Probes

| Probe | Mechanism | Detection Method |
|-------|-----------|------------------|
| `substitution_basic` | S variables | Objective range check |
| `demand_route_constraint` | S_out ≤ demand | UNBOUNDED detection |
| `no_substitution` | Empty edges | Spurious benefit detection |
| `production_capacity` | Prod cap | Objective lower bound |
| `storage_capacity` | Storage cap | INFEASIBLE detection |
| `aging_dynamics` | Shelf-life | Waste cost verification |
| `lost_sales_slack` | L variable | INFEASIBLE detection |
| `nonnegativity` | I ≥ 0 | Negative inventory check |

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
outgoing_edges = {p: [] for p in products}
incoming_edges = {p: [] for p in products}
for p_from, p_to in sub_edges:
    outgoing_edges[p_from].append(p_to)  # p_from sends demand OUT to p_to
    incoming_edges[p_to].append(p_from)  # p_to receives requests IN from p_from
```

### Key Constraints

```python
# demand_route: can't substitute more than own demand
for p in products:
    outbound = sum(S[p, pt, l, t] for pt in outgoing_edges[p])
    m.addConstr(outbound <= demand[p, l, t])

# sales_conservation: balance equation
for p in products:
    inbound = sum(S[pf, p, l, t] for pf in incoming_edges[p])
    outbound = sum(S[p, pt, l, t] for pt in outgoing_edges[p])
    m.addConstr(sum_a(y[p,l,t,a]) + L[p,l,t] == demand[p,l,t] + inbound - outbound)
```

---

## 7. Scenario Families

| Family | Name | Archetypes | Key Mechanisms |
|--------|------|------------|----------------|
| F1 | Operations | 4 | Inventory, demand, lost sales |
| F2 | Assortment | 6 | Substitution, promotions |
| F3 | Resources | 4 | Storage, production capacity |
| F4 | Dynamics | 6 | Shelf-life, disruptions |
| F5 | Feasibility | 4 | Stress tests, slack |
| F6 | Logistics | 4 | MOQ, pack size, lead time |
| F7 | Network | 6 | Transshipment, multi-echelon |
| F8 | Omni-channel | 4 | Returns, labor, sustainability |

**Total:** 38 archetypes × 5 variants = **190 instances**

---

## 8. Directory Structure

```
reloop/
├── solvers/
│   └── universal_retail_solver.py     # Reference MILP (ground truth)
│
├── agents/
│   ├── orchestrator_graph.py          # LangGraph state machine
│   ├── schemas.py                     # Pydantic data models
│   ├── step_prompts/                  # 8 prompt files
│   └── tools/
│       ├── semantic_probes.py         # 8 probes implementation
│       ├── sanity_checker.py          # Logic validation
│       └── static_auditor.py          # Pattern checking
│
├── tools/
│   ├── generate_prompts.py            # Prompt generator
│   └── retail_benchmark_generator.py  # Instance generator
│
├── scenarios/
│   ├── spec/
│   │   ├── retail_spec.md             # This file
│   │   ├── retail_prompts.md          # Prompt documentation
│   │   └── archetypes.yaml            # 38 archetype definitions
│   ├── data/                          # 190 JSON instances
│   └── prompts/                       # Generated prompts
│       ├── {id}.base.txt              # For ReLoop Agent
│       └── {id}.scenario.txt          # For Zero-shot
│
└── eval/
    └── run_benchmark.py               # Evaluation script
```

---

## 9. JSON Schema

Each instance has this structure:

```json
{
  "scenario_id": "retail_f1_52_weeks_v0",
  "periods": 52,
  "products": ["SKU_Basic", "SKU_Premium", "SKU_ShortLife"],
  "locations": ["DC1", "DC2", "DC3"],
  
  "shelf_life": {"SKU_Basic": 10, "SKU_Premium": 8, "SKU_ShortLife": 4},
  "lead_time": {"SKU_Basic": 0, "SKU_Premium": 0, "SKU_ShortLife": 0},
  
  "demand_curve": {
    "SKU_Basic": [100, 120, ...],
    "SKU_Premium": [50, 60, ...]
  },
  "demand_share": {"DC1": 0.4, "DC2": 0.35, "DC3": 0.25},
  
  "network": {
    "sub_edges": [["SKU_Basic", "SKU_Premium"]],
    "trans_edges": []
  },
  
  "costs": {
    "inventory": {"SKU_Basic": 0.5, ...},
    "waste": {"SKU_Basic": 2.0, ...},
    "lost_sales": {"SKU_Basic": 10.0, ...},
    "purchasing": {"SKU_Basic": 5.0, ...}
  },
  
  "production_cap": {"SKU_Basic": [500, 500, ...], ...},
  "cold_capacity": {"DC1": 5000, ...},
  "cold_usage": {"SKU_Basic": 1.0, ...}
}
```

---

## 10. Evaluation Pipeline

### Step 1: Generate Prompts

```bash
python -m reloop.tools.generate_prompts
```

### Step 2: Run Zero-shot Baseline

```bash
# Set model
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4o"

# Run
python -m reloop.eval.run_benchmark \
  --mode zero-shot \
  --out results/gpt4_zero_shot/
```

### Step 3: Run ReLoop Agent

```bash
# Set model (recommend Qwen-Max)
export OPENAI_API_KEY="..."
export OPENAI_MODEL="qwen-max"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

# Run
python -m reloop.agents.cli.run_benchmark \
  --suite suite.txt \
  --out results/reloop/
```

---

## 11. Universal Retail Solver Settings

| Parameter | Value | Purpose |
|-----------|-------|---------|
| TimeLimit | 60s | Prevent stalling |
| MIPGap | 0.01 | 1% tolerance |
| OutputFlag | 0 | Suppress logs |

### Status Mapping

| Status | Meaning | Has Solution |
|--------|---------|--------------|
| OPTIMAL | Proven optimal | ✓ |
| OPTIMAL (TL) | Time limit with solution | ✓ |
| TIMEOUT | Time limit, no solution | ✗ |
| INFEASIBLE | No feasible solution | ✗ |

---

## 12. Research Directions

### Current Limitations

| Limitation | Description |
|------------|-------------|
| Manual probe design | 8 probes hand-crafted by experts |
| No coverage guarantee | May miss some error types |
| Domain-specific | Probes designed for retail OR |

### Future Work

**Direction 1: Automatic Probe Generation**
```
Given constraint C(x) ≤ b, auto-generate test data that
distinguishes correct from incorrect implementations
Method: SMT solving / symbolic execution / LLM-in-the-loop
```

**Direction 2: Semantic Coverage Analysis**
```
Definition (Coverage):
  Coverage(P) = |{c ∈ Constraints : ∃p ∈ P tests c}| / |Constraints|

Challenge: Constraints interact - testing A may implicitly test B
```

**Direction 3: Formal Verification**
```
Definition (Probe Soundness):
  Probe P is sound for constraint C ⟺
    ∀ implementation M' missing C: P(M') = FAIL

Open Problem:
  Construct minimal sound and complete probe set
```

---

## 13. Citation

```bibtex
@misc{reloop2026,
  author = {Junbo Jacob Lian and Yujun Sam Sun and Diego Klabjan},
  title  = {ReLoop: Closing the Silent Failure Gap in LLM-based 
            Optimization Modeling via Semantic Probes},
  year   = {2026},
}
```