# RetailOpt-190: Prompt System Documentation

> File: `reloop/scenarios/spec/retail_prompts.md`

---

## 1. Prompt Architecture

RetailOpt-190 uses **two prompt formats** for different evaluation modes:

### Two Prompt Files Per Scenario

| File | Content | Used By |
|------|---------|---------|
| `{scenario_id}.base.txt` | Scenario description only | ReLoop Agent (guardrails injected by step_prompts) |
| `{scenario_id}.scenario.txt` | Scenario + guardrails + instructions | Zero-shot baseline (single LLM call) |

### Generation

```bash
python -m reloop.tools.generate_prompts
# Output: scenarios/prompts/{scenario_id}.base.txt
#         scenarios/prompts/{scenario_id}.scenario.txt
```

---

## 2. Evaluation Modes

### Mode 1: Zero-shot Baseline

**Input:** `{scenario_id}.scenario.txt` (complete prompt)

```
┌─────────────────────────────────────────┐
│ [SCENARIO]                              │
│ Family, archetype, business narrative   │
│                                         │
│ [MODELING GUIDELINES]                   │
│ Guardrails: data format, substitution,  │
│ shelf-life, boundary conditions         │
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

**For:** GPT-4, Claude, DeepSeek, Qwen (all baselines)

### Mode 2: ReLoop Agent (Multi-step)

**Input:** `{scenario_id}.base.txt` (scenario only)

```
┌─────────────────────────────────────────┐
│ [SCENARIO]                              │
│ Family, archetype, business narrative   │
│ (NO guardrails - injected separately)   │
└─────────────────────────────────────────┘
         ↓
   step1: contract extraction
         ↓
   step2: spec sheet
         ↓
   step3: constraint templates
         ↓
   step4: codegen (guardrails injected here)
         ↓
  [Probes] → repair if failed
         ↓
      Results
```

**For:** Qwen-Max primary experiments

---

## 3. Guardrails Content

The guardrails are **critical rules** that prevent common errors:

### 3.1 Substitution Semantics (Most Important)

```
Edge [p_from, p_to] = "upward substitution"
Meaning: p_to can serve p_from's demand

Example: ["SKU_Basic", "SKU_Premium"]
→ Premium serves Basic's demand (NOT Basic serves Premium)

Build edge mappings BEFORE constraints:
  for p_from, p_to in sub_edges:
      outgoing_edges[p_from].append(p_to)  # p_from sends demand OUT
      incoming_edges[p_to].append(p_from)  # p_to receives requests IN
```

### 3.2 Boundary Conditions

```
[INITIALIZATION at t=1]
  I[p,l,1,a] = 0  for a < shelf_life[p]
  I[p,l,1,shelf_life] = Q[p,l,1] if lead_time=0, else 0

[AGING BOUNDARY at t=T]
  Do NOT add aging constraints for t=T (would reference T+1)

[FRESH INFLOW]
  if t > lead_time[p]:
      I[p,l,t,shelf_life] = Q[p,l,t-lead_time]
  else:
      I[p,l,t,shelf_life] = 0
  NEVER access Q[p,l,0] or negative indices
```

### 3.3 Variable Naming Contract

```
I[p,l,t,a]: START-OF-PERIOD inventory by remaining life bucket
y[p,l,t,a]: sales from life bucket (during the period)
W[p,l,t]:   waste (expired from bucket a=1)
Q[p,l,t]:   orders/production
L[p,l,t]:   lost sales (MUST include this slack variable)
S[p_from,p_to,l,t]: substitution flow

NOTE: I is START-of-period inventory (before sales).
      End-of-period inventory = I - y (after sales).
```

### 3.4 Holding Cost (CRITICAL)

```
Holding cost applies to END-OF-PERIOD inventory, not start-of-period:

CORRECT:
  for a in range(2, shelf_life[p] + 1):
      obj += cost * (I[p,l,t,a] - y[p,l,t,a])  # END-of-period

WRONG:
  obj += cost * I[p,l,t,a]  # This is START-of-period!

Only apply to buckets a >= 2 (bucket a=1 expires, not held overnight)
```

---

## 4. Semantic Probe Verification

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

### 14 Probes

| # | Probe | Mechanism | Detection Method |
|---|-------|-----------|------------------|
| 1 | `substitution_basic` | S variables | Objective range check |
| 2 | `demand_route_constraint` | S_out ≤ demand | UNBOUNDED detection |
| 3 | `no_substitution` | Empty edges | Spurious benefit detection |
| 4 | `production_capacity` | Prod cap | Objective lower bound |
| 5 | `storage_capacity` | Storage cap | INFEASIBLE detection |
| 6 | `aging_dynamics` | Shelf-life | Waste cost verification |
| 7 | `lost_sales_slack` | L variable | INFEASIBLE detection |
| 8 | `nonnegativity` | I ≥ 0 | Negative inventory check |
| 9 | `lead_time` | Lead time handling | Delivery timing |
| 10 | `transshipment` | Network flows | Trans constraint |
| 11 | `labor_capacity` | Labor constraints | Capacity check |
| 12 | `moq` | Minimum order quantity | MOQ enforcement |
| 13 | `initialization` | t=1 init (I=0 for a<SL) | Objective = 0 detection |
| 14 | `holding_cost` | End-of-period (I-y) | Objective too low detection |

### Critical Probes (Silent Failure Detection)

| Probe | Common Error | Symptom |
|-------|--------------|---------|
| `initialization` | Missing `I[p,l,1,a]=0` for a < shelf_life | Objective ≈ 0 (free inventory) |
| `holding_cost` | Using `I` instead of `I-y` | Objective 60% too low |

### Key Insight

Probes test **behavior**, not **code**. They work on any implementation without parsing.

---

## 5. Evaluation Metrics (4 Dimensions)

| Metric | Definition | Formula |
|--------|------------|---------|
| **Syntax Pass Rate** | Code compiles without error | `compile_ok / total` |
| **Execution Pass Rate** | Code runs and solver returns status | `exec_ok / total` |
| **Silent Failure Rate** | Runs OK but wrong answer | `(exec_ok - correct) / exec_ok` |
| **Overall Accuracy** | Objective within 1% of ground truth | `correct / total` |

---

## 6. Ablation Study Design

### Three Ablation Dimensions

| Dimension | Options |
|-----------|---------|
| **STEP** (Pipeline Depth) | Zero-shot, 3-step, 5-step |
| **PROBE** (Verification) | No probe, Probe only, Probe + Diagnosis |
| **REPAIR** (Iteration) | No repair, Blind repair, Guided repair |

### Ablation Configurations

| Config | Steps | Probes | Repair | Description |
|--------|-------|--------|--------|-------------|
| A1 | Zero-shot | No | No | Baseline: single LLM call |
| A2 | 5-step | No | No | Pipeline only |
| A3 | 5-step | Yes | No | + Probe verification |
| A4 | 5-step | No | Yes (blind) | + Blind repair |
| A5 | 5-step | Yes | Yes (guided) | Full ReLoop |

### Research Questions

| RQ | Question | Ablation |
|----|----------|----------|
| RQ1 | Does step-by-step pipeline improve accuracy? | A1 vs A2 |
| RQ2 | Do probes detect silent failures? | A2 vs A3 |
| RQ3 | Does probe diagnosis improve repair? | A4 vs A5 |
| RQ4 | What is the marginal value of each probe? | Per-probe ablation |

---

## 7. Model Recommendations

### Current: API Testing

```bash
# Claude (via proxy)
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://your-proxy/v1"
export OPENAI_MODEL="claude-opus-4-20250514"

# Qwen-Max (via DashScope)
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPENAI_MODEL="qwen-max"

# GPT-4o (direct)
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o"
```

### Future: Self-Hosted Deployment

```bash
# vLLM / Ollama / TGI
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_MODEL="Qwen2.5-Coder-32B"
```

### Model Comparison

| Model | API | Notes |
|-------|-----|-------|
| `qwen-max` | DashScope | Recommended |
| `qwen2.5-coder-32b-instruct` | DashScope | Code-specialized |
| `deepseek-chat` | DeepSeek | Cost-effective |
| `gpt-4o` | OpenAI | Strongest but expensive |
| `claude-opus-4-20250514` | Anthropic | Strong reasoning |

---

## 8. Silent Failure Problem

### Definition

**Silent Failure:** Code that executes successfully and returns OPTIMAL, but produces wrong answers due to constraint semantic errors.

### Common Causes

| Error Type | Frequency | Example |
|------------|-----------|---------|
| Substitution direction | ~35% | Edge [A,B] misread as "A replaces B" |
| Missing constraint | ~20% | No demand_route → UNBOUNDED |
| Wrong holding cost | ~20% | Using I instead of I-y |
| Missing initialization | ~15% | No I[p,l,1,a]=0 → obj=0 |
| Boundary errors | ~10% | t=1/t=T edge cases |

---

## 9. Research Directions

### Current Limitations

| Limitation | Description |
|------------|-------------|
| Manual probe design | 14 probes hand-crafted |
| No coverage guarantee | May miss some errors |
| Domain-specific | Retail OR only |

### Future Work

**Direction 1: Automatic Probe Generation**
```
Given constraint C(x) ≤ b, auto-generate test data that
distinguishes correct from incorrect implementations
```

**Direction 2: Formal Verification**
```
Definition (Probe Soundness):
  Probe P is sound for constraint C ⟺
    ∀ implementation M' missing C: P(M') = FAIL

Open Problem:
  Construct minimal sound and complete probe set
```

---

## 10. Citation

```bibtex
@misc{reloop2026,
  author = {Junbo Jacob Lian and Yujun Sam Sun and Diego Klabjan},
  title  = {ReLoop: Closing the Silent Failure Gap in LLM-based 
            Optimization Modeling via Semantic Probes},
  year   = {2026},
}
```