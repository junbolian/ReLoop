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
I[p,l,t,a]: inventory by remaining life bucket
y[p,l,t,a]: sales from life bucket
W[p,l,t]:   waste (expired)
Q[p,l,t]:   orders/production
L[p,l,t]:   lost sales
S[p_from,p_to,l,t]: substitution flow
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

## 5. Model Recommendations

### Primary: Qwen-Max

```bash
export OPENAI_API_KEY="your-dashscope-key"
export OPENAI_MODEL="qwen-max"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

**Why:** Best balance of reasoning + code generation capability

### Alternatives

| Model | API | Notes |
|-------|-----|-------|
| `qwen-max` | DashScope | Recommended |
| `qwen2.5-coder-32b-instruct` | DashScope | Code-specialized |
| `deepseek-chat` | DeepSeek | Cost-effective |
| `gpt-4o` | OpenAI | Strongest but expensive |

---

## 6. Silent Failure Problem

### Definition

**Silent Failure:** Code that executes successfully and returns OPTIMAL, but produces wrong answers due to constraint semantic errors.

### Common Causes

| Error Type | Frequency | Example |
|------------|-----------|---------|
| Substitution direction | ~45% | Edge [A,B] misread as "A replaces B" |
| Missing constraint | ~25% | No demand_route → UNBOUNDED |
| Wrong indexing | ~20% | Holding cost on all buckets |
| Boundary errors | ~10% | t=1 initialization wrong |

---

## 7. Ablation Studies

### No Guardrails

```bash
python -m reloop.tools.generate_prompts --no_guardrails
```

Expected: Silent failure rate increases significantly

### No Probes

```bash
python -m reloop.agents.cli.run_one --scenario xxx --no-probes
```

Expected: Cannot detect silent failures

### No Repair

```bash
python -m reloop.agents.cli.run_one --scenario xxx --repair-limit 0
```

Expected: Lower correct rate, same silent failure rate

---

## 8. Research Directions

### Current Limitations

| Limitation | Description |
|------------|-------------|
| Manual probe design | 8 probes hand-crafted |
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

## 9. Citation

```bibtex
@misc{reloop2026,
  author = {Junbo Jacob Lian and Yujun Sam Sun and Diego Klabjan},
  title  = {ReLoop: Closing the Silent Failure Gap in LLM-based 
            Optimization Modeling via Semantic Probes},
  year   = {2026},
}
```