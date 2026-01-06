# ReLoop Agent Runner

LangGraph orchestrator for retail MILP codegen with **Semantic Probe Verification**.

Pipeline: `contract → spec → templates → sanity → codegen → audit → probe → run/repair`

---

## Quick Start

```bash
# 1. Set API credentials
$env:OPENAI_API_KEY = "sk-..."
$env:OPENAI_BASE_URL = "https://your-api-endpoint/v1"
$env:OPENAI_MODEL = "claude-opus-4-20250514"

# 2. Run single scenario
python -m reloop.agents.cli.run_one --scenario retail_f1_base_v0 --out artifacts

# 3. Check output
# Expected: Semantic Probes: 14/14 passed, Solver status: OPTIMAL
```

---

## LLM Configuration

### Current: API Testing

```bash
# Claude (via proxy)
$env:OPENAI_API_KEY = "sk-..."
$env:OPENAI_BASE_URL = "https://your-proxy/v1"
$env:OPENAI_MODEL = "claude-opus-4-20250514"

# Qwen-Max (via DashScope)
$env:OPENAI_API_KEY = "sk-..."
$env:OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:OPENAI_MODEL = "qwen-max"

# GPT-4o (direct)
$env:OPENAI_API_KEY = "sk-..."
$env:OPENAI_MODEL = "gpt-4o"
```

### Future: Self-Hosted Deployment

```bash
# vLLM / Ollama / TGI
$env:OPENAI_BASE_URL = "http://localhost:8000/v1"
$env:OPENAI_MODEL = "Qwen2.5-Coder-32B"
```

---

## How Semantic Probes Work

Probes verify constraints via **code execution**, not LLM prompting:

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Construct boundary test data                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ data = {                                                │   │
│  │   "demand": {"Basic": 100, "Premium": 0},              │   │
│  │   "production_cap": {"Basic": 0, "Premium": 80}        │   │
│  │ }                                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         ↓                                       │
│  Step 2: Execute LLM code with subprocess                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ result = subprocess.run(                                │   │
│  │   [python, llm_generated_code],                        │   │
│  │   input=data                                            │   │
│  │ )                                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         ↓                                       │
│  Step 3: Check observable outcomes                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ if status == UNBOUNDED → missing constraint            │   │
│  │ if objective < expected → wrong implementation         │   │
│  │ if objective in range → PASS                           │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 14 Semantic Probes

| # | Probe | Mechanism | Detection Method |
|---|-------|-----------|------------------|
| 1 | `substitution_basic` | Substitution direction | Objective range check |
| 2 | `demand_route_constraint` | S_out ≤ demand | UNBOUNDED detection |
| 3 | `no_substitution` | Empty sub_edges | Spurious S benefit |
| 4 | `production_capacity` | Prod cap enforcement | Objective lower bound |
| 5 | `storage_capacity` | Storage cap enforcement | INFEASIBLE detection |
| 6 | `aging_dynamics` | Shelf-life aging | Waste cost check |
| 7 | `lost_sales_slack` | L variable presence | INFEASIBLE detection |
| 8 | `nonnegativity` | I ≥ 0 | Negative inventory |
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

---

## Evaluation Metrics (4 Dimensions)

| Metric | Definition | Formula |
|--------|------------|---------|
| **Syntax Pass Rate** | Code compiles without error | `compile_ok / total` |
| **Execution Pass Rate** | Code runs and solver returns status | `exec_ok / total` |
| **Silent Failure Rate** | Runs OK but wrong answer | `(exec_ok - correct) / exec_ok` |
| **Overall Accuracy** | Objective within 1% of ground truth | `correct / total` |

### Expected Results by Configuration

| Configuration | Syntax | Execution | Silent Failure | Accuracy |
|---------------|--------|-----------|----------------|----------|
| Zero-shot (no probe) | ~90% | ~70% | ~50% | ~35% |
| ReLoop (with probe) | ~95% | ~85% | ~15% | ~70% |
| ReLoop + Repair | ~95% | ~90% | ~10% | ~80% |

---

## Ablation Study Design

### Three Ablation Dimensions

```
┌─────────────────────────────────────────────────────────────────┐
│                     ABLATION MATRIX                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Dimension 1: STEP (Pipeline Depth)                            │
│  ├── Zero-shot: Single LLM call, no pipeline                   │
│  ├── 3-step: contract → spec → codegen                         │
│  └── 5-step: Full pipeline (contract → spec → templates →      │
│              sanity → codegen)                                  │
│                                                                 │
│  Dimension 2: PROBE (Verification)                             │
│  ├── No Probe: Skip semantic verification                      │
│  ├── Probe (no repair): Verify but don't repair                │
│  └── Probe + Diagnosis: Feed diagnosis to repair               │
│                                                                 │
│  Dimension 3: REPAIR (Iteration)                               │
│  ├── No Repair: Single attempt                                 │
│  ├── Blind Repair: Retry without diagnosis                     │
│  └── Guided Repair: Use probe/error diagnosis                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Ablation Configurations

| Config ID | Steps | Probes | Repair | Description |
|-----------|-------|--------|--------|-------------|
| A1 | Zero-shot | No | No | Baseline: single LLM call |
| A2 | 5-step | No | No | Pipeline only |
| A3 | 5-step | Yes | No | + Probe verification |
| A4 | 5-step | No | Yes (blind) | + Blind repair |
| A5 | 5-step | Yes | Yes (guided) | Full ReLoop |

### Expected Ablation Results

| Config | Syntax | Execution | Silent Fail | Accuracy |
|--------|--------|-----------|-------------|----------|
| A1 (baseline) | 85% | 65% | 55% | 30% |
| A2 (+pipeline) | 90% | 75% | 45% | 40% |
| A3 (+probe) | 90% | 75% | 20% | 60% |
| A4 (+blind repair) | 92% | 80% | 40% | 50% |
| A5 (full) | 95% | 90% | 10% | 80% |

### Research Questions

| RQ | Question | Ablation |
|----|----------|----------|
| RQ1 | Does step-by-step pipeline improve accuracy? | A1 vs A2 |
| RQ2 | Do probes detect silent failures? | A2 vs A3 |
| RQ3 | Does probe diagnosis improve repair? | A4 vs A5 |
| RQ4 | What is the marginal value of each probe? | Per-probe ablation |

---

## Probe Deep Dive (Future Work)

### Probe Coverage Analysis

```
For each constraint type C:
  - Count: How many scenarios require C?
  - Detection: Does probe detect missing C?
  - Precision: False positive rate?
  - Recall: False negative rate?
```

### Probe Design Principles

| Principle | Description | Example |
|-----------|-------------|---------|
| **Isolation** | Test ONE mechanism per probe | `substitution_basic` only tests S |
| **Boundary** | Use extreme values | demand=100, supply=0 |
| **Observable** | Check status/objective only | No code parsing |
| **Deterministic** | Same code → same result | Fixed random seed |

### Automatic Probe Generation (Research Direction)

```
Input: Constraint specification (e.g., "S_out ≤ demand")
Output: Test data that distinguishes correct vs incorrect implementation

Method:
1. Symbolic analysis of constraint
2. Generate boundary case where constraint is binding
3. Predict expected behavior (OPTIMAL/INFEASIBLE/objective range)
4. Execute and compare
```

---

## CLI Reference

### Run Single Scenario

```bash
python -m reloop.agents.cli.run_one \
  --scenario retail_f1_base_v0 \
  --out artifacts \
  --repair-limit 5 \
  --max-turns 12
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--scenario` | required | Scenario ID or JSON path |
| `--out` | `artifacts` | Output directory |
| `--repair-limit` | 5 | Max repair iterations |
| `--max-turns` | 12 | Max total node executions |
| `--no-probes` | false | Disable semantic probes (for ablation) |
| `--mock-llm` | false | Use mock LLM (offline testing) |

### Run Benchmark Suite

```bash
python -m reloop.agents.cli.run_benchmark \
  --suite suite.txt \
  --out artifacts
```

---

## Output Structure

```
artifacts/<run_id>/
├── meta.json              # Run metadata
├── events.jsonl           # Event log
└── turns/<k>/
    ├── messages.json      # LLM prompts/responses
    ├── step_outputs.json  # Structured outputs (contract/spec/templates)
    ├── code.py            # Generated Gurobi script
    ├── static_audit.json  # Static analysis results
    ├── semantic_probe.json # Probe results (14 probes)
    ├── solve.json         # Solver status + objective
    └── stdout.txt         # Runtime logs
```

---

## Key Design Choices

1. **No code template** in Step 4 - LLM must translate math → code
2. **Probes via execution** - not LLM self-checking
3. **Diagnosis feedback** - probe results guide repair
4. **Substitution semantics enforced** - `[A,B]` = A's demand served by B
5. **Holding cost = I - y** - end-of-period inventory, not start-of-period

---

## Bug Fixes (2025-01-05)

### 1. Routing Logic Fix

**Problem:** Pipeline skipped probe and run when `turn_index >= max_turns`, producing no output.

**Fix:** Ensure at least one complete run (probe + run) before checking max_turns limit.

### 2. Holding Cost Fix

**Problem:** LLM generated `cost * I` instead of `cost * (I - y)`.

**Fix:** Updated prompts to explicitly state:
- I = START-OF-PERIOD inventory
- Holding cost = END-OF-PERIOD inventory = (I - y)

### 3. New Probes Added

- `initialization`: Detects missing t=1 initialization (objective ≈ 0)
- `holding_cost`: Detects wrong holding cost formula (objective 60% too low)

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| No output after run | Routing bug | Update orchestrator_graph.py |
| Objective 60% too low | Wrong holding cost | Check if using (I-y) |
| Objective ≈ 0 | Missing t=1 init | Add I[p,l,1,a]=0 for a<SL |
| UNBOUNDED | Missing demand_route | Add S_out ≤ demand |
| INFEASIBLE | Missing L variable | Add lost sales slack |