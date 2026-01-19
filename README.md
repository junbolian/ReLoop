# ReLoop: Semantic Probe Verification for LLM-based Optimization

ReLoop is the public codebase for a research project on **retail supply-chain optimization** and **LLM-based text-to-MILP modeling** with **semantic probe verification**.

This repository releases:
- **RetailOpt-190**: A vertically-focused retail text-to-optimization benchmark
- **Universal Retail Solver**: Reference MILP implementation with ground truth
- **Semantic Probes**: 8 boundary tests for detecting constraint errors
- **ReLoop Agent**: LangGraph-based orchestrator with probe-guided repair

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Silent Failure Detection** | Identifies code that runs but produces wrong answers |
| **Semantic Probes** | 8 boundary tests via code execution (not LLM prompting) |
| **No Code Template** | Tests actual math→code translation ability |
| **Probe-Guided Repair** | Diagnosis fed back for targeted fixes |

---

## Repository Structure

```
reloop/
├── solvers/
│   └── universal_retail_solver.py    # Reference MILP (ground truth)
│
├── agents/
│   ├── orchestrator_graph.py         # LangGraph state machine
│   ├── schemas.py                    # Pydantic data models
│   ├── prompt_stack.py               # Prompt management
│   ├── step_prompts/                 # 8 prompt files
│   ├── tools/
│   │   ├── semantic_probes.py        # 8 probes implementation
│   │   ├── script_runner.py          # Code execution
│   │   ├── sanity_checker.py         # Logic validation
│   │   └── static_auditor.py         # Pattern checking
│   └── cli/
│       ├── run_one.py                # Single scenario
│       └── run_benchmark.py          # Full benchmark
│
├── scenarios/
│   ├── spec/
│   │   ├── retail_spec.md            # Archetype specifications
│   │   └── retail_prompts.md         # Prompt templates
│   ├── data/                         # 190 JSON instances
│   └── prompts/                      # Per-instance prompts
│
└── eval/
    ├── run_benchmark.py              # Evaluation script
    └── evaluate_with_probes.py       # Probe-based evaluation
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/junbolian/ReLoop.git
cd ReLoop
pip install -r requirements.txt
```

Requirements:
- Python ≥ 3.10
- Gurobi with valid license
- `gurobipy`, `numpy`, `pandas`, `pydantic`, `langgraph`

### 2. Model Configuration

**Recommended: Qwen-Max** (best balance of capability and cost)

```bash
# Qwen-Max via DashScope API
export OPENAI_API_KEY="your-dashscope-key"
export OPENAI_MODEL="qwen-max"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

**Alternative Models:**

| Model | API | Notes |
|-------|-----|-------|
| `qwen-max` | DashScope | Recommended - best quality |
| `qwen2.5-coder-32b-instruct` | DashScope | Good for code tasks |
| `deepseek-chat` | DeepSeek | Cost-effective alternative |
| `gpt-4o` | OpenAI | Strongest but expensive |

### 3. Run Reference Solver

```bash
python -m reloop.solvers.universal_retail_solver \
  --json scenarios/data/retail_f1_52_weeks_v0.json
```

### 4. Run Agent on One Scenario

```bash
python -m reloop.agents.cli.run_one \
  --scenario retail_f1_52_weeks_v0 \
  --out artifacts
```

### 5. Run Full Benchmark

```bash
python -m reloop.agents.cli.run_benchmark \
  --suite suite.txt \
  --out artifacts
```

---

## RetailOpt-190 Benchmark

**190 instances** = 38 archetypes × 5 numerical variants

### 8 Mechanism Families:

| Family | Archetypes | Key Mechanisms |
|--------|------------|----------------|
| F1: Operations | 4 | Inventory, demand, lost sales, waste |
| F2: Assortment | 6 | Substitution, cannibalization, promotions |
| F3: Resources | 4 | Storage capacity, production capacity |
| F4: Dynamics | 6 | Shelf-life, demand surge, supply risk |
| F5: Feasibility | 4 | Stress tests, slack variables |
| F6: Logistics | 4 | MOQ, pack size, lead time, fixed cost |
| F7: Network | 6 | Transshipment, multi-echelon, hub-spoke |
| F8: Omni-channel | 4 | Returns, labor, sustainability |

---

## Semantic Probes

### How Probes Work

Semantic probes verify constraint correctness through **code execution**, not LLM prompting:

```
1. Construct boundary test data (e.g., zero production for one product)
2. Execute LLM-generated code with test data (subprocess)
3. Check observable outcomes (objective value, solver status)
4. Compare against expected behavior → PASS/FAIL
```

### 8 Probes:

| Probe | Tests | Detection Method |
|-------|-------|------------------|
| `substitution_basic` | S implementation | Objective range check |
| `demand_route_constraint` | S_out ≤ demand | UNBOUNDED detection |
| `no_substitution` | Empty edges | Spurious benefit detection |
| `production_capacity` | Prod cap | Objective lower bound |
| `storage_capacity` | Storage cap | INFEASIBLE detection |
| `aging_dynamics` | Shelf-life | Waste cost verification |
| `lost_sales_slack` | L variable | INFEASIBLE detection |
| `nonnegativity` | I ≥ 0 | Negative inventory check |

### Key Insight

Probes test **behavior**, not **code**. They work on any implementation without parsing it.

---

## ReLoop Pipeline

### Complete Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         ReLoop Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Input                                                     │  │
│  │ ├── Business Narrative (scenario description)             │  │
│  │ ├── Data Schema (JSON structure)                          │  │
│  │ └── Full Data (for execution)                             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Step 0: Data Profile (automatic)                          │  │
│  │ → Extract parameter roles for verification                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Step 1: Problem Understanding (LLM)                       │  │
│  │ Input: Narrative + Schema                                 │  │
│  │ Output: Objective, decisions, constraints (JSON)          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Step 2: Mathematical Specification (LLM)                  │  │
│  │ Input: Step1 output + Schema                              │  │
│  │ Output: Sets, variables, objective, constraint formulas   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Step 3: Code Generation (LLM)                             │  │
│  │ Input: Step2 output + Schema                              │  │
│  │ Output: GurobiPy code                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Step 4: Sensitivity Verification (automatic)              │  │
│  │ Input: Code + Full data + Parameter roles                 │  │
│  │ Output: Verification report (parameter anomalies)         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                    ┌─────────┴─────────┐                       │
│                    ▼                   ▼                       │
│               [All Pass]          [Anomaly]                    │
│                    │                   │                       │
│                    ▼                   ▼                       │
│               Output code    ┌─────────────────────┐           │
│                              │ Step 5: Repair (LLM)│           │
│                              │ Input: Code + Report │           │
│                              │ Output: Fixed code   │           │
│                              └──────────┬──────────┘           │
│                                         │                      │
│                                         └───▶ Back to Step 4   │
│                                              (max N retries)   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Steps

| Step | Type | Input | Output |
|------|------|-------|--------|
| 0 | Auto | JSON data | Parameter roles |
| 1 | LLM | Narrative + Schema | Problem understanding (JSON) |
| 2 | LLM | Step 1 + Schema | Math specification (JSON) |
| 3 | LLM | Step 2 + Schema | GurobiPy code |
| 4 | Auto | Code + Data | Verification report |
| 5 | LLM | Code + Report | Repaired code |

### Data Usage Principle

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

### Key Design Principles

1. **Minimal prompt**: Only give business narrative + data schema, let LLM decide modeling
2. **Grounded verification**: Test actual code behavior, not LLM-generated explanations
3. **Targeted repair**: Diagnosis guides specific fixes instead of full regeneration
4. **Data isolation**: LLM sees structure, verification uses values, repair sees diagnostics

---

## Citation

```bibtex
@misc{reloop2026,
  author = {Junbo Jacob Lian and Yujun Sam Sun and Huiling Chen and Chaoyu Zhang and Chung-Piaw Teo},
  title  = {ReLoop: Closing the Silent Failure Gap in LLM-based 
            Optimization Modeling via Semantic Probes},
  year   = {2026},
}
```

---

## License

Released for research and educational use. See `LICENSE` for details.