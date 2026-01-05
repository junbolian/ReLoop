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

## Agent Pipeline

```
profile_data → step1 → step2 → step3 → sanity → step4 → audit → probe → run
               contract spec   templates        codegen        verify
                                                    ↑            │
                                                    └── repair ──┘
```

### Steps:
1. **Step 1**: Lock task contract (optimize, controls, constraints)
2. **Step 2**: Build spec sheet (sets, decisions, objective)
3. **Step 3**: Map to constraint templates (math formulas)
4. **Step 4**: Generate Gurobi code (no template given)
5. **Probe**: Run 8 semantic probes
6. **Repair**: If probes fail, diagnose and retry (up to 5x)

---

## Research Directions

### Current Limitations

1. **Manual probe design**: 8 probes are hand-crafted by domain experts
2. **No coverage guarantee**: Unknown if probes cover all constraint types
3. **No formal framework**: Probe correctness is empirical, not proven

### Future Work

| Direction | Research Question | Potential Venue |
|-----------|-------------------|-----------------|
| **Automatic Probe Generation** | Given constraint C, auto-generate test data that distinguishes correct/incorrect implementations | ICML/NeurIPS |
| **Semantic Coverage** | How to measure and ensure probes cover all critical constraints? | AAAI/IJCAI |
| **Formal Verification** | Can we prove probe soundness (no false negatives)? | POPL/CAV |
| **Cross-Domain Generalization** | Do probe design principles transfer to other OR domains? | CPAIOR |

### Formalization Sketch

```
Definition (Silent Failure):
  Model M has silent failure on data d ⟺ 
    M.solve(d).status = OPTIMAL ∧ M.solve(d).obj ≠ ground_truth(d)

Definition (Probe Soundness):
  Probe P is sound for constraint C ⟺ 
    ∀ implementation M missing C: P(M) = FAIL

Open Problem:
  Given optimization model spec S, construct minimal probe set {P₁...Pₖ} 
  that is sound and complete for all single-constraint errors.
```

---

## Citation

```bibtex
@misc{reloop2026,
  author = {Junbo Jacob Lian and Yujun Sam Sun and Diego Klabjan},
  title  = {ReLoop: Closing the Silent Failure Gap in LLM-based 
            Optimization Modeling via Semantic Probes},
  year   = {2026},
}
```

---

## License

Released for research and educational use. See `LICENSE` for details.