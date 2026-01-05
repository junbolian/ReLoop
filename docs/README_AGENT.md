# ReLoop Agent Runner

LangGraph orchestrator for retail MILP codegen with **Semantic Probe Verification**.

Pipeline: `contract → spec → templates → sanity → codegen → audit → probe → run/repair`

---

## How Semantic Probes Work

Probes verify constraints via **code execution**, not LLM prompting:

```
1. Construct boundary test data
   → e.g., zero production for one product, 100% demand for another

2. Execute LLM-generated code (subprocess)
   → Run actual Gurobi optimization

3. Check observable outcomes
   → Solver status (OPTIMAL/INFEASIBLE/UNBOUNDED)
   → Objective value range

4. Diagnose failures
   → UNBOUNDED = missing constraint
   → Objective too low = wrong implementation
```

---

## Recommended Model

**Qwen-Max** (primary choice)
- Strong reasoning + code generation
- Good balance of capability and cost
- Via DashScope API (OpenAI-compatible)

**Alternatives:** Qwen2.5-Coder-32B, GPT-4o, DeepSeek-V3

---

## Setup

1. Python ≥ 3.10 and Gurobi with valid license
2. `pip install -e .`
3. Set LLM credentials:

```bash
# Qwen-Max (recommended)
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="qwen-max"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

# Or GPT-4
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o"
```

---

## Run One Scenario

```bash
python -m reloop.agents.cli.run_one \
  --scenario retail_f1_52_weeks_v0 \
  --out artifacts \
  --repair-limit 5 \
  --max-turns 12
```

### Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--scenario` | required | Scenario ID or JSON path |
| `--out` | `artifacts` | Output directory |
| `--repair-limit` | 5 | Max repair iterations |
| `--max-turns` | 12 | Max total node executions |
| `--no-probes` | false | Disable semantic probes |
| `--mock-llm` | false | Use mock LLM (offline testing) |

---

## Run Benchmark Suite

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
├── events.jsonl           # Event log (optional)
└── turns/<k>/
    ├── messages.json      # LLM prompts/responses
    ├── step_outputs.json  # Structured outputs
    ├── code.py            # Generated script
    ├── semantic_probe.json # Probe results
    ├── solve.json         # Solver output
    └── stdout.txt         # Runtime logs
```

---

## 8 Semantic Probes

| Probe | Tests | Detection Method |
|-------|-------|------------------|
| `substitution_basic` | S implementation | Objective range check |
| `demand_route_constraint` | S_out ≤ demand | UNBOUNDED detection |
| `no_substitution` | Empty sub_edges | Spurious S benefit |
| `production_capacity` | Prod cap | Objective lower bound |
| `storage_capacity` | Storage cap | INFEASIBLE detection |
| `aging_dynamics` | Shelf-life aging | Waste cost check |
| `lost_sales_slack` | L variable | INFEASIBLE detection |
| `nonnegativity` | I ≥ 0 | Negative inventory |

Probe failures trigger targeted repair with diagnosis.

---

## Key Design Choices

1. **No code template** in Step 4 - LLM must translate math → code
2. **Probes via execution** - not LLM self-checking
3. **Diagnosis feedback** - probe results guide repair
4. **Substitution semantics enforced** - `[A,B]` = A's demand served by B