# ReLoop: Detecting Silent Failures in LLM-Generated Optimization Code via Behavioral Verification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-preprint-b31b1b.svg)]()

This repository contains the official implementation of the paper:

> **ReLoop: Detecting Silent Failures in LLM-Generated Optimization Code via Behavioral Verification**
>
> Junbo Jacob Lian, Yujun Sun, Huiling Chen, Chaoyu Zhang, Chung-Piaw Teo
>
> arXiv preprint, 2026

---

## Related Resources

| Resource | Link | Description |
|----------|------|-------------|
| **RetailOpt-190 Dataset** | [Hugging Face](https://huggingface.co/datasets/Jacoblian/RetailOpt-190) | Download dataset directly |
| **RetailOpt-190 Repository** | [GitHub](https://github.com/junbolian/RetailOpt-190) | Dataset details, prompt generation, universal solver |

The **RetailOpt-190** benchmark is a key contribution of this paper, featuring:
- 190 retail inventory optimization scenarios with varying complexity
- Natural language problem descriptions with ground-truth optimal solutions
- Prompt generation scripts and universal Gurobi solver implementation

---

## Overview

ReLoop is a behavioral verification framework that detects **silent failures** in LLM-generated optimization code—code that executes successfully but produces incorrect results. Our key insight is that correct optimization models satisfy fundamental behavioral invariants (execution correctness, constraint presence, objective completeness) that can be tested without ground truth via solver-based perturbation testing.

**Key Results:**
- Consistent improvements across all five models and three benchmarks
- Detection of missing cost/revenue terms, missing constraints, and data access errors
- No ground truth required for verification

---

## Framework Architecture

![ReLoop Framework Architecture](fig/Reloop_framework.png)

### Key Components

| Component | Description |
|-----------|-------------|
| **Chain-of-Thought Generation** | 4-stage structured generation: Understand → Formalize (with variable type reasoning) → Synthesize → Verify Completeness |
| **L1 Execution & Solver** | Syntax, runtime, and solver status checks with IIS/unbounded diagnostics (FATAL → regeneration) + duality check (INFO) |
| **L2 Behavioral Testing (CPT + OPT)** | Solver-based perturbation testing: CPT detects missing constraints, OPT detects missing objective terms |
| **Unified Diagnostic Schema** | All layers output `Diagnostic` objects; unified `build_repair_prompt()` assembles repair prompts |
| **Diagnostic-Based Repair** | Conservative strategy: only WARNING triggers repair, INFO is reference only |
| **Repair Safety Guardrails** | Prevents repair LLM from modifying input data or introducing dangerous operations |
| **Repair Regression Guard** | Post-repair rollback on crash, status degradation, or objective value drift (>4%) vs pre-repair baseline |
| **Repair Skip Guard** | Skip repair loop entirely when code is already VERIFIED with no actionable diagnostics |

### Design Principles

1. **Universal Verification**: All checks work without ground truth or domain-specific rules
2. **Behavioral Testing**: Solver-based perturbation detects missing model components without declarative analysis
3. **Conservative Repair**: Only high-confidence issues (WARNING) trigger repairs
4. **Robustness Guarantee**: L1 FATAL triggers regeneration; L2 behavioral tests are diagnostic only
5. **Repair Safety**: Repair code is validated before execution; data variable cannot be reassigned
6. **Regression Guard**: Repair never makes results worse — rollback on crash, status degradation, or value drift >4%
7. **Repair Skip**: Already-verified code is not subjected to unnecessary repair attempts

### Generation: Chain-of-Thought (CoT)

A single LLM call with 4-stage structured reasoning (`generation.py`, `prompts.py`):

1. **Understand** — Identify objective (min/max), decisions, constraints, parameters
2. **Formalize** — Write the mathematical model: sets, parameters, decision variables (with explicit CONTINUOUS/INTEGER/BINARY type reasoning), constraints, objective
3. **Synthesize** — Generate executable Gurobi Python code, implementing ALL constraints from the original problem description (not just those in Step 2)
4. **Verify Completeness** — Cross-check code against original problem: every cost/revenue term in objective, every constraint implemented, all data values correct

The code receives a pre-defined `data` dict and must not redefine it. Big-M values must be computed dynamically from data, never hardcoded.

#### Extraction + Fallback Strategy

CoT generation uses a two-phase approach to maximize both data accuracy and execution reliability (`generation.py`):

1. **Try Extraction** — LLM extracts ALL numerical parameters from the problem description into a structured JSON dict (`DATA_EXTRACTION_PROMPT`). If successful, a second LLM call generates code that references the pre-loaded `data["key"]` dict (`CHAIN_OF_THOUGHT_WITH_DATA_PROMPT`).
2. **Fallback** — If extraction fails (JSONDecodeError, empty result, or code still uses `json.loads`), the system falls back to standard self-contained generation where the LLM embeds data directly in code.

| Phase | When | Code Pattern | L2 Perturbation |
|-------|------|-------------|-----------------|
| Extraction succeeds | LLM produces valid JSON | `data["key"]` (external dict) | Data-dict perturbation works directly |
| Extraction fails | JSONDecodeError or invalid output | `data = json.loads("""...""")` (self-contained) | Auto-stripped before perturbation (see below) |

**Why extraction matters:** RetailOpt problems contain 20+ parameters (costs, prices, capacities, demands). When the LLM must copy these values into code, transcription errors are common. Extraction separates data handling from code logic, passing accurate numerical values via the pre-loaded dict.

### L1: Execution Verification (Blocking Layer)

Checks whether generated code executes and produces a valid solution (`verification.py:_layer1`).

**Sequential checks:**

| Step | Check | On Failure |
|------|-------|------------|
| L1.1 | Syntax (AST parse) | FATAL |
| L1.2 | Runtime execution | FATAL (with stderr) |
| L1.3 | Solver status | See below |
| L1.4 | Duality gap (add-on) | INFO only |

**L1.3 Solver status handling:**

| Status | Severity | Action | Diagnostics |
|--------|----------|--------|-------------|
| INFEASIBLE | FATAL | Regenerate | IIS constraint names + conflicting bounds |
| UNBOUNDED | FATAL | Regenerate | Unbounded variable rays (`InfUnbdInfo=1`) |
| TIMEOUT (no solution) | FATAL | Regenerate | — |
| OPTIMAL / feasible | PASS | Continue | — |

**L1.4 Duality check** — After OPTIMAL, compute relative primal-dual gap:
- Gap > 1%: INFO (likely numerical artifact, no repair)
- Gap <= 1%: PASS

**FATAL handling:** Triggers regeneration (up to `max_regeneration_attempts=3`). The LLM receives the failed code and error message to generate corrected code.

### L2: Behavioral Testing (CPT + OPT)

L2 uses **solver-based perturbation testing** to detect missing model components without declarative analysis (`verification.py:_layer2_cpt`, `verification.py:_layer2_opt`). Enabled by `--enable-cpt`.

#### CPT: Constraint Presence Testing

Tests whether expected constraints are actually active in the generated model.

**Procedure:**

1. **Extract candidates** — LLM reads problem description, outputs candidate constraints with type (capacity / demand / other) and related parameters
2. **Perturb and re-solve** — For each candidate, apply extreme perturbation to related parameter:

   | Constraint Type | Perturbation | Rationale |
   |----------------|-------------|-----------|
   | Capacity | x 0.001 (near-zero) | If capacity constraint exists, near-zero capacity should drastically change objective |
   | Demand | x 100 (scale up) | If demand constraint exists, extreme demand should cause infeasibility or large change |
   | Other | x 0.01 (1%) | General extreme perturbation |

3. **Grade by change ratio** — `change_ratio = |new_obj - baseline| / |baseline|`

   | Change Ratio | Severity | Meaning |
   |-------------|----------|---------|
   | < 5% | WARNING | Constraint likely **missing** (triggers repair) |
   | 5% - 30% | INFO | Uncertain (no repair) |
   | > 30% | PASS | Constraint is **present** |
   | INFEASIBLE | PASS | Constraint is active (extreme perturbation caused infeasibility) |

#### OPT: Objective Presence Testing

Tests whether expected objective function terms (costs, revenues) are present in the generated model.

**Procedure:**

1. **Extract candidates** — LLM reads problem description, outputs candidate objective terms with role (cost / revenue / other) and related parameters
2. **Perturb and re-solve** — For each candidate, apply role-specific perturbation to related parameter:

   | Objective Role | Perturbation | Rationale |
   |---------------|-------------|-----------|
   | Cost | x 0.001 (near-zero cost) | If cost term exists, near-zero cost should significantly change objective |
   | Revenue | x 100 (extreme revenue) | If revenue term exists, extreme revenue should significantly change objective |
   | Other | x 0.01 | General extreme perturbation |

3. **Grade by change ratio** — Same threshold-based detection as CPT:

   | Change Ratio | Severity | Meaning |
   |-------------|----------|---------|
   | < 5% | WARNING | Objective term likely **missing** (triggers repair) |
   | 5% - 30% | INFO | Uncertain (no repair) |
   | > 30% | PASS | Objective term is **present** |

**Example Output:**
```
[L2] CPT
  Extracted 3 candidates:
  ├─ [MISSING] minimum protein requirement - 0.0% change
  ├─ [UNCERTAIN] minimum carbs requirement - 21.1% change
  └─ [PRESENT] minimum calories requirement - 38.0% change

[L2] OPT
  Extracted 2 candidates:
  ├─ [MISSING] unit procurement cost - 1.2% change
  └─ [PRESENT] holding cost per unit - 45.3% change
```

### Diagnostic-Based Repair

All layers output unified `Diagnostic` objects with severity and evidence (`verification.py:Diagnostic`, `pipeline.py`).

**Severity-to-action mapping:**

| Severity | Source | Action |
|----------|--------|--------|
| FATAL | L1 execution errors | Regenerate code (not repair) |
| WARNING | L2 CPT missing constraints, L2 OPT missing objective terms | **Should fix** — included in repair prompt |
| INFO | L1 duality, L2 uncertain results | **Reference only** — shown but not fixed |

**Repair loop** (`pipeline.py`, budget `N=3`):
1. Collect all diagnostics → `build_repair_prompt()` assembles prompt with actionable issues (WARNING) and reference context (INFO)
2. LLM generates repaired code
3. **Safety guardrail** (`repair_safety.py`) — Validates repair code before execution:
   - Blocks: `data = {...}` (reassignment), `data["key"] = val` (mutation), dangerous imports (`os`, `subprocess`)
   - Exception: `data = json.loads(...)` is allowed (re-parses existing data)
   - On violation: 1 guided retry (does not consume repair budget); 2nd failure → keep original code
4. Re-verify repaired code
5. **Repair skip guard** — If the code is already VERIFIED with no WARNING diagnostics, skip the entire repair loop (prevents unnecessary modifications to correct code)
6. **Regression guard** — Compare post-repair result against pre-repair baseline:
   - Crash regression: objective was not None → now None
   - Status regression: status rank decreased (VERIFIED=3 > WARNINGS=2 > ERRORS=1 > FAILED=0)
   - Value regression: same status but objective shifted >4% relative (prevents "value drift" where repair changes a correct answer)
   - On regression: **rollback** to pre-repair code and stop all further repair
7. Repeat until: no actionable diagnostics, no change, regression detected, or budget exhausted

---

## Installation

```bash
git clone https://github.com/junbolian/ReLoop.git
cd ReLoop
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- Gurobi >= 11.0 (with valid license)
- Core: `numpy`, `pandas`, `openai`, `pydantic`, `tenacity`, `tiktoken`
- Agent stack: `langchain>=1.0`, `langgraph>=1.0`, `langchain-openai`
- See `requirements.txt` for full dependency list

```bash
# Verify installation
python -c "from reloop import ReLoopVerifier; print('OK')"
```

---

## Quick Start

ReLoop uses the OpenAI Python SDK, so any **OpenAI-compatible API** works out of the box (OpenAI, vLLM, Ollama, llama.cpp server, LiteLLM, etc.).

Set environment variables and run:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="http://localhost:8000/v1"   # your API endpoint

python run_ablation.py \
    -d data/RetailOpt-190.jsonl \
    -m gpt-4.1 \
    --enable-cpt \
    --workers 5 \
    -v
```

Results are saved to `experiment_results/<dataset-name>/<model>/`:
- `ablation_report.csv` — per-problem objectives and pass/fail at each verification stage
- `chat_logs.jsonl` — full LLM conversation logs

### CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `-d, --dataset` | *(required)* | Path to dataset JSONL file |
| `-m, --model` | `gpt-4.1` | Model name (passed to OpenAI SDK) |
| `--base-url` | `$OPENAI_BASE_URL` | Override API base URL (alternative to env var) |
| `--base-urls` | off | Comma-separated API endpoints; requests are round-robin distributed across endpoints |
| `--local` | off | Local mode; default endpoint becomes `http://127.0.0.1:8000/v1` and local dummy key is supported |
| `--api-key` | `$OPENAI_API_KEY` | Override API key for this run (`EMPTY` is fine for most local servers) |
| `--request-timeout` | 300 | Per-request timeout (seconds) |
| `-o, --output-dir` | `experiment_results/<dataset>/<model>` | Custom output directory |
| `--enable-cpt` | off | Enable L2 Behavioral Testing (CPT + OPT) |
| `--workers` | 20 | Number of concurrent workers |
| `--no-cot` | off | Skip CoT, use direct generation (for ablation baseline only; not recommended) |
| `--no-verify` | off | Skip verification layers (CoT generation only, for ablation baseline) |
| `-v, --verbose` | off | Print verbose logs to stdout |

### Local Multi-GPU Deployment (GPU 4/5/6/7)

Install optional local inference dependencies:

```bash
pip install -r requirements.local-inference.txt
```

Start local server for HF format models (`models/Qwen3-32B`, `models/SIRL-Gurobi32B`) via vLLM:

```bash
python scripts/deploy_local_llm.py \
  --model qwen3-32b \
  --backend vllm \
  --gpus 4,5,6,7 \
  --port 8000 \
  --trust-remote-code
```

Start local server for GGUF model (`models/OptMATH-Qwen2.5-32B-Instruct-GGUF`) via llama.cpp server:

```bash
python scripts/deploy_local_llm.py \
  --model optmath-gguf \
  --backend llama_cpp \
  --gpus 4,5,6,7 \
  --port 8001
```

Run ablation against local endpoint:

```bash
export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"

python run_ablation.py \
  -d data/RetailOpt-190.jsonl \
  -m Qwen3-32B \
  --local \
  --workers 5 \
  --enable-cpt \
  -v
```

If you deploy multiple local endpoints, use `--base-urls` for endpoint-level parallel load balancing:

```bash
python run_ablation.py \
  -d data/RetailOpt-190.jsonl \
  -m Qwen3-32B \
  --local \
  --base-urls "http://127.0.0.1:8000/v1,http://127.0.0.1:8002/v1" \
  --workers 8 \
  --enable-cpt \
  -v
```

### Python API

For programmatic usage:

```python
from reloop import run_reloop, DataExtractor

llm_client = YourLLMClient()  # must implement generate(prompt, system=None) -> str

extractor = DataExtractor(llm_client)
data = extractor.extract("Minimize cost with capacity 500 and demands [100, 150, 200]...")

result = run_reloop(
    problem_description="...",
    data=data,
    llm_client=llm_client,
    verbose=True
)

print(f"Status: {result.final_report.status}")  # VERIFIED, WARNINGS, ERRORS, FAILED
print(f"Objective: {result.final_report.objective}")
```

---

## Available Datasets

| Dataset | # Problems | Avg Tokens | Tolerance | Format |
|---------|:---:|:---:|:---:|--------|
| `RetailOpt-190.jsonl` | 190 | ~2,900 | 10⁻⁴ / 10⁻² | Data-embedded |
| `MAMO_ComplexLP_fixed.jsonl` | 203 | ~459 | 10⁻⁶ | Data-embedded |
| `IndustryOR_fixedV2.jsonl` | 100 | ~267 | 10⁻⁶ | Data-embedded |
All datasets use data-embedded format (full data in prompt) for evaluation. RetailOpt additionally provides schema-based prompts for ReLoop's verification pipeline, where the LLM sees only data schema and actual values are injected at runtime.

---

## Compared Models

| Type | Model | Provider | Temp. | Max Tokens | Notes |
|------|-------|----------|:---:|:---:|-------|
| Foundation | Claude Opus 4.6 | Anthropic API | 0.0 | 8192 | |
| Foundation | DeepSeek-V3.2 | DeepSeek API | 0.0 | 8192 | |
| Foundation | Qwen3-32B | Local (vLLM) | 0.0 | 8192 | BF16 |
| Offline SFT | OptMATH-Qwen2.5-32B | Local (llama.cpp) | 0.0 | 8192 | GGUF |
| Online RL | SIRL-Qwen2.5-32B | Local (vLLM) | 0.0 | 8192 | BF16 |

---

## Experiment Results

### Table 1: Main Results on RetailOpt-190

> Foundation models: Base = direct generation, CoT = structured chain-of-thought, ReLoop = CoT + L1–L2 verification with repair. SFT/RL models: Base = native generation format, CoT = our structured CoT replacing native prompting, ReLoop = CoT + L1–L2 (same pipeline as foundation models).

| Type | Model | Exec% ||| Acc% (ε=10⁻⁴) ||| Acc% (ε=10⁻²) |||
|------|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | | Base | CoT | ReLoop | Base | CoT | ReLoop | Base | CoT | ReLoop |
| Foundation | Claude Opus 4.6 | 72.1 | 93.7 | **100.0** | 22.6 | 31.1 | **31.1** | 26.8 | 34.7 | **35.3** |
| Foundation | DeepSeek-V3.2 | 91.1 | 53.2 | **97.4** | 0.5 | 3.7 | **5.8** | 3.7 | 5.8 | **11.1** |
| Foundation | Qwen3-32B | 0.0 | 0.0 | **2.1** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Offline SFT | OptMATH-Qwen2.5-32B | 0.5 | 0.0 | **4.7** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Online RL | SIRL-Qwen2.5-32B | 1.6 | 1.6 | 1.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |



### Table 2: Cross-Benchmark Generalization (Acc% pass@1, ε=10⁻⁶)

| | | MAMO-ComplexLP ||| IndustryOR |||
|------|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| Type | Model | Base | CoT | +ReLoop | Base | CoT | +ReLoop |
| Foundation | Claude Opus 4.6 | 70.4 | 73.9 | **79.8** | 66.0 | 66.0 | **68.0** |
| Foundation | DeepSeek-V3.2 | 60.1 | 59.6 | **62.6** | 50.0 | 54.0 | **62.0** |
| Foundation | Qwen3-32B | 29.1 | 30.0 | **36.0** | 38.0 | 44.0 | **47.0** |
| Offline SFT | OptMATH-Qwen2.5-32B | 44.8 | 45.8 | **46.3** | 31.0 | 33.0 | **33.0** |
| Online RL | SIRL-Qwen2.5-32B | 56.7 | 54.2 | **57.6** | 47.0 | 47.0 | **48.0** |

> For reference from SIRL (Chen et al., 2025) and StepORLM (Zhou et al., 2025): GPT-4 (49.3%/33.0%), DeepSeek-R1 (67.9%/45.0%), o3 (51.2%/44.0%) on MAMO/IndustryOR.

### Table 3: Ablation on RetailOpt-190

| Config | Claude Opus 4.6 ||| DeepSeek-V3.2 |||
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| | Exec% | ε=10⁻⁴ | ε=10⁻² | Exec% | ε=10⁻⁴ | ε=10⁻² |
| Direct | 72.1 | 22.6 | 26.8 | 91.1 | 0.5 | 3.7 |
| +CoT | 93.7 | 31.1 | 34.7 | 53.2 | 3.7 | 5.8 |
| +CoT+L1 | 99.5 | 31.1 | 35.3 | 97.4 | 5.8 | 10.5 |
| +CoT+L1+L2 (Full ReLoop) | **100.0** | **31.1** | **35.3** | **97.4** | **5.8** | **11.1** |



> **Model strength determines layer contributions and overall effectiveness:**
>
> - **Claude Opus 4.6** (strong model): CoT is the primary accuracy contributor (+8.5pp at ε=10⁻⁴, from 22.6% to 31.1%), demonstrating that structured 4-stage reasoning significantly improves formulation quality. L1 adds crash recovery (+5.8pp Exec%, 93.7% → 99.5%) and L2 behavioral testing brings execution to 100.0%. Accuracy is stable across layers — Claude's CoT formulations are already near-correct, so verification layers primarily serve as a safety net.
> - **DeepSeek-V3.2** (mid-tier model): L1 crash recovery is the dominant contributor (+44.2pp Exec%, 53.2% → 97.4%). CoT extraction frequently fails for DeepSeek (Exec drops from 91.1% Direct to 53.2% CoT), but L1 regeneration recovers most crashes and improves accuracy from 0.5% to 5.8% (ε=10⁻⁴) and 3.7% to 11.1% (ε=10⁻²). L2 behavioral testing contributes +1 problem at ε=10⁻² via the repair loop.
> - **32B models** (Qwen3, OptMATH, SIRL): Near-zero accuracy on RetailOpt-190 regardless of pipeline configuration. These models lack fundamental capacity for complex multi-period, multi-product retail optimization with 20+ parameters and dozens of constraints. ReLoop can improve execution rates (e.g., Qwen3: 0% → 2.1%) but cannot compensate for fundamentally incorrect formulations.
> - **Implication**: ReLoop's verification layers are most effective when the base model can produce *approximately correct* formulations that contain localized, detectable errors. For models below this capability threshold (on a given problem complexity), crash recovery (L1) provides the primary benefit.

---

## Appendix Tables

### Table A1: Per-Family Breakdown on RetailOpt-190 (Acc% pass@1, ε=10⁻⁴)

> Base = direct generation for foundation models, native format for SFT/RL models. +ReLoop = CoT + L1–L2 for all models.

| Family | #Inst | Claude Opus 4.6 || DeepSeek-V3.2 || Qwen3-32B || OptMATH-32B || SIRL-32B ||
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | | Base | +ReLoop | Base | +ReLoop | Base | +ReLoop | Base | +ReLoop | Base | +ReLoop |
| F1 Core Ops | 20 | 55.0 | **95.0** | 5.0 | **5.0** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| F2 Assort & Sub | 30 | 50.0 | **53.3** | 0.0 | **3.3** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| F3 Resource | 20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| F4 Demand Dyn | 30 | 3.3 | **20.0** | 0.0 | **10.0** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| F5 Feasibility | 20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| F6 Discrete Log | 20 | 0.0 | 0.0 | 0.0 | **20.0** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| F7 Network & ME | 30 | 20.0 | **26.7** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| F8 Omni-channel | 20 | **50.0** | **50.0** | 0.0 | **10.0** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **Total** | **190** | **22.6** | **31.1** | **0.5** | **5.8** | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** |

### Table A2: Silent Failure Analysis (Claude Opus 4.6 × RetailOpt-190)

> Of 190 baseline (direct generation) problems, 94 are silent failures: execute successfully but produce incorrect results (ε=10⁻⁴). We analyze how each ReLoop component addresses them.

**Coverage by component:**

| Component | Mechanism | Addressed |
|-----------|-----------|:---------:|
| Structured CoT | 4-stage reasoning with self-verification corrects formulation errors | 20 corrected |
| L1 crash recovery | Regeneration recovers 12 crashes (Exec 93.7% → 100.0%) | 0 additional accuracy |
| L2 behavioral testing (CPT + OPT) | Detects missing constraints/objective terms via perturbation | 0 additional accuracy |
| **CoT regression** | CoT generates different (incorrect) formulation for 4 problems | 4 hurt |
| **Net** | | **+16 net (20 corrected − 4 hurt)** |

> The 4 hurt problems (P18, P20, P140, P169) all break at the CoT stage — the structured reasoning produces a different formulation that executes successfully but with incorrect objectives. Verification layers (L1, L2) cannot detect these because the code runs without errors. The remaining ~74 silent failures involve complex structural formulation differences beyond automated verification scope.

**Detection capability by error type:**

| Error Type | L2 CPT | L2 OPT | Example |
|------------|:---:|:---:|---------|
| Missing objective term | — | ✓ | Revenue/cost component omitted |
| Missing constraint | ✓ | — | Capacity or demand limit not enforced |
| Variable type error | — | — | Continuous vs integer decision |
| Structural formulation | — | — | Incorrect decomposition or modeling approach |

---

## Ablation Study

The ablation study measures each component's marginal contribution. The pipeline records intermediate checkpoints during a single run:

| Config | What it captures | Pipeline state |
|--------|-----------------|----------------|
| **Direct** | Direct generation without CoT | No structured reasoning |
| **+CoT** | Structured generation output | Before any verification |
| **+CoT+L1** | After L1 verify + FATAL regeneration | Before L2 behavioral testing |
| **+CoT+L1+L2** (Full ReLoop) | After L2 behavioral testing + repair | Complete |

### Running ablation experiments

`run_ablation.py` records all checkpoints automatically (see [Quick Start](#quick-start) for full CLI usage):

```bash
python run_ablation.py -d data/RetailOpt-190.jsonl -m <model> --enable-cpt --workers 5 -v
```

### Analyzing layer contributions

```bash
python analyze_layers.py experiment_results/RetailOpt-190/<model>/ablation_report.csv
```

Output includes:
- Pass counts at each stage (auto-detected tolerance: ε=10⁻⁴/10⁻² for RetailOpt, ε=10⁻⁶ for cross-benchmark)
- Layer transitions: fail→pass (helped) vs pass→fail (hurt) per layer
- Crash recovery statistics
- Net contribution summary per layer

### Checkpoint design

The `PipelineResult` dataclass includes `l1_checkpoint_obj/status` and `l2_checkpoint_obj/status` fields, recorded automatically during `pipeline.run()`:

1. **L1 checkpoint**: Recorded after L1 verification and FATAL regeneration loop. Captures L1's contribution through crash recovery via code regeneration.
2. **L2 checkpoint**: Recorded after L2 behavioral testing (CPT + OPT). Since L2 behavioral testing does not modify code (only adds diagnostics), L2 checkpoint has the same objective as L1 checkpoint.
3. **Final**: The standard pipeline output after the full diagnostic-based repair loop, which processes diagnostics from all enabled layers (L1 + L2 CPT + L2 OPT).

---

## Project Structure

```
reloop/
├── reloop/                       # Core package
│   ├── __init__.py               # Public API exports
│   ├── param_utils.py            # Parameter extraction & perturbation (data-dict)
│   ├── perturbation.py           # Source-code level perturbation (AST-based) + mode detection
│   ├── executor.py               # Isolated subprocess execution with IIS/unbounded diagnostics
│   ├── verification.py           # 2-layer verification engine (L1 + L2 behavioral) + unified Diagnostic schema
│   ├── prompts.py                # LLM prompt templates
│   ├── generation.py             # Code generation
│   ├── repair.py                 # Diagnostic-based repair
│   ├── repair_safety.py          # Repair safety guardrails (data protection, import blocking)
│   ├── pipeline.py               # Pipeline orchestration (with ablation checkpoints)
│   ├── data_extraction.py        # NL → structured data extraction
│   └── experiment_runner.py      # Batch experiment runner
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_perturbation.py      # Unit tests for perturbation module
│   ├── test_e2e_perturbation.py  # E2E tests for L2 perturbation modes
│   └── test_repair_safety.py     # Unit tests for repair safety guardrails
├── scripts/                      # Deployment & automation scripts
│   ├── deploy_local_llm.py       # Launch local OpenAI-compatible LLM server (vLLM / llama.cpp)
│   └── run_all_local_ablation.sh # Run ablation experiments for all local models
├── data/                         # Benchmark datasets (JSONL)
├── fig/                          # Architecture diagrams
│   └── Reloop_framework.png      # System architecture diagram
├── run_ablation.py               # Ablation experiment runner (per-layer contribution)
├── analyze_layers.py             # Layer contribution analysis from ablation CSV
├── requirements.txt              # Python dependencies
├── requirements.local-inference.txt  # Optional: vLLM / llama.cpp for local serving
├── pyproject.toml                # Project configuration
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `ReLoopVerifier` | 2-layer verification engine (L1 execution + L2 behavioral testing) |
| `Diagnostic` | Unified diagnostic schema for all layers |
| `CodeGenerator` | Generate Gurobi code from problem description |
| `CodeRepairer` | Repair code based on diagnostics |
| `ReLoopPipeline` | Complete generate→verify→repair pipeline |
| `DataExtractor` | Extract structured data from natural language |
| `ExperimentRunner` | Run batch experiments on datasets |

### Verification

```python
from reloop import ReLoopVerifier, verify_code

# Quick verification
report = verify_code(code, data)

# With behavioral testing (requires LLM)
verifier = ReLoopVerifier(llm_client=llm)
report = verifier.verify(
    code, data,
    problem_description="...",
    enable_cpt=True  # enables both CPT and OPT
)

# Report fields
report.status      # 'VERIFIED' | 'WARNINGS' | 'ERRORS' | 'FAILED'
report.objective   # float (always present if L1 passes)
report.confidence  # 0.0 - 1.0
report.layer_results  # List[LayerResult]
```

### Pipeline

```python
from reloop import ReLoopPipeline, run_reloop

# Using class (full control)
pipeline = ReLoopPipeline(
    llm_client,
    max_repair_iterations=3,        # Repair attempts
    max_regeneration_attempts=3,    # L1 FATAL regeneration attempts
    enable_cpt=True,                # Enable L2 behavioral testing (CPT + OPT)
    use_structured_generation=True  # Use 4-stage CoT pipeline
)
result = pipeline.run(problem_description, data)

# Using convenience function
result = run_reloop(
    problem_description, data, llm_client,
    code=None,                 # Optional pre-generated code
    max_iterations=3,          # Repair iterations
    max_regenerations=3,       # Regeneration attempts
    enable_cpt=True,           # Enable L2 behavioral testing
    use_structured_generation=True,
    verbose=False
)

# Result fields
result.final_code         # str
result.final_report       # VerificationReport
result.iterations         # int (total verification iterations)
result.success            # bool
result.regeneration_count # int (L1 FATAL regenerations)
```

### Data Extraction

```python
from reloop import DataExtractor

extractor = DataExtractor(llm_client)
data = extractor.extract("""
    Factory has capacity 500 units.
    Product costs: $10, $15, $8.
    Demands: 100, 150, 200.
""")
# Returns: {"capacity": 500, "cost": [10, 15, 8], "demand": [100, 150, 200]}
```

### Experiments

```python
from reloop import ExperimentRunner, run_experiment

runner = ExperimentRunner(llm_client, output_dir="results")
summary = runner.run_dataset("data/RetailOpt-190.jsonl")

# Summary fields
summary.total_problems
summary.verified_count
summary.detection_rate
summary.false_positive_rate
summary.repair_success_rate
summary.avg_objective_error
summary.by_difficulty  # Dict[str, Dict]
```

---

## Chain-of-Thought Code Generation

ReLoop uses **Chain-of-Thought (CoT)** 4-stage generation:

```
Problem → [STEP 1: Understand] → [STEP 2: Formalize] → [STEP 3: Code] → [STEP 4: Verify] → Output
                    ↓                    ↓                   ↓                  ↓
              (same context)      (same context)      (same context)    (same context)
```

**Key Design:**

- **Step 2 (Formalize)** explicitly prompts the LLM to determine variable types (CONTINUOUS, INTEGER, or BINARY) for each decision variable by analyzing physical context. For example, "number of trucks" implies integer, while "volume of liquid" implies continuous.
- **Step 3 (Code)** instructs the model to implement ALL constraints from the **original problem description**, not just those in the Step 2 formulation. This prevents error propagation where Step 2 omissions cascade into the code.
- **Step 4 (Verify)** cross-checks the generated code against the original problem: are all cost/revenue terms in the objective? Are all constraints implemented? Are all data values correct? If anything is missing, the model fixes the code before returning.

**Generation Approaches:**

| Approach | Description | Note |
|----------|-------------|------|
| Single-Stage | Direct problem → code (baseline) | Higher error rate |
| 4-Stage CoT (single call) | All steps in one prompt | **Recommended** |
| 3-Stage CoT (separate calls) | 3 API calls (loses context) | Not recommended (loses reasoning chain) |

---

## Data Input Modes

ReLoop supports two data input modes for code generation. Both produce code that accesses data via `data["key"]` at runtime.

| Mode | LLM Sees | Use Case |
|------|----------|----------|
| **Data-embedded** | Full data values in prompt | Benchmark evaluation (all experiments in this paper) |
| **Schema-based** | Schema only (keys, types, dimensions) | Industrial deployment (data too large or sensitive for prompt) |

**Data-embedded mode** (used for all benchmark evaluations):
- The complete data dict is embedded directly in the problem prompt
- LLM sees actual values and generates code accordingly
- Simpler and more reliable for fixed-size benchmark problems

**Schema-based mode** (for production deployment):
- LLM sees only schema descriptions (e.g., `capacity: int`, `costs: list[3] of int`)
- Actual data values are injected at runtime via the `data` dict
- Enables deployment on large-scale or confidential datasets where embedding full data in the prompt is impractical

**Code Design Principle:** Regardless of input mode, generated code must use `data["key"]` to access data at runtime. The `data` dict is injected by the executor, and repair prompts also include data context to ensure consistency.

**Big-M Guidelines (for indicator/logical constraints):**
- NEVER hardcode Big-M values like `M = 1e6`
- ALWAYS compute M dynamically: `M = sum(data["demand"]) * 1.5`
- Use problem-relevant parameters for M calculation

---

## 2-Layer Verification Architecture

| Layer | Name | Type | Description |
|-------|------|------|-------------|
| L1 | Execution & Solver + Duality | Blocking | Syntax, runtime, solver status with IIS/unbounded diagnostics → triggers regeneration on FATAL; duality check as add-on diagnostic (INFO) |
| L2 | Behavioral Testing (CPT + OPT) | Diagnostic | Solver-based perturbation testing: CPT detects missing constraints (WARNING/INFO), OPT detects missing objective terms (WARNING/INFO) |

**Severity Levels (Conservative Repair Strategy):**

| Severity | Confidence | Source | Repair Action |
|----------|------------|--------|---------------|
| `FATAL` | 100% | L1 only | Triggers regeneration (up to 3 attempts) |
| `WARNING` | 80%+ | L2 CPT (cpt_missing), L2 OPT (opt_missing) | **SHOULD fix** |
| `INFO` | <80% | L1 duality, L2 CPT/OPT (uncertain) | **DO NOT fix** (reference only) |
| `PASS` | - | All layers | No action needed |

**Key Design Principle:**
- Only WARNING triggers repair
- INFO is for reference only — likely normal optimization behavior (slack constraints, numerical artifacts)
- This prevents over-correction that was causing ReLoop to perform worse than baseline

### Perturbation Modes (Source-Code vs Data-Dict)

L2 behavioral testing (CPT + OPT) performs parameter perturbation to test model completeness. ReLoop supports two perturbation strategies with automatic detection:

| Mode | Detection Criterion | Perturbation Strategy | Typical Datasets |
|------|---------------------|-----------------------|------------------|
| `data_dict` | Code uses `data["key"]` pattern | Perturb the data dict values (existing) | RetailOpt-190 |
| `source_code` | Code hardcodes numeric values, no `data[` access | Perturb via AST-based code rewriting | IndustryOR, MAMO |
| `hybrid` | Both patterns present | Try data-dict first; fallback to source-code if no effect | Mixed |

**Auto-Detection:** `detect_perturbation_mode(code, data)` inspects the generated code for `data[` access patterns and counts hardcoded numeric assignments. The result determines which perturbation strategy L2 uses.

**Source-Code Perturbation Flow:**
```
LLM Code:    capacity = 500; demand = 300
                     ↓ AST Parse
Extract:     [{name: "capacity", value: 500, access_path: "capacity"}, ...]
                     ↓ perturb_code(code, "capacity", 1.2)
Perturbed:   capacity = 600; demand = 300
                     ↓ Execute
Objective:   Compare with baseline to detect anomalies
```

**Hybrid Fallback:** For `hybrid` mode, L2 first tries data-dict perturbation. If the perturbed objective is unchanged (parameter hardcoded in code rather than read from data), the system falls back to AST-based source-code perturbation.

**JSON.loads Data Override Stripping:** When generated code contains `data = json.loads("""...""")` (self-contained pattern from fallback generation), this line overwrites the externally-injected `data` dict, making all data-dict perturbations ineffective (0% change). ReLoop automatically detects and strips these data override lines before perturbation execution (`perturbation.py:strip_data_override`), allowing the perturbed external data dict to flow through correctly. This handles three patterns:
- Indirect: `data_json = """..."""; data = json.loads(data_json)` (both lines removed)
- Inline: `data = json.loads("""...""")` (entire statement removed)
- Single-line: `data = json.loads('...')` (line removed)

### L2 CPT: Constraint Presence Testing

```
┌─────────────────────────────────────────────────────────────┐
│                 L2 CPT Flow                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Problem Description ──→ LLM Extract Constraints            │
│                              │                              │
│                              ▼                              │
│               Candidate Constraints List                    │
│               [protein_min, carbs_min, ...]                 │
│                              │                              │
│                              ▼                              │
│         For each constraint: Extreme Perturbation           │
│         (e.g., scale demand 100x, capacity to 0.001)        │
│                              │                              │
│                              ▼                              │
│         Measure Objective Change Ratio                      │
│              < 5%  → WARNING (likely missing)               │
│            5-30%   → INFO (uncertain)                       │
│              > 30% → PASS (constraint present)              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### L2 OPT: Objective Presence Testing

```
┌─────────────────────────────────────────────────────────────┐
│                 L2 OPT Flow                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Problem Description ──→ LLM Extract Objective Terms        │
│                              │                              │
│                              ▼                              │
│               Candidate Objective Terms List                │
│               [{unit_cost, role=cost}, ...]                 │
│                              │                              │
│                              ▼                              │
│         For each term: Role-specific Perturbation           │
│         (cost ×0.001, revenue ×100, other ×0.01)            │
│                              │                              │
│                              ▼                              │
│         Measure Objective Change Ratio                      │
│              < 5%  → WARNING (likely missing)               │
│            5-30%   → INFO (uncertain)                       │
│              > 30% → PASS (objective term present)          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### L1: IIS and Unbounded Diagnostics

When the solver returns **INFEASIBLE**, L1 automatically computes the **Irreducible Inconsistent Subsystem (IIS)** using Gurobi, identifying the minimal set of conflicting constraints:

```
[L1] INFEASIBLE - IIS Analysis:
  Conflicting constraints:
    - demand_1 (> 100.0)
    - supply_1 (< 50.0)
  Conflicting variable bounds:
    - x UB=10
```

For **UNBOUNDED** models, L1 reports the unbounded ray variables:

```
[L1] UNBOUNDED - Ray Analysis:
  Unbounded variables:
    - x (ray=1.0000)
    - y (ray=0.5000)
```

**Implementation:** The executor uses an `exec()` + `try/except` wrapper (not `atexit`) to run IIS diagnostics while the Gurobi model is still alive in memory.

### Unified Diagnostic Schema

All verification layers output results in a unified `Diagnostic` format:

```python
@dataclass
class Diagnostic:
    layer: str          # "L1", "L2"
    issue_type: str     # "INFEASIBLE", "UNBOUNDED", "RUNTIME_ERROR", "MISSING_CONSTRAINT", "MISSING_OBJECTIVE_TERM", etc.
    severity: str       # "WARNING", "INFO"
    target_name: str    # Which parameter/constraint
    evidence: str       # Auto-generated evidence description
    triggers_repair: bool  # Whether this triggers repair
```

The repair pipeline collects `Diagnostic` objects from all layers and uses `build_repair_prompt()` to assemble a structured repair prompt with:
- **Actionable issues** (triggers_repair=True): listed with full evidence
- **Reference items** (triggers_repair=False): shown as context only, explicitly marked "DO NOT FIX"

### Repair Safety

ReLoop enforces safety guardrails on repair LLM outputs to prevent data corruption:

| Check | Method | What it catches |
|-------|--------|----------------|
| Data reassignment (dict literal) | AST + Regex | `data = {...}` (fabricated data) |
| Data reassignment (other) | AST | `data = some_function()` (unknown source) |
| Data mutation | AST + Regex | `data["key"] = value`, `data["key"] += value` |
| Dangerous imports | Regex | `import os`, `import subprocess` |

Note: `data = json.loads(...)` is **allowed** — it re-parses existing data rather than fabricating new values. Prompts discourage this pattern but it is not blocked by the safety check.

**Enforcement flow:**
1. Repair LLM generates fixed code
2. `validate_repair_code()` checks for safety violations
3. If violations found: guided re-repair with explicit safety rules (1 retry, does not consume repair budget)
4. If second attempt also violates: discard repair, keep original code

**Motivation:** In testing, repair LLMs were observed redefining the `data` variable (e.g., changing `protein_requirement` from 83 to 120), which corrupted the problem and produced worse results than the original code.

---

## Data Format

JSONL format, one problem per line:

```json
{"en_question": "Problem description...", "en_answer": "123.45", "difficulty": "Easy", "id": 1}
```

---

## Citation

```bibtex
@article{lian2026reloop,
  title   = {ReLoop: Detecting Silent Failures in LLM-Generated Optimization Code via Behavioral Verification},
  author  = {Lian, Junbo Jacob and Sun, Yujun and Chen, Huiling and Zhang, Chaoyu and Teo, Chung-Piaw},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
