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

ReLoop is a behavioral verification framework that detects **silent failures** in LLM-generated optimization code—code that executes successfully but produces incorrect results. Our key insight is that correct optimization models satisfy fundamental invariants (execution correctness, semantic consistency, constraint presence) that can be tested without ground truth.

**Key Results:**
- Consistent improvements across all five models and three benchmarks
- Detection of missing cost terms, wrong coefficients, missing constraints, and data access errors
- No ground truth required for verification

---

## Framework Architecture

![ReLoop Framework Architecture](fig/Reloop_framework.png)

### Key Components

| Component | Description |
|-----------|-------------|
| **Chain-of-Thought Generation** | 3-stage structured generation: Understand → Formalize (with variable type reasoning) → Synthesize |
| **L1 Execution & Solver** | Syntax, runtime, and solver status checks with IIS/unbounded diagnostics (FATAL → regeneration) + duality check (INFO) |
| **L2 Semantic Audit** | LLM-based adversarial audit: compares problem description vs code for missing terms, wrong coefficients, missing constraints |
| **L3 Constraint Presence Test** | LLM-based verification of expected constraints in generated code (CPT) |
| **Unified Diagnostic Schema** | All layers output `Diagnostic` objects; unified `build_repair_prompt()` assembles repair prompts |
| **Diagnostic-Based Repair** | Conservative strategy: only ERROR/WARNING trigger repair, INFO is reference only |
| **Repair Safety Guardrails** | Prevents repair LLM from modifying input data or introducing dangerous operations |
| **Repair Regression Guard** | Post-repair rollback on crash, status degradation, or objective value drift (>4%) vs pre-repair baseline |
| **Repair Skip Guard** | Skip repair loop entirely when code is already VERIFIED with no actionable diagnostics |

### Design Principles

1. **Universal Verification**: All checks work without ground truth or domain-specific rules
2. **Conservative Repair**: Only high-confidence issues (ERROR/WARNING) trigger repairs
3. **Robustness Guarantee**: L1 FATAL triggers regeneration; L2-L3 are diagnostic only
4. **LLM-Only Analysis**: No keyword-based heuristics; all semantic analysis uses LLM
5. **Repair Safety**: Repair code is validated before execution; data variable cannot be reassigned
6. **Regression Guard**: Repair never makes results worse — rollback on crash, status degradation, or value drift >4%
7. **Repair Skip**: Already-verified code is not subjected to unnecessary repair attempts

### Generation: Chain-of-Thought (CoT)

A single LLM call with 3-stage structured reasoning (`generation.py`, `prompts.py`):

1. **Understand** — Identify objective (min/max), decisions, constraints, parameters
2. **Formalize** — Write the mathematical model: sets, parameters, decision variables (with explicit CONTINUOUS/INTEGER/BINARY type reasoning), constraints, objective
3. **Synthesize** — Generate executable Gurobi Python code

The code receives a pre-defined `data` dict and must not redefine it. Big-M values must be computed dynamically from data, never hardcoded.

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

### L2: Semantic Audit (Adversarial)

Detects semantic discrepancies between the problem description and generated code (`l2_direction.py`, `pipeline.py:_run_l2_adversarial_loop`).

**Procedure:**

1. **LLM_audit** (temperature=0) — Single LLM call compares problem description against generated code using a 3-part checklist:
   - **Objective function completeness**: Is every cost/revenue term mentioned in the problem present in the code's objective?
   - **Coefficient semantic correctness**: Are coefficients correctly interpreted? (rate vs total return, per-unit vs aggregate, percentage vs absolute)
   - **Constraint completeness**: Are all constraints from the problem description present in the code?
   - Output per issue: `issue_type` (missing_term / wrong_coefficient / missing_constraint), `target`, `evidence`, `confidence` (0.0-1.0), `suggested_fix`
2. **LLM_repair** (temperature=0.3) — For each finding with confidence >= 0.5, repair LLM decides:
   - **Accept** — Agrees the finding is correct, provides fixed code
   - **Reject** — Disagrees with the finding, provides rejection reason
   - Higher temperature encourages diverse reasoning for the adversarial "devil's advocate" role
3. **Re-audit** — Rejected findings are re-audited with rejection context (up to `max_rejections=2` per finding)

**Exit conditions:**

| Condition | Exit Reason | Severity |
|-----------|-------------|----------|
| No issues found | `all_pass` | PASS |
| All findings rejected + L1/L3 PASS | `all_rejected_others_pass` | INFO |
| Max rejections reached for all findings | `max_rejections` (downgrade) | INFO |
| Some accepted, code fixed | `accepted_fixed` | ERROR (triggers repair) |
| Max L2 iterations (3) reached | `max_iterations` | ERROR |

### L3: Constraint Presence Test (CPT, Optional)

Tests whether expected constraints are actually active in the generated model (`verification.py:_layer3`). Enabled by `--enable-cpt`.

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

### Diagnostic-Based Repair

All layers output unified `Diagnostic` objects with severity and evidence (`verification.py:Diagnostic`, `pipeline.py`).

**Severity-to-action mapping:**

| Severity | Source | Action |
|----------|--------|--------|
| FATAL | L1 execution errors | Regenerate code (not repair) |
| ERROR | L2 accepted semantic audit findings | **Must fix** — included in repair prompt |
| WARNING | L3 CPT missing constraints | **Should fix** — included in repair prompt |
| INFO | L1 duality, L2 rejected, L3 uncertain | **Reference only** — shown but not fixed |

**Repair loop** (`pipeline.py`, budget `N=3`):
1. Collect all diagnostics → `build_repair_prompt()` assembles prompt with actionable issues (ERROR/WARNING) and reference context (INFO)
2. LLM generates repaired code
3. **Safety guardrail** (`repair_safety.py`) — Validates repair code before execution:
   - Blocks: `data = {...}` (reassignment), `data["key"] = val` (mutation), dangerous imports (`os`, `subprocess`)
   - Exception: `data = json.loads(...)` is allowed (re-parses existing data)
   - On violation: 1 guided retry (does not consume repair budget); 2nd failure → keep original code
4. Re-verify repaired code
5. **Repair skip guard** — If the code is already VERIFIED with no ERROR/WARNING diagnostics, skip the entire repair loop (prevents unnecessary modifications to correct code)
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
| `--enable-cpt` | off | Enable L3 Constraint Presence Test |
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
| `MAMO_EasyLP_fixed.jsonl` | 642 | ~168 | 10⁻⁶ | Data-embedded |
| `IndustryOR_fixedV2.jsonl` | 100 | ~267 | 10⁻⁶ | Data-embedded |
All datasets use data-embedded format (full data in prompt) for evaluation. RetailOpt additionally provides schema-based prompts for ReLoop's verification pipeline, where the LLM sees only data schema and actual values are injected at runtime.

---

## Compared Models

| Type | Model | Provider | Temp. | Max Tokens | Notes |
|------|-------|----------|:---:|:---:|-------|
| Foundation | Claude Opus 4.6 | Anthropic API | 0.0 | 8192 | |
| Foundation | DeepSeek-V3.1 | DeepSeek API | 0.0 | 8192 | |
| Foundation | Qwen3-32B | Local (vLLM) | 0.0 | 8192 | BF16 |
| Offline SFT | OptMATH-Qwen2.5-32B | Local (llama.cpp) | 0.0 | 8192 | GGUF |
| Online RL | SIRL-Qwen2.5-32B | Local (vLLM) | 0.0 | 8192 | BF16 |

---

## Experiment Results

### Table 1: Main Results on RetailOpt-190

> Foundation models: Base = direct generation, CoT = structured chain-of-thought, ReLoop = CoT + L1–L3 verification with repair. SFT/RL models: Base = native generation format, CoT = our structured CoT replacing native prompting, ReLoop = CoT + L1–L3 (same pipeline as foundation models).

| Type | Model | Exec% ||| Acc% (ε=10⁻⁴) ||| Acc% (ε=10⁻²) |||
|------|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | | Base | CoT | ReLoop | Base | CoT | ReLoop | Base | CoT | ReLoop |
| Foundation | Claude Opus 4.6 | 80.0 | 79.5 | **98.4** | 27.4 | 28.9 | **31.6** | 31.1 | 31.6 | **35.3** |
| Foundation | DeepSeek-V3.1 | 76.8 | 69.5 | **96.3** | **2.1** | 1.1 | 1.1 | **6.3** | 3.7 | 5.3 |
| Foundation | Qwen3-32B | 0.0 | 0.0 | **2.1** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Offline SFT | OptMATH-Qwen2.5-32B | 0.5 | 0.0 | **4.7** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Online RL | SIRL-Qwen2.5-32B | 1.6 | 1.6 | 1.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### Table 2: Cross-Benchmark Generalization (Acc% pass@1, ε=10⁻⁶)

| | | MAMO-ComplexLP ||| IndustryOR |||
|------|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| Type | Model | Base | CoT | +ReLoop | Base | CoT | +ReLoop |
| Foundation | Claude Opus 4.6 | 78.8 | 79.8 | **82.3** | 69.0 | 69.0 | **69.0** |
| Foundation | DeepSeek-V3.1 | 60.6 | 62.1 | **63.5** | 44.0 | 58.0 | **60.0** |
| Foundation | Qwen3-32B | 29.1 | 30.0 | **36.0** | 38.0 | 44.0 | **47.0** |
| Offline SFT | OptMATH-Qwen2.5-32B | 44.8 | 45.8 | **46.3** | 31.0 | 33.0 | **33.0** |
| Online RL | SIRL-Qwen2.5-32B | 56.7 | 54.2 | **57.6** | 47.0 | 47.0 | **48.0** |

> For reference from SIRL (Chen et al., 2025) and StepORLM (Zhou et al., 2025): GPT-4 (49.3%/33.0%), DeepSeek-R1 (67.9%/45.0%), o3 (51.2%/44.0%) on MAMO/IndustryOR.

### Table 3: Ablation on RetailOpt-190

| Config | Claude Opus 4.6 ||| DeepSeek-V3.1 |||
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| | Exec% | ε=10⁻⁴ | ε=10⁻² | Exec% | ε=10⁻⁴ | ε=10⁻² |
| Direct | 80.0 | 27.4 | 31.1 | 76.8 | **2.1** | **6.3** |
| +CoT | 79.5 | 28.9 | 31.6 | 69.5 | 1.1 | 3.7 |
| +CoT+L1 | 97.4 | 31.6 | 35.3 | **96.3** | 1.1 | 5.3 |
| +CoT+L1+L2 | 97.4 | 31.6 | 35.3 | **96.3** | 1.1 | 5.3 |
| +CoT+L1+L2+L3 | **98.4** | **31.6** | **35.3** | **96.3** | 1.1 | 5.3 |

> **Model strength determines layer contributions and overall effectiveness:**
>
> - **Claude Opus 4.6** (strong model): CoT and L1 provide the primary accuracy gains (+4.2pp at ε=10⁻⁴). L2/L3 serve as safety nets — the regression guard prevents over-repair. Claude's baseline formulations are already near-correct, so L2 semantic fixes rarely improve accuracy. L3 contributes +1pp Exec%.
> - **DeepSeek-V3.1** (mid-tier model): L1 crash recovery is the dominant contributor (+26.8pp Exec%). However, **CoT actually hurts accuracy** (2.1% → 1.1% at ε=10⁻⁴): the structured reasoning format constrains a model that lacks sufficient capacity for complex retail optimization, producing worse formulations than direct generation. L2/L3 have no additional effect — the formulation errors in the surviving (non-crashing) code are structural (wrong decomposition, incorrect modeling approach) rather than the localized semantic issues (missing terms, wrong coefficients) that L2 can detect.
> - **32B models** (Qwen3, OptMATH, SIRL): Near-zero accuracy on RetailOpt-190 regardless of pipeline configuration. These models lack fundamental capacity for complex multi-period, multi-product retail optimization with 20+ parameters and dozens of constraints. ReLoop can improve execution rates (e.g., Qwen3: 0% → 2.1%) but cannot compensate for fundamentally incorrect formulations.
> - **Implication**: ReLoop's verification layers are most effective when the base model can produce *approximately correct* formulations that contain localized, detectable errors. For models below this capability threshold (on a given problem complexity), crash recovery (L1) provides the primary benefit.

---

## Appendix Tables

### Table A1: Per-Family Breakdown on RetailOpt-190 (Acc% pass@1, ε=10⁻⁴)

> Base = direct generation for foundation models, native format for SFT/RL models. +ReLoop = CoT + L1–L3 for all models.

| Family | #Inst | Claude Opus 4.6 || DeepSeek-V3.1 || Qwen3-32B || OptMATH-32B || SIRL-32B ||
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | | Base | +ReLoop | Base | +ReLoop | Base | +ReLoop | Base | +ReLoop | Base | +ReLoop |
| F1 Core Ops | 20 | 85.0 | **90.0** | 5.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| F2 Assort & Sub | 30 | 46.7 | **46.7** | 3.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| F3 Resource | 20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| F4 Demand Dyn | 30 | 23.3 | **33.3** | 0.0 | **6.7** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| F5 Feasibility | 20 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| F6 Discrete Log | 20 | 0.0 | 0.0 | 5.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| F7 Network & ME | 30 | 13.3 | **23.3** | 3.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| F8 Omni-channel | 20 | 50.0 | **55.0** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **Total** | **190** | **27.4** | **31.6** | **2.1** | **1.1** | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** |

### Table A2: Silent Failure Analysis (Claude Opus 4.6 × RetailOpt-190)

> Of 190 baseline (direct generation) problems, 100 are silent failures: execute successfully but produce incorrect results (ε=10⁻⁴). We analyze how each ReLoop component addresses them.

**Coverage by component:**

| Component | Mechanism | Addressed |
|-----------|-----------|:---------:|
| Structured CoT | 3-stage reasoning prevents formulation errors | 9 corrected |
| L1 crash recovery | Regeneration yields correct formulation | 1 corrected |
| L2 semantic audit | Detects missing objective terms, wrong coefficients, constraint gaps | 5 flagged |
| L3 CPT | Detects missing/inactive constraints via perturbation testing | 7 flagged |
| Regression guard | Prevents over-repair from degrading correct solutions (>4% drift) | 2 protected |
| **Total** | | **10 corrected, 11 flagged** |

> The remaining 79 involve complex structural formulation differences (e.g., alternative model decompositions, multi-component errors) beyond automated verification scope.

**Detection capability by error type:**

| Error Type | L2 | L3 | Example |
|------------|:---:|:---:|---------|
| Missing objective term | ✓ | — | Revenue/cost component omitted |
| Wrong coefficient/unit | ✓ | — | Per-unit cost vs aggregate, rate vs total |
| Missing constraint | ✓ | ✓ | Capacity or demand limit not enforced |
| Variable type error | — | — | Continuous vs integer decision |
| Structural formulation | — | — | Incorrect decomposition or modeling approach |

---

## Ablation Study

The ablation study measures each component's marginal contribution. The pipeline records intermediate checkpoints during a single run:

| Config | What it captures | Pipeline state |
|--------|-----------------|----------------|
| **Direct** | Direct generation without CoT | No structured reasoning |
| **+CoT** | Structured generation output | Before any verification |
| **+CoT+L1** | After L1 verify + FATAL regeneration | Before L2 |
| **+CoT+L1+L2** | After L2 semantic audit | Before L3 |
| **+CoT+L1+L2+L3** (Full ReLoop) | After full pipeline | Complete |

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

1. **L1 checkpoint**: Recorded after L1 verification and FATAL regeneration loop (Step 3). Captures L1's contribution through crash recovery via code regeneration.
2. **L2 checkpoint**: Recorded after L2 semantic audit loop (Step 4). Captures L2's additional contribution through semantic audit and repair.
3. **Final**: The standard pipeline output after the full diagnostic-based repair loop (Step 5), which processes diagnostics from all enabled layers.

---

## Project Structure

```
reloop/
├── reloop/                       # Core package
│   ├── __init__.py               # Public API exports
│   ├── param_utils.py            # Parameter extraction & perturbation (data-dict)
│   ├── perturbation.py           # Source-code level perturbation (AST-based) + mode detection
│   ├── executor.py               # Isolated subprocess execution with IIS/unbounded diagnostics
│   ├── verification.py           # 3-layer verification engine (L1-L3) + unified Diagnostic schema
│   ├── l2_direction.py           # L2 Semantic Audit (adversarial LLM debate)
│   ├── prompts.py                # LLM prompt templates
│   ├── generation.py             # Code generation
│   ├── repair.py                 # Diagnostic-based repair with L2 Accept/Reject
│   ├── repair_safety.py          # Repair safety guardrails (data protection, import blocking)
│   ├── pipeline.py               # Pipeline orchestration (with ablation checkpoints)
│   ├── data_extraction.py        # NL → structured data extraction
│   └── experiment_runner.py      # Batch experiment runner
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_perturbation.py      # Unit tests for perturbation module
│   ├── test_e2e_perturbation.py  # E2E tests for L3 perturbation modes
│   └── test_repair_safety.py     # Unit tests for repair safety guardrails
├── scripts/                      # Deployment & automation scripts
│   ├── deploy_local_llm.py       # Launch local OpenAI-compatible LLM server (vLLM / llama.cpp)
│   └── run_all_local_ablation.sh # Run ablation experiments for all local models
├── data/                         # Benchmark datasets (JSONL)
├── fig/                          # Architecture diagrams
│   ├── Reloop_framework.png      # System architecture diagram
│   └── L2.png                    # L2 Semantic Audit diagram
├── run_ablation.py               # Ablation experiment runner (per-layer contribution)
├── run_one.py                    # Run single problem for debugging
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
| `ReLoopVerifier` | 3-layer verification engine |
| `Diagnostic` | Unified diagnostic schema for all layers |
| `L2DirectionVerifier` | L2 Semantic Audit with LLM adversarial debate |
| `CodeGenerator` | Generate Gurobi code from problem description |
| `CodeRepairer` | Repair code based on diagnostics (with L2 Accept/Reject) |
| `ReLoopPipeline` | Complete generate→verify→repair pipeline |
| `DataExtractor` | Extract structured data from natural language |
| `ExperimentRunner` | Run batch experiments on datasets |

### Verification

```python
from reloop import ReLoopVerifier, verify_code

# Quick verification
report = verify_code(code, data)

# With CPT (requires LLM)
verifier = ReLoopVerifier(llm_client=llm)
report = verifier.verify(
    code, data,
    problem_description="...",
    enable_cpt=True
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
    max_repair_iterations=3,        # L2-L3 repair attempts
    max_regeneration_attempts=3,    # L1 FATAL regeneration attempts
    enable_cpt=True,                # Enable L3 CPT layer
    enable_l2_adversarial=True,     # Enable L2 Semantic Audit
    use_structured_generation=True  # Use 3-stage pipeline
)
result = pipeline.run(problem_description, data)

# Using convenience function
result = run_reloop(
    problem_description, data, llm_client,
    code=None,                 # Optional pre-generated code
    max_iterations=3,          # Repair iterations
    max_regenerations=3,       # Regeneration attempts
    enable_cpt=True,           # Enable L3 CPT layer
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

ReLoop uses **Chain-of-Thought (CoT)** 3-stage generation:

```
Problem → [STEP 1: Understand] → [STEP 2: Formalize] → [STEP 3: Code] → Output
                    ↓                    ↓                   ↓
              (same context)      (same context)      (same context)
```

**Formalization Stage — Variable Type Reasoning:**

The formalization step explicitly prompts the LLM to determine variable types (CONTINUOUS, INTEGER, or BINARY) for each decision variable by analyzing physical context. For example, "number of trucks" implies integer, while "volume of liquid" implies continuous. This reduces downstream variable type errors.

**Generation Approaches:**

| Approach | Description | Note |
|----------|-------------|------|
| Single-Stage | Direct problem → code (baseline) | Higher error rate |
| 3-Stage CoT (single call) | All steps in one prompt | **Recommended** |
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

## 3-Layer Verification Architecture

| Layer | Name | Type | Description |
|-------|------|------|-------------|
| L1 | Execution & Solver + Duality | Blocking | Syntax, runtime, solver status with IIS/unbounded diagnostics → triggers regeneration on FATAL; duality check as add-on diagnostic (INFO) |
| L2 | Semantic Audit | Diagnostic | LLM-based adversarial semantic verification with Accept/Reject |
| L3 | Constraint Presence Test (CPT) | Enhancement | LLM-based constraint testing (WARNING/INFO) |

**Severity Levels (Conservative Repair Strategy):**

| Severity | Confidence | Source | Repair Action |
|----------|------------|--------|---------------|
| `FATAL` | 100% | L1 only | Triggers regeneration (up to 3 attempts) |
| `ERROR` | 99%+ | L2 semantic audit (accepted) | **MUST fix** |
| `WARNING` | 80%+ | L3 cpt_missing | **SHOULD fix** |
| `INFO` | <80% | L1 duality, L2 audit (rejected/inconclusive) | **DO NOT fix** (reference only) |
| `PASS` | - | All layers | No action needed |

**Key Design Principle:**
- Only ERROR and WARNING trigger repair
- INFO is for reference only - likely normal optimization behavior (slack constraints, numerical artifacts)
- This prevents over-correction that was causing ReLoop to perform worse than baseline

### Perturbation Modes (Source-Code vs Data-Dict)

L3 (CPT) performs parameter perturbation to test constraint presence. ReLoop supports two perturbation strategies with automatic detection:

| Mode | Detection Criterion | Perturbation Strategy | Typical Datasets |
|------|---------------------|-----------------------|------------------|
| `data_dict` | Code uses `data["key"]` pattern | Perturb the data dict values (existing) | RetailOpt-190 |
| `source_code` | Code hardcodes numeric values, no `data[` access | Perturb via AST-based code rewriting | IndustryOR, MAMO |
| `hybrid` | Both patterns present | Try data-dict first; fallback to source-code if no effect | Mixed |

**Auto-Detection:** `detect_perturbation_mode(code, data)` inspects the generated code for `data[` access patterns and counts hardcoded numeric assignments. The result determines which perturbation strategy L3 uses.

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

**Hybrid Fallback:** For `hybrid` mode, L3 first tries data-dict perturbation. If the perturbed objective is unchanged (parameter hardcoded in code rather than read from data), the system falls back to AST-based source-code perturbation.

Note: L2 (Semantic Audit) does not use perturbation — it performs direct LLM-based comparison of problem description vs code.

### L2: Semantic Audit (LLM-based Adversarial)

L2 uses an **adversarial mechanism** where two LLM roles debate to converge on the correct analysis:

```
┌─────────────────────────────────────────────────────────────┐
│                 L2 Adversarial Flow                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LLM_audit ──→ "Purchasing cost missing from objective"     │
│       │                                                     │
│       ▼                                                     │
│  LLM_repair ──→ Accept? ──→ YES ──→ ERROR (must fix)       │
│       │              │                                      │
│       │              └──→ NO (Reject) ──→ Re-audit          │
│       │                        │                            │
│       │                        ▼                            │
│       │              LLM_audit (with rejection feedback)    │
│       │                        │                            │
│       │                        ▼                            │
│       │              [Repeat until Accept or max rejections]│
│       │                                                     │
│       └──→ Max rejections reached ──→ INFO (inconclusive)  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Semantic Audit Checklist:**
1. **Objective completeness**: Every cost/revenue term in problem → present in code objective?
2. **Coefficient semantics**: Rate vs total return? Per-unit vs aggregate? Percentage vs absolute?
3. **Constraint completeness**: All constraints from problem description → present in code?

**Exit Conditions (forces output):**
1. `all_pass`: No issues found → output
2. `all_rejected_others_pass`: All findings rejected + L1/L3 PASS → output (INFO level)
3. `max_rejections`: Max rejections per finding reached (default 2) → downgrade to INFO → output
4. `max_iterations`: Reached max L2 iterations (default 3) → output
5. `accepted_fixed`: Some accepted, code fixed → re-verify and continue

**Key Parameters:**
- `max_l2_rejections`: Max times a finding can be rejected before downgrade (default: 2)
- `max_l2_iterations`: Max L2 loop iterations (default: 3)

> **Design Principle:** The adversarial mechanism allows two LLM perspectives to debate.
> The audit role identifies potential semantic discrepancies, while the repair role acts as a "devil's advocate" to filter false positives before any code changes are made.

**Robustness Guarantee:**
- L1 `FATAL` triggers regeneration, not immediate termination
- L2-L3 are diagnostic only: never block output
- L2 loop always exits with output (one of the exit conditions will be met)
- L2 regression guard: accepted fixes that cause objective drift >5% are rejected (prevents over-repair)
- False positives don't affect result values (objective/solution always returned if L1 passes)
- INFO-level issues do NOT trigger repair (prevents over-correction)

### L3: CPT (Constraint Presence Testing)

L3 uses **LLM-based constraint extraction** to identify expected constraints, then tests if they are present in the generated code:

```
┌─────────────────────────────────────────────────────────────┐
│                 L3 CPT Flow                                  │
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

**Threshold-Based Detection:**

| Change Ratio | Severity | Interpretation |
|--------------|----------|----------------|
| < 5% | **WARNING** | Constraint likely missing - extreme perturbation had no effect |
| 5-30% | INFO | Uncertain - may be partially active |
| > 30% | PASS | Constraint present - perturbation significantly affected objective |

**Example Output:**
```
[L3] CPT
  Extracted 3 candidates:
  ├─ [MISSING] minimum protein requirement - 0.0% change
  ├─ [UNCERTAIN] minimum carbs requirement - 21.1% change
  └─ [PRESENT] minimum calories requirement - 38.0% change
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
    layer: str          # "L1", "L2", "L3"
    issue_type: str     # "INFEASIBLE", "UNBOUNDED", "RUNTIME_ERROR", "missing_term", "wrong_coefficient", "missing_constraint", etc.
    severity: str       # "ERROR", "WARNING", "INFO"
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
