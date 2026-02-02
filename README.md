# ReLoop: Detecting Silent Failures in LLM-Generated Optimization Code via Behavioral Verification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-red.svg)]()

This repository contains the official implementation of the paper:

> **ReLoop: Detecting Silent Failures in LLM-Generated Optimization Code via Behavioral Verification**
>
> Junbo Jacob Lian, Yujun Sun, Huiling Chen, Chaoyu Zhang, Chung-Piaw Teo
>
> NeurIPS 2026

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

ReLoop is a behavioral verification framework that detects **silent failures** in LLM-generated optimization code—code that executes successfully but produces incorrect results. Our key insight is that correct optimization models satisfy fundamental mathematical invariants (anomaly detection, dual consistency, direction analysis, constraint presence) that can be tested without ground truth.

**Key Results:**
- 94% detection rate for silent failures
- 3% false positive rate on correct code
- No ground truth required for verification

---

## Framework Architecture

> 📄 **Full architecture diagram:** [fig/Reloop_framework.pdf](fig/Reloop_framework.pdf)
>
> *Download the PDF for the complete visual architecture used in the paper.*

### Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           ReLoop Pipeline                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    GENERATION (Chain-of-Thought)                     │    │
│  │  Problem → [Understand] → [Formalize] → [Synthesize] → Gurobi Code  │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                            │
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    5-LAYER VERIFICATION                              │    │
│  │                                                                      │    │
│  │  L1: Execution & Solver ─────────────────────────────┐              │    │
│  │      (Blocking: FATAL → Regenerate up to 3x)         │              │    │
│  │                                                       ▼              │    │
│  │  L2: Anomaly Detection ──────────────────────────────┤              │    │
│  │      (Both ↑↓ improve → ERROR)                        │              │    │
│  │                                                       ▼              │    │
│  │  L3: Dual Consistency ───────────────────────────────┤              │    │
│  │      (Primal-dual gap → INFO)                         │              │    │
│  │                                                       ▼              │    │
│  │  L4: Adversarial Direction ──────────────────────────┤              │    │
│  │      (LLM debate: verify ↔ repair → Accept/Reject)    │              │    │
│  │                                                       ▼              │    │
│  │  L5: Constraint Presence Test ───────────────────────┘              │    │
│  │      (LLM-based CPT → WARNING/INFO)                                 │    │
│  │                                                                      │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                            │
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    DIAGNOSTIC-BASED REPAIR                           │    │
│  │                                                                      │    │
│  │  ERROR (L2 anomaly) ─────────────────────→ MUST fix                 │    │
│  │  WARNING (L4 accepted, L5 missing) ──────→ SHOULD fix               │    │
│  │  INFO (L2 no_effect, L3, L4 rejected) ───→ Reference only           │    │
│  │                                                                      │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                            │
│                                 ▼                                            │
│                          [Verified Code]                                     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Chain-of-Thought Generation** | 3-stage structured generation: Understand → Formalize → Synthesize |
| **L1 Execution & Solver** | Syntax, runtime, and solver status checks (FATAL → regeneration) |
| **L2 Anomaly Detection** | Bidirectional perturbation: if both ↑ and ↓ improve objective → ERROR |
| **L3 Dual Consistency** | Primal-dual gap verification (numerical tolerance: 1e-4) |
| **L4 Adversarial Direction** | LLM-based debate between verify and repair roles with Accept/Reject |
| **L5 Constraint Presence Test** | LLM-based verification of expected constraints in generated code |
| **Diagnostic-Based Repair** | Conservative strategy: only ERROR/WARNING trigger repair, INFO is reference only |

### Design Principles

1. **Universal Verification**: All checks work without ground truth or domain-specific rules
2. **Conservative Repair**: Only high-confidence issues (ERROR/WARNING) trigger repairs
3. **Robustness Guarantee**: L1 FATAL triggers regeneration; L2-L5 are diagnostic only
4. **LLM-Only Analysis**: No keyword-based heuristics; all semantic analysis uses LLM

---

## Installation

```bash
git clone https://github.com/junbolian/ReLoop.git
cd ReLoop
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- Gurobi >= 9.0 (with valid license)
- numpy
- (Optional) LLM API access for L5 CPT layer

```bash
# Verify installation
python -c "from reloop import ReLoopVerifier; print('OK')"
```

---

## Quick Start

```python
from reloop import run_reloop, DataExtractor

# Your LLM client (must implement generate(prompt, system=None) -> str)
llm_client = YourLLMClient()

# Extract structured data from natural language
extractor = DataExtractor(llm_client)
data = extractor.extract("Minimize cost with capacity 500 and demands [100, 150, 200]...")

# Run pipeline: Generate → Verify → Repair
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

## Reproducing Paper Results

### Run experiments on benchmark datasets

```python
from reloop import run_experiment

summary = run_experiment(
    dataset_path="data/RetailOpt-190.jsonl",
    llm_client=your_llm_client,
    output_dir="results",
    verbose=True
)

print(f"Detection Rate: {summary.detection_rate:.1%}")
print(f"False Positive Rate: {summary.false_positive_rate:.1%}")
print(f"Repair Success Rate: {summary.repair_success_rate:.1%}")
```

### Available Datasets

| Dataset | # Problems | Description |
|---------|------------|-------------|
| `RetailOpt-190.jsonl` | 190 | **Our benchmark** - Retail optimization scenarios |
| `IndustryOR_fixedV2.jsonl` | 100 | Industry OR problems with difficulty labels |
| `MAMO_EasyLP_fixed.jsonl` | 642 | Easy LP problems |
| `MAMO_ComplexLP_fixed.jsonl` | - | Complex LP problems |
| `NL4OPT.jsonl` | 245 | NL4OPT benchmark |
| `OptMATH_Bench_166.jsonl` | 166 | OptMATH benchmark |
| `OptMATH_Bench_193.jsonl` | 193 | OptMATH benchmark |
| `OptiBench.jsonl` | - | OptiBench problems |

---

## Project Structure

```
reloop/
├── reloop/                       # Core package
│   ├── __init__.py               # Public API exports
│   ├── param_utils.py            # Parameter extraction & perturbation
│   ├── executor.py               # Isolated subprocess execution
│   ├── verification.py           # 5-layer verification engine (L1-L3, L5)
│   ├── l4_adversarial.py         # L4 Adversarial Direction Analysis
│   ├── prompts.py                # LLM prompt templates
│   ├── generation.py             # Code generation
│   ├── repair.py                 # Diagnostic-based repair with L4 Accept/Reject
│   ├── pipeline.py               # Pipeline orchestration
│   ├── data_extraction.py        # NL → structured data extraction
│   └── experiment_runner.py      # Batch experiment runner
├── data/                         # Benchmark datasets (JSONL)
├── fig/                          # Architecture diagrams
│   └── Reloop_framework.pdf      # System architecture diagram
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project configuration
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `ReLoopVerifier` | 5-layer verification engine |
| `L4AdversarialVerifier` | L4 Adversarial Direction Analysis with LLM debate |
| `CodeGenerator` | Generate Gurobi code from problem description |
| `CodeRepairer` | Repair code based on diagnostics (with L4 Accept/Reject) |
| `ReLoopPipeline` | Complete generate→verify→repair pipeline |
| `DataExtractor` | Extract structured data from natural language |
| `ExperimentRunner` | Run batch experiments on datasets |

### Verification

```python
from reloop import ReLoopVerifier, verify_code

# Quick verification
report = verify_code(code, data, obj_sense="minimize")

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
    max_repair_iterations=3,        # L2-L5 repair attempts
    max_regeneration_attempts=3,    # L1 FATAL regeneration attempts
    max_l4_rejections=2,            # Max rejections per param before L4 → INFO
    enable_cpt=True,                # Enable L5 CPT layer
    enable_l4_adversarial=True,     # Enable L4 Adversarial Direction Analysis
    use_structured_generation=True  # Use 3-stage pipeline
)
result = pipeline.run(problem_description, data)

# Using convenience function
result = run_reloop(
    problem_description, data, llm_client,
    max_iterations=3,          # Repair iterations
    max_regenerations=3,       # Regeneration attempts
    use_structured_generation=True
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
summary = runner.run_dataset("data/NL4OPT.jsonl")

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

**Generation Approaches:**

| Approach | Description | Error Rate |
|----------|-------------|------------|
| Single-Stage | Direct problem → code (baseline) | - |
| 3-Stage CoT (single call) | All steps in one prompt | 2.17% |
| 3-Stage CoT (separate calls) | 3 API calls (loses context) | 10.85% |

**Recommended:** 3-Stage CoT with single API call (preserves reasoning chain)

---

## Schema-Only Visibility Design

A key architectural principle is that generated code uses **schema-only visibility**:

```
┌─────────────────────────────────────────────────────────────┐
│ Data Dict (external)                                         │
│   {"capacity": 500, "costs": [10,15,8], "demand": [100,150]} │
└─────────────────────────────────────────────────────────────┘
                          ↓
                   Schema Description (sent to LLM)
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ - capacity: int (scalar)                                     │
│ - costs: list[3] of int                                      │
│ - demand: list[3] of int                                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
                   Generated Code uses data["key"]
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ # Code accesses data at runtime                              │
│ m.addConstr(x <= data["capacity"])                           │
│ for i in range(len(data["costs"])): ...                      │
└─────────────────────────────────────────────────────────────┘
```

**Key Points:**
1. LLM sees schema (keys, types, dimensions) but NOT actual values
2. Code must use `data["key"]` to access runtime-injected data
3. The `data` dict is injected by executor at runtime
4. Generation prompts include schema; repair prompts also include schema to ensure consistency

**Big-M Guidelines (for indicator/logical constraints):**
- NEVER hardcode Big-M values like `M = 1e6`
- ALWAYS compute M dynamically: `M = sum(data["demand"]) * 1.5`
- Use problem-relevant parameters for M calculation

**Edge Case Handling:**
- Check array length: `if data["items"]: ...`
- Avoid division by zero: `max(value, 1e-6)`
- Use `.get()` for optional keys: `data.get("key", default)`

---

## 5-Layer Verification Architecture

| Layer | Name | Type | Description |
|-------|------|------|-------------|
| L1 | Execution & Solver | Blocking | Syntax, runtime, solver status → triggers regeneration on FATAL |
| L2 | Anomaly Detection | Diagnostic | Bidirectional perturbation: both-improve → ERROR |
| L3 | Dual Consistency | Diagnostic | Primal-dual gap (INFO level - likely numerical) |
| L4 | Adversarial Direction Analysis | Diagnostic | LLM-based direction verification with Accept/Reject |
| L5 | CPT | Enhancement | LLM-based constraint testing (WARNING/INFO) |

**Severity Levels (Conservative Repair Strategy):**

| Severity | Confidence | Source | Repair Action |
|----------|------------|--------|---------------|
| `FATAL` | 100% | L1 only | Triggers regeneration (up to 3 attempts) |
| `ERROR` | 99%+ | L2 anomaly | **MUST fix** |
| `WARNING` | 80%+ | L4 direction (accepted), L5 cpt_missing | **SHOULD fix** |
| `INFO` | <80% | L2 no_effect, L2 sensitivity, L3 duality, L4 (rejected/inconclusive) | **DO NOT fix** (reference only) |
| `PASS` | - | All layers | No action needed |

**Key Design Principle:**
- Only ERROR and WARNING trigger repair
- INFO is for reference only - likely normal optimization behavior (slack constraints, numerical artifacts)
- This prevents over-correction that was causing ReLoop to perform worse than baseline

### L2: Anomaly Detection (Bidirectional Perturbation)

L2 uses **bidirectional perturbation** to detect physically impossible behavior:

| Check | Severity | Method | Rationale |
|-------|----------|--------|-----------|
| **Anomaly** | **ERROR** | Both ↑ and ↓ improve objective | Physically impossible (99%+ confidence) |
| No Effect | INFO | Neither direction affects objective | Likely slack constraint (normal) |
| High Sensitivity | INFO | Extreme sensitivity to perturbation | Normal for well-optimized models |

```
Anomaly Detection Principle:
├── Perturb parameter UP (+20%) → measure objective change
├── Perturb parameter DOWN (-20%) → measure objective change
├── If BOTH directions IMPROVE objective → ERROR (impossible)
├── If one improves, one worsens → normal (monotonic)
├── If neither affects → INFO (slack constraint)
└── Works for ANY domain without keyword matching
```

### L4: Adversarial Direction Analysis (LLM-based)

L4 uses an **adversarial mechanism** where two LLM roles debate to converge on the correct analysis:

```
┌─────────────────────────────────────────────────────────────┐
│                 L4 Adversarial Flow                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LLM_verify ──→ "Parameter X should decrease objective"     │
│       │                                                     │
│       ▼                                                     │
│  LLM_repair ──→ Accept? ──→ YES ──→ WARNING (should fix)   │
│       │              │                                      │
│       │              └──→ NO (Reject) ──→ Re-analyze        │
│       │                        │                            │
│       │                        ▼                            │
│       │              LLM_verify (with rejection feedback)   │
│       │                        │                            │
│       │                        ▼                            │
│       │              [Repeat until Accept or max rejections]│
│       │                                                     │
│       └──→ Max rejections reached ──→ INFO (inconclusive)  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Exit Conditions (forces output):**
1. `all_pass`: No violations found → output
2. `all_rejected_others_pass`: All L4 rejected + L2/L3/L5 PASS → output (INFO level)
3. `max_rejections`: Max rejections per param reached (default 2) → downgrade to INFO → output
4. `max_iterations`: Reached max L4 iterations (default 3) → output
5. `accepted_fixed`: Some accepted, code fixed → re-verify and continue

**Key Parameters:**
- `max_l4_rejections`: Max times a param can be rejected before downgrade (default: 2)
- `max_l4_iterations`: Max L4 loop iterations (default: 3)

> **Design Principle:** The adversarial mechanism allows two LLM perspectives to debate.
> This is more reliable than single-LLM analysis because errors get caught by the other role.
> Keyword-based direction verification has been completely removed.

**Robustness Guarantee:**
- L1 `FATAL` triggers regeneration, not immediate termination
- L2-L5 are diagnostic only: never block output
- L4 loop always exits with output (one of the exit conditions will be met)
- False positives don't affect result values (objective/solution always returned if L1 passes)
- INFO-level issues do NOT trigger repair (prevents over-correction)

### L5: CPT (Constraint Presence Testing)

L5 uses **LLM-based constraint extraction** to identify expected constraints, then tests if they are present in the generated code:

```
┌─────────────────────────────────────────────────────────────┐
│                 L5 CPT Flow                                  │
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
[L5] CPT
  Extracted 3 candidates:
  ├─ [MISSING] minimum protein requirement - 0.0% change ⚠️
  ├─ [UNCERTAIN] minimum carbs requirement - 21.1% change
  └─ [PRESENT] minimum calories requirement - 38.0% change ✅
```

---

## Data Format

JSONL format, one problem per line:

```json
{"en_question": "Problem description...", "en_answer": "123.45", "difficulty": "Easy", "id": 1}
```

---

## Citation

```bibtex
@inproceedings{lian2026reloop,
  title     = {ReLoop: Detecting Silent Failures in LLM-Generated Optimization Code via Behavioral Verification},
  author    = {Lian, Junbo Jacob and Sun, Yujun and Chen, Huiling and Zhang, Chaoyu and Teo, Chung-Piaw},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
