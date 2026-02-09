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

ReLoop is a behavioral verification framework that detects **silent failures** in LLM-generated optimization code—code that executes successfully but produces incorrect results. Our key insight is that correct optimization models satisfy fundamental mathematical invariants (execution correctness, direction consistency, constraint presence) that can be tested without ground truth.

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
│  │                    4-LAYER VERIFICATION                              │    │
│  │                                                                      │    │
│  │  L1: Execution & Solver + Duality Check ───────────┐              │    │
│  │      (Blocking: FATAL → Regenerate up to 3x)        │              │    │
│  │      (Duality gap → INFO, add-on diagnostic)        │              │    │
│  │                                                       ▼              │    │
│  │  L2: Direction Consistency Analysis ───────────────┤              │    │
│  │      (LLM debate: verify ↔ repair → Accept/Reject)  │              │    │
│  │                                                       ▼              │    │
│  │  L3: Constraint Presence Test (CPT) ───────────────┤              │    │
│  │      (LLM-based CPT → WARNING/INFO)                  │              │    │
│  │                                                       ▼              │    │
│  │  L4: Specification Compliance Check ───────────────┘              │    │
│  │      (White-box code review → ERROR/WARNING)                       │    │
│  │                                                                      │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                            │
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    DIAGNOSTIC-BASED REPAIR                           │    │
│  │                                                                      │    │
│  │  ERROR (L2 accepted, L4 spec fail) ─────→ MUST fix                  │    │
│  │  WARNING (L3 cpt_missing) ──────────────→ SHOULD fix                │    │
│  │  INFO (L1 duality, L2 rejected) ────────→ Reference only            │    │
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
| **Chain-of-Thought Generation** | 3-stage structured generation: Understand → Formalize (with variable type reasoning) → Synthesize |
| **L1 Execution & Solver** | Syntax, runtime, and solver status checks with IIS/unbounded diagnostics (FATAL → regeneration) + duality check (INFO) |
| **L2 Direction Consistency** | LLM-based adversarial debate between verify and repair roles with Accept/Reject |
| **L3 Constraint Presence Test** | LLM-based verification of expected constraints in generated code (CPT) |
| **L4 Specification Compliance** | White-box code review: 6-category focused extraction + context-aware 3-step verification |
| **Unified Diagnostic Schema** | All layers output `Diagnostic` objects; unified `build_repair_prompt()` assembles repair prompts |
| **Diagnostic-Based Repair** | Conservative strategy: only ERROR/WARNING trigger repair, INFO is reference only |
| **Repair Safety Guardrails** | Prevents repair LLM from modifying input data or introducing dangerous operations |

### Design Principles

1. **Universal Verification**: All checks work without ground truth or domain-specific rules
2. **Conservative Repair**: Only high-confidence issues (ERROR/WARNING) trigger repairs
3. **Robustness Guarantee**: L1 FATAL triggers regeneration; L2-L3 are diagnostic only
4. **LLM-Only Analysis**: No keyword-based heuristics; all semantic analysis uses LLM
5. **Repair Safety**: Repair code is validated before execution; data variable cannot be reassigned
6. **Dual Verification**: Black-box behavioral testing (L1-L3) + white-box code review (L4)

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
- (Optional) LLM API access for L2/L3 layers

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
│   ├── param_utils.py            # Parameter extraction & perturbation (data-dict)
│   ├── perturbation.py           # Source-code level perturbation (AST-based) + mode detection
│   ├── executor.py               # Isolated subprocess execution with IIS/unbounded diagnostics
│   ├── verification.py           # 4-layer verification engine (L1-L3) + unified Diagnostic schema
│   ├── l2_direction.py           # L2 Direction Consistency Analysis (adversarial LLM debate)
│   ├── specification.py          # L4 Specification Compliance Checking (white-box code review)
│   ├── prompts.py                # LLM prompt templates
│   ├── generation.py             # Code generation
│   ├── repair.py                 # Diagnostic-based repair with L2 Accept/Reject
│   ├── repair_safety.py          # Repair safety guardrails (data protection, import blocking)
│   ├── pipeline.py               # Pipeline orchestration
│   ├── data_extraction.py        # NL → structured data extraction
│   └── experiment_runner.py      # Batch experiment runner
├── tests/                        # Test suite
│   ├── test_perturbation.py      # Unit tests for perturbation module
│   ├── test_e2e_perturbation.py  # E2E tests for L2 perturbation modes
│   ├── test_repair_safety.py     # Unit tests for repair safety guardrails
│   └── test_specification.py    # Unit tests for L4 specification compliance
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
| `ReLoopVerifier` | 4-layer verification engine |
| `Diagnostic` | Unified diagnostic schema for all layers |
| `L2DirectionVerifier` | L2 Direction Consistency Analysis with LLM adversarial debate |
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
    max_l4_rejections=2,            # Max rejections per param before L2 → INFO
    enable_cpt=True,                # Enable L3 CPT layer
    enable_l4_adversarial=True,     # Enable L2 Direction Consistency Analysis
    enable_l4_specification=False,  # Enable L4 Specification Compliance Check
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

**Formalization Stage — Variable Type Reasoning:**

The formalization step explicitly prompts the LLM to determine variable types (CONTINUOUS, INTEGER, or BINARY) for each decision variable by analyzing physical context. For example, "number of trucks" implies integer, while "volume of liquid" implies continuous. This reduces downstream errors that L4 would otherwise need to catch.

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

## 4-Layer Verification Architecture

| Layer | Name | Type | Description |
|-------|------|------|-------------|
| L1 | Execution & Solver + Duality | Blocking | Syntax, runtime, solver status with IIS/unbounded diagnostics → triggers regeneration on FATAL; duality check as add-on diagnostic (INFO) |
| L2 | Direction Consistency Analysis | Diagnostic | LLM-based adversarial direction verification with Accept/Reject |
| L3 | Constraint Presence Test (CPT) | Enhancement | LLM-based constraint testing (WARNING/INFO) |
| L4 | Specification Compliance Check | Semantic | White-box code review: 6-category focused extraction with L1-L3 context-aware 3-step verification (ERROR/WARNING) |

**Severity Levels (Conservative Repair Strategy):**

| Severity | Confidence | Source | Repair Action |
|----------|------------|--------|---------------|
| `FATAL` | 100% | L1 only | Triggers regeneration (up to 3 attempts) |
| `ERROR` | 99%+ | L2 direction (accepted), L4 spec violation | **MUST fix** |
| `WARNING` | 80%+ | L3 cpt_missing, L4 spec uncertain | **SHOULD fix** |
| `INFO` | <80% | L1 duality, L2 (rejected/inconclusive) | **DO NOT fix** (reference only) |
| `PASS` | - | All layers | No action needed |

**Key Design Principle:**
- Only ERROR and WARNING trigger repair
- INFO is for reference only - likely normal optimization behavior (slack constraints, numerical artifacts)
- This prevents over-correction that was causing ReLoop to perform worse than baseline

### Perturbation Modes (Source-Code vs Data-Dict)

L2 and L3 perform parameter perturbation to test model behavior. ReLoop supports two perturbation strategies with automatic detection:

| Mode | Detection Criterion | Perturbation Strategy | Typical Datasets |
|------|---------------------|-----------------------|------------------|
| `data_dict` | Code uses `data["key"]` pattern | Perturb the data dict values (existing) | RetailOpt-190 |
| `source_code` | Code hardcodes numeric values, no `data[` access | Perturb via AST-based code rewriting | IndustryOR, MAMO |
| `hybrid` | Both patterns present | Try data-dict first; fallback to source-code if no effect | Mixed |

**Auto-Detection:** `detect_perturbation_mode(code, data)` inspects the generated code for `data[` access patterns and counts hardcoded numeric assignments. The result determines which perturbation strategy L2-L3 use.

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

**Hybrid Fallback:** For `hybrid` mode, L2-L3 first try data-dict perturbation. If the perturbed objective is unchanged (parameter hardcoded in code rather than read from data), the system falls back to AST-based source-code perturbation.

**Key Design Constraint:** The detection/judgment logic (thresholds, direction analysis, CPT thresholds) is completely unchanged. Only the perturbation mechanism is extended.

### L2: Direction Consistency Analysis (LLM-based Adversarial)

L2 uses an **adversarial mechanism** where two LLM roles debate to converge on the correct analysis:

```
┌─────────────────────────────────────────────────────────────┐
│                 L2 Adversarial Flow                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LLM_verify ──→ "Parameter X should decrease objective"     │
│       │                                                     │
│       ▼                                                     │
│  LLM_repair ──→ Accept? ──→ YES ──→ ERROR (must fix)       │
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
2. `all_rejected_others_pass`: All L2 rejected + L1/L3 PASS → output (INFO level)
3. `max_rejections`: Max rejections per param reached (default 2) → downgrade to INFO → output
4. `max_iterations`: Reached max L2 iterations (default 3) → output
5. `accepted_fixed`: Some accepted, code fixed → re-verify and continue

**Key Parameters:**
- `max_l4_rejections`: Max times a param can be rejected before downgrade (default: 2)
- `max_l2_iterations`: Max L2 loop iterations (default: 3)

> **Design Principle:** The adversarial mechanism allows two LLM perspectives to debate.
> This is more reliable than single-LLM analysis because errors get caught by the other role.
> Keyword-based direction verification has been completely removed.

**Robustness Guarantee:**
- L1 `FATAL` triggers regeneration, not immediate termination
- L2-L4 are diagnostic only: never block output
- L2 loop always exits with output (one of the exit conditions will be met)
- L4 FAIL may not be true positive; repair budget N=3 naturally prevents infinite loops
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

### L4: Specification Compliance Checking (White-Box)

L4 takes a completely different approach from L1-L3. While L1-L3 perform **black-box behavioral testing** (perturbing inputs and observing outputs), L4 performs **white-box code review** — directly reading the code and comparing it against the problem description.

```
┌─────────────────────────────────────────────────────────────┐
│         L4 Specification Compliance Flow (Enhanced)          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: EXTRACT (6 Category-Specific Prompts)             │
│  Problem Description                                        │
│       ├──→ VARIABLE_TYPE prompt ──→ [specs]                 │
│       ├──→ VARIABLE_SCOPE prompt ──→ [specs]                │
│       ├──→ CONSTRAINT prompt ──→ [specs]                    │
│       ├──→ OBJECTIVE prompt ──→ [specs]                     │
│       ├──→ LOGICAL_CONDITION prompt ──→ [specs]             │
│       └──→ DATA_MAPPING prompt ──→ [specs]                  │
│                              │                              │
│                              ▼                              │
│        Merge + Deduplicate → Unified Checklist              │
│                              │                              │
│  Phase 2: VERIFY (3-Step Reasoning + L1-L3 Context)         │
│                              ▼                              │
│  Checklist + Code + L1-L3 Diagnostics                       │
│       ──→ For each spec:                                    │
│           1. STATE what requirement demands                  │
│           2. FIND what code actually does (quote lines)      │
│           3. COMPARE: match or mismatch?                     │
│                              │                              │
│                              ▼                              │
│        Per-item Verdict:                                    │
│              PASS → No action                               │
│              FAIL → ERROR (triggers repair)                  │
│              UNCERTAIN → WARNING (reference only)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Specification Categories:**

| Category | What it checks | Example |
|----------|---------------|---------|
| `VARIABLE_TYPE` | Integer vs continuous vs binary | "number of trucks" → integer |
| `VARIABLE_SCOPE` | Decision variable index ranges | "any region can ship to any region" → all pairs |
| `CONSTRAINT` | Required constraints present | "protein >= minimum requirement" |
| `OBJECTIVE` | Correct objective formulation | "minimize total transportation cost" |
| `LOGICAL_CONDITION` | Conditional constraints | "if demand > supply, penalize" |
| `DATA_MAPPING` | Parameter interpretation | "return rate 0.7 means total = 1.7x" |

**Key Design Decisions:**
- Phase 1 uses **6 category-specific prompts** (not one generic prompt) for higher recall
- Phase 2 uses **3-step reasoning** (STATE → FIND → COMPARE) to prevent skipping code inspection
- Phase 2 receives **L1-L3 execution context** (objective value, diagnostics) for cross-layer insight
- Results are deduplicated across categories (exact-match on checkable_criterion)
- FAIL → ERROR (triggers_repair=True), UNCERTAIN → WARNING (triggers_repair=False)
- `max_specs=25` limits the number of specification items to control LLM cost
- Temperature=0 for deterministic output

**What L4 catches that L1-L3 cannot:**
- Variable type errors (continuous variables that should be integer)
- Model scope errors (restricted routing instead of general transshipment)
- Coefficient interpretation errors (rate vs total return)

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
    layer: str          # "L1", "L2", "L3", "L4"
    issue_type: str     # "INFEASIBLE", "DIRECTION_VIOLATION", "MISSING_CONSTRAINT", "SPEC_VIOLATION", etc.
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
| Data reassignment | AST + Regex | `data = {...}`, `data = json.loads(...)` |
| Data mutation | AST + Regex | `data["key"] = value`, `data["key"] += value` |
| Dangerous imports | Regex | `import os`, `import subprocess` |

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
