# ReLoop: Reliable LLM-based Optimization Modeling via Sensitivity-Based Behavioral Verification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

---

## Overview

**ReLoop** is a framework for improving the reliability of LLM-generated optimization code through:

1. **Structured Generation**: Multi-step prompting that simulates expert engineer reasoning
2. **Behavioral Verification**: 7-layer sensitivity-based testing to detect silent failures
3. **Guided Repair**: Diagnosis-driven code correction loop

### The Problem We Solve

```
Traditional Approach:
  LLM → Code → Executes? → ✓ Done

The Problem:
  Code may EXECUTE SUCCESSFULLY but produce WRONG RESULTS
  This is called "Silent Failure" - the most dangerous type of bug

ReLoop's Solution:
  LLM → Code → Executes? → Behavior Correct? → ✓ Done
                              ↓ No
                           Diagnose → Repair → Retry
```

### Key Insight

> "We don't check if the code structure is correct,
> we check if the model **behavior** makes sense."

If `demand ↑ 20%` but `cost ↓`, something is wrong with the demand constraint.

### ReLoop vs Training-time Methods

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  SIRL/ORLM/LLMOPT = Train a better model (Training-time)        │
│                                                                 │
│  ReLoop = Verify and repair ANY model's output (Inference-time) │
│                                                                 │
│  Different levels — ReLoop is a "safety net" for all methods!   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

   Any Base Model ──→ ReLoop Verification ──→ More Reliable Output

   Even well-trained SIRL/ORLM can have Silent Failures
   ReLoop catches these errors at inference time
```

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ReLoop = Think like an Engineer + Test like a QA + Fix like an Expert
│                                                                 │
│           Structured        Behavioral         Guided           │
│           Generation    +   Verification   +   Repair           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ReLoop Pipeline                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT                                                          │
│  ├── Business Narrative (natural language description)         │
│  └── Data Schema (structure only, NOT full data values)        │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  STRUCTURED GENERATION (simulate expert thinking)              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Step 0: Data Profile (automatic, no LLM)                │   │
│  │   └── Extract dimensions, features, parameter roles     │   │
│  │                                                         │   │
│  │ Step 1: Problem Understanding (LLM)                     │   │
│  │   └── Output: objective, decisions, constraints (JSON)  │   │
│  │                                                         │   │
│  │ Step 2: Mathematical Specification (LLM)                │   │
│  │   └── Output: sets, variables, formulas (JSON)          │   │
│  │                                                         │   │
│  │ Step 3: Code Generation (LLM)                           │   │
│  │   └── Output: executable GurobiPy code                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  BEHAVIORAL VERIFICATION (7-layer system)                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ══════════ BASIC (L1-L2) ══════════                     │   │
│  │ Layer 1: Execution [MANDATORY]                          │   │
│  │   └── Code must run without errors                      │   │
│  │ Layer 2: Feasibility [LENIENT]                          │   │
│  │   └── OPTIMAL? TIME_LIMIT with obj OK                   │   │
│  │                                                         │   │
│  │ ══════════ STRUCTURE (L3) ══════════                    │   │
│  │ Layer 3: Code Structure (AST) [UNIVERSAL, fast]         │   │
│  │   └── Objective? Variables? Constraints? Boundaries?    │   │
│  │   └── Sales availability? (sales <= I)                  │   │
│  │                                                         │   │
│  │ ══════════ SEMANTIC (L4-L6) ══════════                  │   │
│  │ Layer 4: Monotonicity (Universal - No Domain Knowledge) │   │
│  │   └── Does each parameter affect objective?             │   │
│  │ Layer 5: Sensitivity (Role-Based)                       │   │
│  │   └── demand↑ → cost↑? capacity↓ → cost↑?               │   │
│  │ Layer 6: Boundary                                       │   │
│  │   └── param=0 behavior? param=∞ behavior?               │   │
│  │                                                         │   │
│  │ ══════════ DOMAIN (L7) ══════════                       │   │
│  │ Layer 7: Domain Probes [OPTIONAL, Retail-specific]      │   │
│  │   └── Enable: enable_layer7=True                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│  Note: Always reports objective value regardless of layer      │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  GUIDED REPAIR (if verification fails)                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Step 5: Targeted Repair (LLM)                           │   │
│  │   └── Input: code + layer context + diagnosis           │   │
│  │   └── Preservation rules: DON'T break working parts     │   │
│  │   └── Constraint patterns: suggest fix based on param   │   │
│  │   └── Output: fixed code (minimal change)               │   │
│  │   └── Early stop: if no progress for 2 iterations       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  OUTPUT                                                         │
│  ├── Verified code                                             │
│  ├── Verification report                                       │
│  └── Execution trace                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3-Step Structured Generation

ReLoop uses a 3-step structured generation process that preserves problem context throughout:

```
STEP 1: Problem Understanding
┌─────────────────────────────────────────────────────────────────┐
│ Input:  Business narrative + Data schema                        │
│ Output: JSON with objective, decisions, constraints             │
│                                                                 │
│ Extracts key components from natural language:                  │
│ - Objective (minimize/maximize what?)                           │
│ - Decision variables (what are we deciding?)                    │
│ - Constraints (what limits the decisions?)                      │
│ - Key relationships (how do components interact?)               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
STEP 2: Mathematical Specification
┌─────────────────────────────────────────────────────────────────┐
│ Input:  Step 1 output + Data schema                             │
│ Output: JSON with sets, parameters, variables, formulas         │
│                                                                 │
│ Converts understanding to formal math:                          │
│ - Define index sets (T, P, L, etc.)                             │
│ - Define parameters (demand, capacity, costs)                   │
│ - Define variables (I, Q, S, W, L, etc.)                        │
│ - Write constraint formulas with proper indexing                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
STEP 3: Code Generation
┌─────────────────────────────────────────────────────────────────┐
│ Input:  Step 2 output + Data access + ORIGINAL PROBLEM CONTEXT  │
│ Output: Executable GurobiPy code                                │
│                                                                 │
│ CRITICAL: Step 3 receives the original problem context to       │
│ ensure key equations (especially indexing) are preserved.       │
│                                                                 │
│ Common indexing errors this prevents:                           │
│ - sales[p,l,t,r] vs sales[p,l,t,r+1] in aging constraints       │
│ - I[p,l,t] vs I[p,l,t-1] in balance constraints                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Design: Step 3 preserves original context**

The original business narrative (with exact equations) is passed to Step 3 as reference.
This prevents information loss through the pipeline and ensures critical equations are
implemented exactly as specified.

---

## 7-Layer Verification System

### Layer 1: Execution Verification [MANDATORY]

**Question:** Does the code run without errors?

**Status:** Must pass - code execution is fundamental.

```python
# What we check:
✓ No syntax errors (SyntaxError)
✓ No runtime errors (NameError, TypeError, etc.)
✓ No import errors (ModuleNotFoundError)
✓ Model object created (m or model variable)
✓ Solver called successfully (m.optimize())

# Common failures detected:
× Missing imports
× Wrong variable names
× Data access errors (KeyError, IndexError)
× Network edges not converted to tuples
× Gurobi license issues
```

### Layer 2: Feasibility Verification [LENIENT]

**Question:** Does the model have a valid solution?

**Status:** Lenient - TIME_LIMIT with objective is acceptable.

```python
# Status checks:
Status = 2 (OPTIMAL)     → ✓ Good
Status = 9 (TIME_LIMIT)  → ✓ OK if objective obtained
Status = 3 (INFEASIBLE)  → ✗ Constraints contradictory
Status = 5 (UNBOUNDED)   → ✗ Missing constraints
Status = 12 (NUMERIC)    → ✗ Coefficient scaling issues

# Additional checks:
⚠ Objective = 0 → Missing costs? Free resources?
⚠ Very large gap → MIP not solved well
```

**Note:** TIME_LIMIT (status=9) is accepted if the solver found a feasible solution. This allows complex models to proceed even without optimal solution.

**Common fixes:**
- INFEASIBLE → Add slack/lost sales variable
- UNBOUNDED → Check objective direction, add bounds
- TIME_LIMIT with no objective → Simplify model or increase time limit
- NUMERIC → Scale coefficients to similar magnitude

### Verification Progression (L3-L7 Run Independently)

```
IMPORTANT: Layers 3-7 run INDEPENDENTLY of each other.
═══════════════════════════════════════════════════════════════════

Execution Flow:
  L1 (Execution)    → MUST PASS → stops if fails
  L2 (Feasibility)  → MUST PASS → stops if fails
  L3 (Code AST)     → Runs first (fast static analysis, no data leakage)
  L4 (Monotonicity) → Runs regardless
  L5 (Sensitivity)  → Runs regardless (even if L4 has failures)
  L6 (Boundary)     → Runs regardless (even if L4/L5 have failures)
  L7 (Domain)       → Runs if enabled (even if L3-L6 have failures)

Why this design:
  - L3 (AST) is fast static analysis - run before expensive runtime tests
  - L4 failure may indicate SLACK constraints, not missing constraints
  - L5/L6/L7 provide different diagnostic information
  - All layers contribute to understanding model behavior
  - Objective value is ALWAYS reported if available

Layer Pass Definition:
  - L3 passes: ALL code structure checks pass
  - L4 passes: ALL monotonicity tests pass (no "no effect" detected)
  - L5 passes: ALL direction tests match expectation
  - L6 passes: ALL boundary tests behave correctly
  - L7 passes: ALL domain probes pass

Note: Layer failures do NOT always indicate wrong models:
  - Slack constraints: L4 may fail but model is correct
  - Alternative formulations: Different but equivalent models
  - Final correctness: Compare objective to ground truth (< 1% gap)
```

### Layer 3: Code Structure Verification (AST-based, Universal)

**Question:** Does the code have proper structure?

```
═══════════════════════════════════════════════════════════════════
🔑 FAST STATIC ANALYSIS - RUN BEFORE EXPENSIVE RUNTIME TESTS
═══════════════════════════════════════════════════════════════════

Principle:
  Analyze code structure WITHOUT running it.
  Does NOT leak data - only examines variable names, patterns, formulas.

Checks:
  ┌─────────────────────────────────────────────────────────────┐
  │ Check                        │ What it detects              │
  ├──────────────────────────────┼──────────────────────────────┤
  │ Objective function exists    │ Missing m.setObjective()     │
  ├──────────────────────────────┼──────────────────────────────┤
  │ Holding cost pattern         │ I vs I-y formula errors      │
  ├──────────────────────────────┼──────────────────────────────┤
  │ Loop index boundaries        │ t-1 at t=1 boundary issues   │
  ├──────────────────────────────┼──────────────────────────────┤
  │ Variable declarations        │ Missing m.addVar() calls     │
  ├──────────────────────────────┼──────────────────────────────┤
  │ Constraint additions         │ Missing m.addConstr() calls  │
  ├──────────────────────────────┼──────────────────────────────┤
  │ Sales availability           │ Missing sales <= I constraint│
  └─────────────────────────────────────────────────────────────┘

Why this is universal:
  - Pure code structure analysis
  - No data values examined
  - No execution required
  - Catches common LLM errors early
```

### Layer 4: Monotonicity Verification (Universal)

**Question:** Does each parameter affect the objective?

```
═══════════════════════════════════════════════════════════════════
🔑 THIS IS THE KEY UNIVERSAL CHECK - NO DOMAIN KNOWLEDGE NEEDED
═══════════════════════════════════════════════════════════════════

Principle:
  If a parameter appears in a constraint, perturbing it should
  change the objective. "No effect" indicates the constraint
  is likely MISSING from the model.

Smart Parameter Filtering (skip parameters that shouldn't affect objective):
  - Zero values: lead_time=0, return_rate=0 (inactive constraints)
  - Big M values: capacity=99999 (won't bind, effectively infinite)
  - Not found: parameter doesn't exist in data

Test procedure for each TESTABLE numeric parameter:
  1. Run baseline              → obj_base
  2. Perturb parameter +20%    → obj_up
  3. Perturb parameter -20%    → obj_down

  Analysis:
  ┌─────────────────────────────────────────────────────────────┐
  │ Case                          │ Interpretation             │
  ├───────────────────────────────┼────────────────────────────┤
  │ obj_up ≈ obj_base AND         │ ⚠️ CRITICAL: Parameter has │
  │ obj_down ≈ obj_base           │ NO EFFECT - constraint     │
  │                               │ likely MISSING!            │
  ├───────────────────────────────┼────────────────────────────┤
  │ obj_up and obj_down change    │ ✓ Parameter affects model  │
  │ in opposite directions        │ (monotonic - expected)     │
  ├───────────────────────────────┼────────────────────────────┤
  │ obj_up and obj_down change    │ ⚠️ Non-monotonic behavior  │
  │ in same direction             │ (unusual, investigate)     │
  └─────────────────────────────────────────────────────────────┘

Why this works universally:
  - Pure mathematical property
  - No need to know what "demand" or "capacity" means
  - Applies to LP, MILP, NLP - any optimization problem
  - Simple principle: used parameters must have effect
```

### Layer 5: Sensitivity Verification (Role-Based)

**Question:** Does the model behave correctly based on parameter semantics?

```
Parameter Role Taxonomy:
═══════════════════════════════════════════════════════════════════

┌───────────────┬─────────────────────────────────────────────────┐
│ REQUIREMENT   │ Things that must be satisfied (demand, orders)  │
│ role          │                                                 │
├───────────────┼─────────────────────────────────────────────────┤
│ Keywords      │ demand, order, request, need, target, rhs,      │
│               │ requirement, quota, goal, customer              │
├───────────────┼─────────────────────────────────────────────────┤
│ Test          │ Increase by 20%                                 │
│ Expected      │ Objective ↑ or INFEASIBLE (harder to satisfy)   │
├───────────────┼─────────────────────────────────────────────────┤
│ If violated   │ Demand constraint missing or wrong direction    │
└───────────────┴─────────────────────────────────────────────────┘

┌───────────────┬─────────────────────────────────────────────────┐
│ CAPACITY      │ Upper bounds on resources                       │
│ role          │                                                 │
├───────────────┼─────────────────────────────────────────────────┤
│ Keywords      │ capacity, cap, limit, max, budget, supply,      │
│               │ available, resource, ub, upper                  │
├───────────────┼─────────────────────────────────────────────────┤
│ Test          │ Decrease by 20%                                 │
│ Expected      │ Objective ↑ or INFEASIBLE (tighter constraint)  │
├───────────────┼─────────────────────────────────────────────────┤
│ If violated   │ Capacity constraint missing or not enforced     │
└───────────────┴─────────────────────────────────────────────────┘

┌───────────────┬─────────────────────────────────────────────────┐
│ COST          │ Cost coefficients in objective                  │
│ role          │                                                 │
├───────────────┼─────────────────────────────────────────────────┤
│ Keywords      │ cost, price, penalty, fee, expense, purchasing, │
│               │ holding, waste, lost_sales, c_, coef            │
├───────────────┼─────────────────────────────────────────────────┤
│ Test          │ Increase by 30%                                 │
│ Expected      │ Objective ↑ (for minimization)                  │
├───────────────┼─────────────────────────────────────────────────┤
│ If violated   │ Cost term missing from objective function       │
└───────────────┴─────────────────────────────────────────────────┘

Role inference:
  1. Match parameter name against keywords
  2. If no match and LLM available, ask LLM to classify
  3. If still unknown, skip role-based test (Layer 4 still runs)
```

### Layer 6: Boundary Verification (Enhanced)

**Question:** Does the model handle extreme values and structural boundaries correctly?

```
═══════════════════════════════════════════════════════════════════
Layer 6 now includes THREE types of tests (all UNIVERSAL):
═══════════════════════════════════════════════════════════════════

6.1: CAPACITY = 0 BOUNDARY (existing)
┌────────────────────┬────────────────────────────────────────────┐
│ TEST               │ EXPECTED BEHAVIOR                          │
├────────────────────┼────────────────────────────────────────────┤
│ capacity = 0       │ INFEASIBLE or very high objective          │
│                    │ (If objective stays low → constraint       │
│                    │  is not enforced!)                         │
└────────────────────┴────────────────────────────────────────────┘

6.2: STRUCTURAL BOUNDARY - periods=1 (NEW)
┌────────────────────────────────────────────────────────────────┐
│ KEY INSIGHT: Multi-period models should degrade gracefully     │
│ to single period. If code crashes → t-1 or t+1 indexing bug.   │
│                                                                │
│ TEST: Set periods = 1                                          │
│ EXPECTED: Code runs without error (OPTIMAL or INFEASIBLE OK)   │
│ FAILURE: Code crashes → Check boundary conditions like:        │
│   • I[t-1] when t=1 (no previous period)                       │
│   • I[t+1] when t=T (no next period)                           │
│   • for t in range(T-1) when T=1 (empty range OK)              │
└────────────────────────────────────────────────────────────────┘

6.3: DIFFERENTIAL VERIFICATION (NEW)
┌────────────────────────────────────────────────────────────────┐
│ KEY INSIGHT: capacity↓ and requirement↑ should BOTH make the   │
│ problem harder (increase cost for minimize, decrease for max). │
│                                                                │
│ TEST: Compare effects of:                                      │
│   • capacity × 0.8  (tighten supply)                           │
│   • requirement × 1.2  (increase demand)                       │
│                                                                │
│ EXPECTED (minimize): Both should increase objective            │
│ FAILURE: Effects go opposite directions → constraint logic bug │
│                                                                │
│ WHY UNIVERSAL: Uses same role inference as L5 (capacity,       │
│ requirement keywords), no domain-specific knowledge needed.    │
└────────────────────────────────────────────────────────────────┘

Why these boundary tests matter:
  - 6.1: Zero values reveal missing constraints
  - 6.2: Single-period tests expose indexing errors at boundaries
  - 6.3: Differential tests catch constraint direction errors
```

### Layer 7: Domain-Specific Probes [OPTIONAL, Retail-specific]

**Question:** Are retail-specific patterns implemented correctly?

**Status:** Optional - Enable with `enable_layer7=True` in verifier.

```
═══════════════════════════════════════════════════════════════════
IMPORTANT: Layer 7 is OPTIONAL and RETAIL-SPECIFIC.
  - Enable:  verifier.verify(..., enable_layer7=True)
  - Default: Disabled (enable_layer7=False)
  - Purpose: Additional retail-specific constraint checks
  - Scope:   Only for RetailOpt-190 dataset

Layers 1-6 are UNIVERSAL and sufficient for MAMO, NL4OPT, etc.
═══════════════════════════════════════════════════════════════════

PROBE 1: Lost Sales Variable (implemented)
┌─────────────────────────────────────────────────────────────────┐
│ Problem:                                                        │
│   Missing L[p,l,t] variable as slack in demand constraint       │
│                                                                 │
│ Test: Set demand >> production_cap (10x)                        │
│ Expected: Model stays OPTIMAL (lost sales absorb excess demand) │
│ Failure: INFEASIBLE → missing lost sales slack variable         │
└─────────────────────────────────────────────────────────────────┘

PROBE 2: Shelf Life Structure (implemented, NEW)
┌─────────────────────────────────────────────────────────────────┐
│ KEY INSIGHT: With shelf_life=1, cost should INCREASE            │
│ (all inventory expires after 1 period → more waste)             │
│                                                                 │
│ Test: Set all shelf_life = 1                                    │
│ Expected: Objective ≥ baseline (harder problem)                 │
│ Failure: Objective drops → aging constraint likely wrong        │
│                                                                 │
│ WHY THIS WORKS: Shorter shelf life = more waste = higher cost   │
│ If cost DECREASES, the aging logic is probably broken.          │
└─────────────────────────────────────────────────────────────────┘

PROBE 3: Substitution Structure (implemented, NEW)
┌─────────────────────────────────────────────────────────────────┐
│ KEY INSIGHT: If sub_edge [A,B] exists and A capacity=0,         │
│ model should still be feasible (B can substitute for A)         │
│                                                                 │
│ Test: Set production_cap[A] = 0 for first substitution edge     │
│ Expected: OPTIMAL (B satisfies A's demand via substitution)     │
│ Failure: INFEASIBLE → substitution not implemented              │
└─────────────────────────────────────────────────────────────────┘

PROBE 4 (future): Initialization (t=1)
┌─────────────────────────────────────────────────────────────────┐
│ Problem:                                                        │
│   Without I[p,l,1,a] = 0 for a < shelf_life[p], the model can  │
│   "use" phantom inventory from older age buckets at t=1.        │
│                                                                 │
│ Symptom: Objective ≈ 0 even when no production is possible      │
│ Test: Set all production_cap = 0, check if objective is low     │
│ Fix: Add I[p,l,1,a] == 0 for all a < shelf_life[p]              │
└─────────────────────────────────────────────────────────────────┘

PROBE 5 (future): Holding Cost Formula
┌─────────────────────────────────────────────────────────────────┐
│ Problem:                                                        │
│   Using I[p,l,t,a] for holding cost instead of                 │
│   (I[p,l,t,a] - y[p,l,t,a])                                    │
│   This charges holding cost on sold inventory (wrong!)          │
│                                                                 │
│ Symptom: Objective 3-5x higher than expected                    │
│ Test: Set demand = production_cap, high holding cost            │
│ Fix: Change holding cost to (I[p,l,t,a] - y[p,l,t,a])          │
└─────────────────────────────────────────────────────────────────┘

PROBE 6 (future): Lost Sales Variable
┌─────────────────────────────────────────────────────────────────┐
│ Problem:                                                        │
│   Missing L[p,l,t] variable as slack in demand constraint       │
│                                                                 │
│ Symptom: INFEASIBLE when demand > supply                        │
│ Test: Set demand >> production_cap                              │
│ Fix: Add L[p,l,t] >= 0 as slack in demand constraint            │
└─────────────────────────────────────────────────────────────────┘

PROBE 4: Substitution
┌─────────────────────────────────────────────────────────────────┐
│ Problem:                                                        │
│   Edge [Basic, Premium] means Premium can serve Basic's demand  │
│   Incorrect implementation leaves substitution non-functional   │
│                                                                 │
│ Symptom: INFEASIBLE when Basic cannot produce but Premium can   │
│ Test: Set production_cap[Basic] = 0, Premium > 0                │
│ Fix: Create S variable, add demand_route constraint             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Supported Datasets

| Dataset | Description | Type | Layers 1-6 | Layer 7 |
|---------|-------------|------|------------|---------|
| **RetailOpt-190** | Industrial retail inventory | MILP | ✅ | ✅ RetailProbes |
| **MAMO-Easy** | Mathematical modeling (easy) | LP/MILP | ✅ | ❌ N/A |
| **MAMO-Complex** | Mathematical modeling (hard) | MILP | ✅ | ❌ N/A |
| **NL4OPT** | NL-to-optimization benchmark | LP/MILP | ✅ | ❌ N/A |
| **Any LP/MILP** | Generic optimization problems | Any | ✅ | Extensible |

---

## Installation

```bash
# Clone repository
git clone https://github.com/junbolian/ReLoop.git
cd ReLoop

# Install dependencies
pip install gurobipy          # Gurobi solver
pip install openai anthropic  # LLM API clients
pip install transformers      # for local models (SIRL, ORLM)
pip install datasets          # for loading MAMO from HuggingFace

# Or install all at once
pip install -r requirements.txt
```

Requirements:
- Python ≥ 3.8
- Gurobi with valid license
- LLM API access (OpenAI, Anthropic, or local models)

---

## Usage Examples

### 1. Basic Pipeline Usage

```python
from reloop import ReLoop, ReLoopConfig, OpenAIClient

# Create LLM client
client = OpenAIClient(model="gpt-4o")

# Create pipeline with configuration
config = ReLoopConfig(
    max_iterations=5,    # Max repair iterations
    delta=0.2,           # Perturbation ratio for sensitivity (20%)
    epsilon=1e-4,        # Threshold for "no effect"
    timeout=60,          # Code execution timeout
    verbose=True         # Print progress
)

pipeline = ReLoop(client, config)

# Run ReLoop
result = pipeline.run(
    problem="Minimize total inventory cost...",
    schema=RETAIL_SCHEMA,
    data=scenario_data,
    obj_sense="minimize"
)

# Results
print(result)  # Shows status, iterations, layers passed
if result.verified:
    print("Success!")
    print(result.code)
else:
    print(f"Failed at layer {result.final_report.failed_layer}")
    print(f"Diagnosis: {result.final_report.diagnosis}")
```

### 2. Convenience Function

```python
from reloop import run_reloop

# Simple one-liner usage
result = run_reloop(
    problem="Minimize total inventory cost...",
    schema=RETAIL_SCHEMA,
    data=scenario_data,
    llm_client=client,
    verbose=True
)
```

### 3. Standalone Verification

```python
from reloop import BehavioralVerifier, verify_code

# Quick verification
report = verify_code(code, data, verbose=True)

# Or with custom configuration
verifier = BehavioralVerifier(
    delta=0.2,           # Perturbation ratio
    epsilon=1e-4,        # No-effect threshold
    timeout=60           # Execution timeout
)

report = verifier.verify(
    code=my_code,
    data=my_data,
    obj_sense="minimize",
    enable_layer7=True,  # Enable domain-specific probes
    verbose=True
)

# Detailed analysis
print(report)  # Full verification report
print(f"Layers passed: {report.count_layers_passed()}/7")
if not report.passed:
    print(f"Failed at layer {report.failed_layer}: {report.diagnosis}")
```

### 4. Baseline Comparison

```python
from reloop import ReLoop, ReLoopConfig

# Run baseline (single shot, no verification loop)
result = pipeline.run_baseline(
    problem="Minimize total inventory cost...",
    schema=RETAIL_SCHEMA,
    data=scenario_data,
    obj_sense="minimize"
)

# Compare with full ReLoop
result_full = pipeline.run(problem, schema, data)

print(f"Baseline: {result.best_layers_passed}/7 layers")
print(f"ReLoop:   {result_full.best_layers_passed}/7 layers")
```

### 5. Command Line

```bash
# Run from command line
python -m reloop.reloop \
    --problem "path/to/problem.txt" \
    --schema "path/to/schema.txt" \
    --data "path/to/data.json" \
    --model gpt-4o \
    --max-iter 5 \
    --verbose
```

---

## Experimental Results

### Main Results (Table 1)

ReLoop provides significant improvements across **all base models**:

| Model | RetailOpt-190 | | MAMO-Complex | |
|-------|---------------|---------|--------------|---------|
| | Direct | +ReLoop | Direct | +ReLoop |
| GPT-4o | 45.2 | **68.5** (+23.3) | 52.1 | **71.8** (+19.7) |
| Claude Opus 4.5 | 48.1 | **70.2** (+22.1) | 55.3 | **73.5** (+18.2) |
| SIRL-7B | 42.0 | **63.8** (+21.8) | 51.7 | **69.2** (+17.5) |
| ORLM-8B | 38.0 | **58.5** (+20.5) | 37.4 | **56.8** (+19.4) |

**Key Findings:**
1. All models show significant improvement (17-23 pp)
2. Larger gains on complex problems
3. Works for both closed-source and open-source models
4. Training-time methods (SIRL) + ReLoop achieve best results

### Ablation Study (Table 2)

Component contributions on GPT-4o with RetailOpt-190:

| Configuration | Obj Acc | Δ |
|---------------|---------|--------|
| Full ReLoop | 68.5% | - |
| − Sensitivity Analysis (Layer 3-4) | 52.3% | -16.2 |
| − Repair Loop | 58.1% | -10.4 |
| − Structured Generation | 62.8% | -5.7 |

**Key Finding:** Sensitivity analysis contributes most (-16.2 pp when removed)

### Error Detection Capability (Table 3)

| Error Type | Detection Rate | Diagnosis Rate |
|------------|----------------|----------------|
| Constraint Missing | 92.3% | 86.1% |
| Wrong Direction | 95.8% | 91.2% |
| Objective Error | 84.5% | 78.3% |
| Coefficient Error | 71.2% | 63.5% |
| **Average** | **85.9%** | **79.8%** |

**Key Finding:** ReLoop detects 86% of silent failures

### Base Models Evaluated

| Model | Type | Notes |
|-------|------|-------|
| GPT-4o | Closed-source | SOTA general LLM |
| Claude Opus 4.5 | Closed-source | SOTA general LLM |
| SIRL-7B | Open-source | Training-time RL method |
| ORLM-8B | Open-source | Training-time SFT method |
| LLMOPT-14B | Open-source | ICLR 2025 |
| OptiChat | Framework | Uses closed-source API |

### Validation Tests

Individual scenario tests confirm the framework works correctly:

| Model | Scenario | Baseline Gap | ReLoop Gap | Layers |
|-------|----------|--------------|------------|--------|
| Claude Opus 4.5 | retail_f1_52_weeks_v0 | 0.00% | 0.00% | 3/7 |
| Claude Opus 4.5 | retail_f5_ultimate_stress_v0 | 1.06% | 1.06% | 7/7 |
| GPT-5.1 | retail_f1_52_weeks_v0 | 2.54% | 2.87% | 3/7 |

**Observations:**
- Claude Opus 4.5 achieves near-optimal results (~0-1% gap) even with single-shot baseline
- GPT-5.1 shows 2.54% gap, demonstrating prompts don't leak answers (if leaked, all models would get ~0%)
- L4 "NO EFFECT" failures can be false positives when constraints have slack
- Final metric is objective gap (< 1% threshold), not layer count alone
- ReLoop provides most value for weaker models on complex multi-constraint scenarios

---

## Repository Structure

```
reloop/
├── __init__.py                       # Package exports (30+ public APIs)
├── reloop.py                         # Main pipeline orchestrator
├── structured_generation.py          # Module 1: 3-step generation
├── behavioral_verification.py        # Module 2: 7-layer verification (Core)
├── diagnosis_repair.py               # Module 3: Diagnosis-guided repair
├── prompts.py                        # Comprehensive prompt templates
├── param_utils.py                    # Parameter utilities for sensitivity
└── error_patterns.py                 # Static error pattern table

scenarios/
├── spec/
│   ├── retail_spec.md                # Benchmark specifications
│   └── retail_prompts.md             # Prompt documentation
├── data/                             # 190 JSON instances
└── prompts/                          # Per-instance prompts

solvers/
└── universal_retail_solver.py        # Reference MILP (ground truth)

eval/
├── run_benchmark.py                  # Evaluation script
└── evaluate_with_probes.py           # Probe-based evaluation

docs/
└── CONTRIBUTIONS.md                  # Research contributions
```

---

## Data Usage Principle

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

| Step | LLM Sees | Full Data Used For |
|------|----------|-------------------|
| 1-3 (Modeling) | Narrative + Schema | - |
| 4 (Verification) | - | Execute code, sensitivity tests |
| 5 (Repair) | Code + "demand anomaly" | - |

---

## Repair Mechanism Details

### Early Stopping
```
If no improvement for 2 consecutive iterations → STOP
This prevents wasting compute on unfixable errors.

To disable early stopping (for research/debugging):
  python run_test_with_log.py --no-early-stop --max-iter 10

Note: Experiments show that increasing iterations without early stop
does NOT improve results for weak models (e.g., gpt-4o ping-pongs
between errors). Early stop is recommended for production use.
```

### Smart Repair Strategy
```
═══════════════════════════════════════════════════════════════════
KEY INSIGHT: Not all L3+ failures are "informational only"
═══════════════════════════════════════════════════════════════════

Problem: Original strategy stopped repair after L1/L2 passed, treating
all L3+ failures as "possibly slack constraints". But this misses:
  - L3 failures: Code structure issues (missing constraints)
  - L4 "NO EFFECT" failures: Parameter not used = constraint MISSING

New Strategy:
┌─────────────────────────────────────────────────────────────────┐
│ Failure Type          │ Action                                 │
├───────────────────────┼────────────────────────────────────────┤
│ L1/L2 failure         │ REPAIR (execution/feasibility bugs)    │
├───────────────────────┼────────────────────────────────────────┤
│ L3 failure            │ REPAIR (code structure issues)         │
├───────────────────────┼────────────────────────────────────────┤
│ L4 "NO EFFECT"        │ REPAIR (constraint missing, not slack) │
├───────────────────────┼────────────────────────────────────────┤
│ L4 direction mismatch │ SKIP (may be slack constraint)         │
├───────────────────────┼────────────────────────────────────────┤
│ L5/L6/L7 failures     │ SKIP (informational, not critical)     │
└─────────────────────────────────────────────────────────────────┘

Why "NO EFFECT" means constraint is missing (not slack):
  - Slack constraint: param change → small effect (objective changes)
  - Missing constraint: param change → NO effect (objective unchanged)

Example:
  cold_capacity ±20% → objective unchanged
  → Storage constraint is NOT in the model!
  → Trigger repair with diagnosis: "cold_capacity has NO EFFECT"
```

### Preservation Rules
```
When repair is triggered at Layer N, the following are PROTECTED:
  - Layer 1 passed → imports, variable definitions preserved
  - Layer 2 passed → objective function, existing constraints preserved

Repair prompt explicitly tells LLM: "DO NOT modify working parts"
```

### Constraint Pattern Hints
```
When L3 fails on parameter 'cold_capacity':
  - System infers: "capacity" → CAPACITY role
  - Suggests: m.addConstr(sum(...) <= data['cold_capacity'][key])

This guides LLM to add the SPECIFIC missing constraint.
```

### Error Pattern Matching
```
L1 errors are matched to specific repair hints:

TypeError patterns:
  - "unhashable type: list" → Convert lists to tuples for Gurobi addVars()
  - "unsupported operand type(s) for *: float and GenExprMax"
    → gp.max_() returns expression, use auxiliary variable instead
  - "'>' not supported between Var and int"
    → Use indicator constraints or Big-M, not direct comparison
  - "Var object has no attribute"
    → Don't access .X during model building, use variable directly

KeyError patterns:
  - Use data.get('key', default) for optional fields
  - Check if key exists before accessing

IndexError patterns:
  - Check array bounds (t-1 for 0-indexed)
  - Verify loop ranges match data dimensions
```

---

## Conversation Logging

ReLoop supports detailed conversation logging for analysis and debugging.

### Running with Logs
```bash
# Run ReLoop with conversation logging
python run_test_with_log.py --scenario retail_f1_base_v4 --model gpt-4o --max-iter 5

# Run baseline (direct generation, no 3-step or repair)
python run_test_with_log.py --baseline --scenario retail_f1_base_v4 --model gpt-4o

# Compare ReLoop vs Baseline side-by-side
python run_test_with_log.py --compare --scenario retail_f1_base_v4 --model gpt-4o

# Output: logs/retail_f1_base_v4_gpt-4o_reloop_20260126_124304.json
#         logs/retail_f1_base_v4_gpt-4o_baseline_20260126_124304.json
```

### Comparison Mode Output
```
============================================================
COMPARISON RESULTS
============================================================
Metric                           Baseline          ReLoop      Delta
------------------------------------------------------------
Layers Passed                           2/7               2/7         +0
LLM Turns                               1               5         +4
Duration (s)                        25.52           96.98     +71.46

------------------------------------------------------------
CONCLUSION: No difference in layers passed
```

### Log Structure
```json
{
  "scenario_id": "retail_f1_base_v4",
  "model": "gpt-4o",
  "start_time": "2026-01-26T12:43:04",
  "total_duration_s": 34.41,
  "iterations": 3,
  "final_status": "NOT_VERIFIED",
  "layers_passed": 0,
  "turns": [
    {"turn_id": 1, "role": "generation", "step": "step1", "prompt": "...", "response": "..."},
    {"turn_id": 2, "role": "generation", "step": "step2", "prompt": "...", "response": "..."},
    {"turn_id": 3, "role": "generation", "step": "step3", "prompt": "...", "response": "..."},
    {"turn_id": 4, "role": "repair", "step": "layer1_repair", "prompt": "...", "response": "..."}
  ],
  "verification_reports": [
    {"iteration": 1, "passed": false, "layers_passed": 0, "failed_layer": 1, "diagnosis": "..."}
  ],
  "final_code": "..."
}
```

This enables:
- Analyzing which module contributes most to success/failure
- Debugging specific error patterns
- Comparing performance across models

---

## Prompt Design Lessons

### L3 Failures and Prompt Clarity

When L3 (Monotonicity) reports "No effect detected" for a parameter, check:

1. **Is the constraint described in the prompt?**
   - Example: `shelf_life` was missing from `retail_f1_base` description
   - Fix: Added explicit constraint semantics to `archetypes.yaml`

2. **Is the constraint semantics clear enough?**
   - Bad: "shelf_life: shelf life in periods per product" (what does this mean?)
   - Good: "Units produced in period t can only be held for shelf_life[p] periods; after that they expire"

3. **Is the constraint formula specified?**
   - For complex constraints, provide the formula:
   - `sum over products of (cold_usage[p] * inventory[p,l,t]) <= cold_capacity[l]`

### Key Parameters That Require Clear Semantics

| Parameter | Required Semantics |
|-----------|-------------------|
| `shelf_life` | Age-indexed inventory I[p,l,t,a], FIFO sales, automatic expiry when age > shelf_life[p] |
| `cold_capacity/cold_usage` | Formula: `sum(cold_usage[p] * inventory[p,l,t]) <= cold_capacity[l]` |
| `lead_time` | Orders placed in period t arrive in period t + lead_time[p]; distinguish in-transit vs on-hand |
| `return_rate` | Fraction of sales returned next period; specify re-entry as age-1 inventory |
| `labor_cap/labor_usage` | Formula: `sum(labor_usage[p] * units_handled[p,l,t]) <= labor_cap[l,t]` |
| `waste_limit_pct` | Global constraint: `sum(waste) <= waste_limit_pct * sum(demand)` |
| `moq` | All-or-nothing: order quantity must be 0 or >= moq |
| `pack_size` | Order quantity must be integer multiple of pack_size |

### Scenario Descriptions (archetypes.yaml)

Each scenario family (F1-F8) in `archetypes.yaml` now includes:
- **Business narrative**: High-level description of the scenario
- **Structure cues**: Explicit constraint semantics with formulas

Example from `retail_f1_base`:
```yaml
- Shelf life: Each product has a shelf life in periods. Inventory must be
  tracked by age (cohorts). Units produced in period t can only be sold or
  held for shelf_life[p] periods; after that they expire and must be
  discarded as waste. The model must use age-indexed inventory I[p,l,t,a]
  where a is age 1..shelf_life[p], with FIFO sales (oldest first) and
  automatic expiry when age exceeds shelf life.
- Storage capacity: sum over products of (cold_usage[p] * total_inventory[p,l,t])
  <= cold_capacity[l]. These limits must be respected.
```

### Regenerating Prompts

After updating `scenarios/spec/archetypes.yaml`:
```bash
python tools/generate_prompts.py
```

This regenerates all `.base.txt` and `.scenario.txt` files in `scenarios/prompts/` (190 scenarios total).

---

## FAQ

**Q: Does ReLoop work for maximization problems?**
A: Yes! Set `obj_sense="maximize"` and expectations are automatically adjusted.

**Q: What if my parameters don't have clear names (e.g., just 'c', 'A', 'b')?**
A: Layer 3 (Monotonicity) is completely name-agnostic. It checks if ANY numeric parameter has ANY effect on the objective.

**Q: Can I add probes for my own domain?**
A: Yes! Extend the framework by creating a new probes class. See RetailProbes for reference.

**Q: Why not use unit tests instead?**
A: Unit tests require knowing the correct answer beforehand. ReLoop checks behavioral REASONABLENESS without needing ground truth.

**Q: Is ReLoop compatible with solvers other than Gurobi?**
A: Yes! COPT is supported. Other solvers can be added by modifying the CodeExecutor class.

---

## Appendix (Additional Experiments)

The following are available in the paper appendix:
- **A.** Full dataset results (including NL4OPT, MAMO-Easy)
- **B.** Verification method comparison (vs Random Testing, Self-Check)
- **C.** Efficiency analysis
- **D.** Case studies
- **E.** Cross-difficulty/problem-type analysis

---

## Citation

```bibtex
@misc{reloop2026,
  author = {Junbo Jacob Lian and Yujun Sam Sun and Huiling Chen and Chaoyu Zhang and Chung-Piaw Teo},
  title  = {ReLoop: Reliable LLM-based Optimization Modeling
            via Sensitivity-Based Behavioral Verification},
  year   = {2026},
}
```

---

## License

MIT License. Released for research and educational use.
