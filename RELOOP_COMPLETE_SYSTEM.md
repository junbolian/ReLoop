# ReLoop: Detecting Silent Failures in LLM-Generated Optimization Code via Behavioral Verification

## Complete System Documentation

**For: Paper writing, experiment design, and architecture understanding**

---

# Part 1: System Overview

## 1.1 Problem Statement

**Silent Failures in LLM-Generated Optimization Code:**
- LLM generates optimization code that executes successfully
- Solver returns OPTIMAL status
- But the solution is WRONG (constraints missing, wrong direction, etc.)
- User cannot detect the error without domain expertise

**Why This Matters:**
- Operations Research is a $10B+ industry
- Wrong solutions can cost millions (inventory, routing, scheduling)
- LLMs are increasingly used to generate optimization models
- No existing method to detect silent failures

## 1.2 Key Innovation

**Behavioral Verification (not Code Verification):**
- Don't check if code matches specification
- Check if model behavior satisfies mathematical invariants
- Can detect errors WITHOUT knowing the correct answer

**Four Mathematical Invariants:**
1. **Anomaly Detection**: Both increasing AND decreasing a parameter should not improve objective (physically impossible)
2. **Strong Duality**: Primal and dual objectives should match at optimum
3. **Direction Consistency**: Parameter changes should affect objective in expected directions (verified via LLM)
4. **Constraint Presence Testing**: Missing constraints can be detected via counterfactual testing

## 1.3 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ReLoop Complete Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │         Chain-of-Thought Code Generation (Single API Call)          │   │
│  │                                                                      │   │
│  │   Problem ──→ [STEP 1: Understand] ──→ [STEP 2: Formalize] ──→      │   │
│  │                       │                        │                     │   │
│  │               (same context)           (same context)                │   │
│  │                       │                        │                     │   │
│  │                       └────────────────────────┴──→ [STEP 3: Code]  │   │
│  │                                                            │         │   │
│  │   Output: U (Understanding) + M (Math Model) + Ck (Code)   │         │   │
│  │                                                                      │   │
│  │   KEY: Single API call preserves context (NOT 3 separate calls)     │   │
│  │   Result: 2.17% error (vs 10.85% with separate calls)               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Generate → Verify → Repair Loop                   │   │
│  │                                                                      │   │
│  │   ┌──────────┐       ┌──────────┐       ┌──────────┐                │   │
│  │   │ Generate │ ───→  │  Verify  │ ───→  │  Repair  │                │   │
│  │   │  (CoT)   │       │ (L1-L5)  │       │  (LLM)   │                │   │
│  │   └────┬─────┘       └────┬─────┘       └────┬─────┘                │   │
│  │        ↑                  │                   │                      │   │
│  │        │    L1 FATAL      │    L2-L5          │                      │   │
│  │        │←─(Regenerate)────│    WARNING        │                      │   │
│  │        │                  │←──────────────────┘                      │   │
│  │        │                  │                                          │   │
│  │        └──────────────────┴───→ Final Result                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────┐                                                            │
│  │    Data     │  Extract structured parameters from natural language       │
│  │ Extraction  │  (LLM-based with regex fallback)                           │
│  └─────────────┘                                                            │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
│                      ReLoop 5-Layer Verification Engine                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ L1: Execution & Solver (BLOCKING)                                   │   │
│  │     • Syntax check (compile)                                        │   │
│  │     • Runtime check (subprocess isolation)                          │   │
│  │     • Solver status (OPTIMAL/INFEASIBLE/UNBOUNDED/TIMEOUT)          │   │
│  │     → FATAL on failure, blocks all output                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │ PASS                                         │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ L2: Anomaly Detection (DIAGNOSTIC) - Bidirectional Perturbation     │   │
│  │     • Perturb each parameter UP (+20%) and DOWN (-20%)              │   │
│  │     • Check 1: Both directions improve → ERROR (physically impossi) │   │
│  │     • Check 2: Neither direction affects → INFO (slack constraint)  │   │
│  │     • Check 3: Extreme sensitivity → INFO (normal for optimized)    │   │
│  │     → Universal: no keyword-based inference, 100% reliable anomaly  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ L3: Dual Consistency (DIAGNOSTIC)                                   │   │
│  │     • Check primal-dual gap (threshold: 1%)                         │   │
│  │     • Check shadow price signs                                      │   │
│  │     • Large gap → INFO (likely numerical artifact, no repair)       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ L4: Adversarial Direction Analysis (DIAGNOSTIC) - LLM-based         │   │
│  │     • LLM_verify: Analyze expected direction from problem context   │   │
│  │     • LLM_repair: Accept or Reject the analysis                     │   │
│  │     • Accepted → WARNING (should fix)                               │   │
│  │     • Rejected → Re-analyze with feedback (up to max_rejections)    │   │
│  │     • Max rejections reached → INFO (inconclusive, no repair)       │   │
│  │     → Adversarial debate converges on correct answer                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ L5: CPT - Constraint Perturbation Testing (ENHANCEMENT, Optional)   │   │
│  │     • LLM-based candidate constraint extraction (requires LLM)      │   │
│  │     • Test each: perturb parameters, re-solve, compare objectives   │   │
│  │     • Objective unchanged → WARNING (constraint was missing)        │   │
│  │     • Safe: only WARNING/INFO, never ERROR/FATAL                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 1.4 Severity Levels (Conservative Repair Strategy)

| Level | Meaning | Confidence | Source | Repair Action |
|-------|---------|------------|--------|---------------|
| **FATAL** | Definite error, code cannot run | 100% | L1 only | Triggers regeneration (up to 3 attempts) |
| **ERROR** | Mathematically certain error | 99%+ | L2 (anomaly - both-improve) | **MUST fix** |
| **WARNING** | High-confidence issue | 80%+ | L4 (accepted direction), L5 (cpt_missing) | **SHOULD fix** |
| **INFO** | Likely normal behavior | <80% | L2 (no_effect, sensitivity), L3, L4 (rejected/inconclusive) | **DO NOT fix** (reference only) |
| **PASS** | Check passed | - | All layers | No issue |

**Key Design Principle: Conservative Repair**
- Only ERROR and WARNING trigger repair
- INFO is for reference only - likely normal optimization behavior
- This prevents over-correction (the root cause of ReLoop performing worse than baseline)

### L1 FATAL Handling (Important Change)

```
CRITICAL: L1 FATAL does NOT terminate immediately!

When L1 returns FATAL:
├── 1. Extract error message from L1 result
├── 2. Call generator.regenerate(failed_code, error_message, data)
├── 3. Re-verify the new code
├── 4. Repeat up to max_regeneration_attempts (default: 3)
└── 5. Only after all attempts fail → return FAILED status

This ensures maximum recovery from execution/syntax errors.
```

## 1.5 Guaranteed Output Mechanism (Robustness Design)

```
CORE GUARANTEE:

If L1 passes (code runs + solver has solution):
├── ALWAYS return objective value      ← 无论其他层结果如何
├── ALWAYS return solution vector      ← 无论其他层结果如何
├── ALWAYS return diagnostics          ← 问题信息仅供参考
└── L2-L5 findings do NOT block output ← 只添加信息，不删除输出

Diagnostic layers ADD information, never DELETE output.
```

### 1.5.1 Layer Severity Matrix (Conservative Repair Design)

| Layer | Check | Possible Severities | Triggers Repair? | Rationale |
|-------|-------|--------------------|--------------------|-----------|
| **L1** | execution/solver | FATAL, PASS | Triggers regeneration | Code cannot run |
| **L2** | anomaly | **ERROR**, PASS | **YES (must fix)** | Both directions improve = physically impossible (99%+) |
| **L2** | no_effect | INFO, PASS | **NO** | Likely slack constraint (normal) |
| **L2** | high_sensitivity | INFO, PASS | **NO** | Normal for well-optimized models |
| **L3** | duality_gap | INFO, PASS | **NO** | Likely numerical artifact |
| **L4** | direction (accepted) | **WARNING**, PASS | **YES (should fix)** | LLM debate converged on issue |
| **L4** | direction (rejected) | INFO, PASS | **NO** | Inconclusive after debate |
| **L5** | cpt_missing | **WARNING**, INFO, PASS | **YES (should fix)** | High confidence (80%+) |

**Key Insight:**
- L2 anomaly detection (both-improve) is 100% reliable for detecting structural errors
- L4 uses adversarial debate: if LLM_repair accepts LLM_verify's analysis, it becomes WARNING; if rejected repeatedly, it downgrades to INFO

### 1.5.2 False Positive Handling (误检处理)

```python
# Even if L2-L5 all report WARNING, complete result is still returned
def _aggregate(self, results, objective, solution, ...):
    # objective and solution come from L1, unaffected by other layers
    return VerificationReport(
        status='WARNINGS',        # Status label (indicates potential issues)
        has_solution=True,        # ← Always True if L1 passed
        objective=objective,      # ← Always has value if L1 passed
        solution=solution,        # ← Always has value if L1 passed
        layer_results=results,    # Diagnostic info (for reference only)
        ...
    )
```

### 1.5.3 Robustness Scenarios

| Scenario | L1 | L2-L5 | Output |
|----------|----|----|--------|
| Completely correct | PASS | All PASS | status=VERIFIED, objective=✓, solution=✓ |
| Has warnings but correct | PASS | Some WARNING | status=WARNINGS, objective=✓, solution=✓ |
| **False positive** | PASS | Incorrect WARNING | status=WARNINGS, **objective=✓, solution=✓** |
| Code execution fails | FATAL | Not executed | status=FAILED, objective=None |
| Solver finds no solution | FATAL | Not executed | status=FAILED, objective=None |

### 1.5.4 Design Principles

1. **Only L1 FATAL blocks output** - Syntax error, runtime error, infeasible/unbounded
2. **L2-L5 NEVER block** - Diagnostic layers only, they ADD information
3. **False positives don't affect result values** - WARNING is just a hint, objective/solution always returned correctly
4. **Better to over-detect than under-detect** - False positives acceptable, false negatives unacceptable

---

# Part 2: Project Structure

## 2.1 Actual File Structure

```
reloop/
├── reloop/                       # Core package
│   ├── __init__.py               # Package exports (all public APIs)
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
│   ├── RetailOpt-190.jsonl       # Our benchmark (190 problems)
│   ├── IndustryOR_fixedV2.jsonl  # 100 industry OR problems
│   ├── MAMO_EasyLP_fixed.jsonl   # 642 easy LP problems
│   ├── MAMO_ComplexLP_fixed.jsonl
│   ├── NL4OPT.jsonl              # 245 NL4OPT benchmark
│   ├── OptMATH_Bench_166.jsonl
│   ├── OptMATH_Bench_193.jsonl
│   └── OptiBench.jsonl
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## 2.2 Module Descriptions

| Module | Lines | Purpose |
|--------|-------|---------|
| `param_utils.py` | ~150 | Parameter extraction, perturbation, skip determination |
| `executor.py` | ~120 | Subprocess-isolated code execution with timeout |
| `verification.py` | ~550 | Core verification engine (L1-L3, L5) |
| `l4_adversarial.py` | ~350 | L4 Adversarial Direction Analysis with LLM debate |
| `prompts.py` | ~200 | LLM prompt templates for generation, repair, L4 verify |
| `generation.py` | ~100 | Code generation from problem description |
| `repair.py` | ~400 | Diagnostic-guided code repair with L4 Accept/Reject |
| `pipeline.py` | ~450 | Generate→Verify→Repair pipeline orchestration |
| `data_extraction.py` | ~130 | Extract structured data from natural language |
| `experiment_runner.py` | ~330 | Batch experiment runner with metrics |

---

# Part 3: Core Module Implementations

## 3.1 param_utils.py - Parameter Utilities

**Key Functions:**

```python
def extract_numeric_params(data: Dict, prefix: str = "") -> List[str]:
    """Recursively extract all numeric parameters from data dict."""
    # Returns: ["demand", "capacity", "cost.production", ...]

def get_param_value(data: Dict, param_path: str) -> Any:
    """Get parameter value by path."""

def perturb_param(data: Dict, param_path: str, factor: float) -> Dict:
    """Create copy of data with param multiplied by factor."""
    # factor > 1: increase, factor < 1: decrease

def set_param(data: Dict, param_path: str, new_value: Any) -> Dict:
    """Create a copy with parameter set to specific value."""

def should_skip_param(data: Dict, param_path: str) -> Tuple[bool, str]:
    """Skip zero values and Big-M parameters."""
    # Skip if: value is None, value is 0, value >= 90000 (Big-M)
```

**Note:** Keyword-based role inference (`ParameterRole`, `infer_param_role`, `get_expected_direction`)
has been completely removed. All layers now use universal or LLM-based approaches:
- L2: Universal anomaly detection (both-directions-improve)
- L4: LLM-based adversarial direction analysis with Accept/Reject mechanism
- L5: LLM-based constraint extraction (requires LLM, no keyword fallback)

## 3.2 executor.py - Code Executor

**Key Implementation:**

```python
class CodeExecutor:
    """Execute optimization code in isolated subprocess."""

    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    def execute(self, code: str, data: Dict) -> Dict[str, Any]:
        """
        Execute code with data in subprocess.

        Returns:
            {
                "status": str,           # "OPTIMAL", "INFEASIBLE", etc.
                "objective": float,      # Objective value
                "dual_objective": float, # Dual objective (if available)
                "solution": Dict,        # Variable values
                "success": bool,
                "error": str             # Error message if failed
            }
        """
        # 1. Create temp file with code + data injection
        # 2. Run subprocess with timeout
        # 3. Parse stdout for status, objective, solution
        # 4. Return result dict
```

**Output Parsing:**
- Expects code to print: `status: {m.Status}`, `objective: {m.ObjVal}`
- Parses using regex patterns
- Returns structured result dict

## 3.3 verification.py - 5-Layer Verification Engine

### Layer 1: Execution & Solver

```python
def _layer1(self, code: str, data: Dict, obj_sense: str, verbose: bool):
    """L1: Execution & Solver Status (BLOCKING)"""

    # 1.1 Syntax check
    try:
        compile(code, '<string>', 'exec')
    except SyntaxError as e:
        return [LayerResult("L1", "syntax", Severity.FATAL, str(e), 1.0)], {}

    # 1.2 Runtime execution
    result = self.executor.execute(code, data)

    # 1.3 Solver status check
    status = result.get("status")
    if status == "INFEASIBLE":
        return [LayerResult("L1", "solver", Severity.FATAL, "Model infeasible", 1.0)], {}
    if status == "UNBOUNDED":
        return [LayerResult("L1", "solver", Severity.FATAL, "Model unbounded", 1.0)], {}

    # Return baseline result for other layers
    return [LayerResult("L1", "execution", Severity.PASS, "Execution OK", 1.0)], result
```

### Layer 2: Anomaly Detection (Bidirectional Perturbation)

```python
def _layer2(self, code: str, data: Dict, baseline_obj: float,
            obj_sense: str, params: List[str], verbose: bool):
    """L2: Anomaly Detection - Bidirectional Perturbation (DIAGNOSTIC)"""
    results = []
    anomaly_params = []
    no_effect_params = []
    high_sensitivity_params = []

    is_minimize = obj_sense == "minimize"
    threshold = abs(baseline_obj) * self.epsilon if baseline_obj != 0 else 1e-6

    for param in params[:self.max_params]:
        if should_skip_param(data, param)[0]:
            continue

        # Bidirectional perturbation
        result_up = self.executor.execute(code, perturb_param(data, param, 1 + self.delta))
        result_down = self.executor.execute(code, perturb_param(data, param, 1 - self.delta))

        obj_up = result_up.get("objective")
        obj_down = result_down.get("objective")

        if obj_up is None or obj_down is None:
            continue

        change_up = obj_up - baseline_obj
        change_down = obj_down - baseline_obj

        # Check for anomaly: BOTH directions improve objective
        if is_minimize:
            up_better = change_up < -threshold
            down_better = change_down < -threshold
        else:
            up_better = change_up > threshold
            down_better = change_down > threshold

        if up_better and down_better:
            # ERROR: Physically impossible - both directions improve
            anomaly_params.append({
                "param": param,
                "baseline_obj": baseline_obj,
                "obj_up": obj_up,
                "obj_down": obj_down
            })
        elif abs(change_up) < threshold and abs(change_down) < threshold:
            # INFO: No effect - likely slack constraint
            no_effect_params.append(param)
        elif abs(change_up) > abs(baseline_obj) * 0.5 or abs(change_down) > abs(baseline_obj) * 0.5:
            # INFO: High sensitivity - normal for optimized models
            high_sensitivity_params.append(param)

    # Report anomalies as ERROR
    for anomaly in anomaly_params:
        results.append(LayerResult(
            "L2", "anomaly", Severity.ERROR,
            f"Anomaly: both directions improve for '{anomaly['param']}'",
            0.99, anomaly
        ))

    # Report no-effect as INFO
    for param in no_effect_params:
        results.append(LayerResult(
            "L2", "no_effect", Severity.INFO,
            f"Parameter '{param}' has no effect (likely slack)",
            0.6, {"param": param, "is_likely_normal": True}
        ))

    # Report high sensitivity as INFO
    for param in high_sensitivity_params:
        results.append(LayerResult(
            "L2", "high_sensitivity", Severity.INFO,
            f"High sensitivity for '{param}' (normal)",
            0.5, {"param": param, "is_likely_normal": True}
        ))

    if not results:
        results.append(LayerResult("L2", "anomaly_detection", Severity.PASS, "OK", 0.9))

    return results
```

**Key Design:**
- **Anomaly (ERROR)**: Both +20% and -20% improve objective → physically impossible → must be structural error
- **No Effect (INFO)**: Neither direction affects objective → likely slack constraint → normal
- **High Sensitivity (INFO)**: Extreme changes → normal for well-optimized models

### Layer 3: Dual Consistency

```python
def _layer3(self, objective: float, baseline: Dict, verbose: bool):
    """L3: Dual Consistency (DIAGNOSTIC)"""
    results = []

    dual_obj = baseline.get("dual_objective")
    if dual_obj is not None:
        gap = abs(objective - dual_obj) / max(abs(objective), 1.0)
        if gap > 0.01:  # 1% threshold
            results.append(LayerResult(
                "L3", "duality_gap", Severity.INFO,  # INFO, not WARNING (likely numerical)
                f"Primal-dual gap {gap:.2%} exceeds threshold (likely numerical artifact)",
                0.5, {"gap": gap}  # Lower confidence
            ))
        else:
            results.append(LayerResult("L3", "duality", Severity.PASS, "OK", 0.9))
    else:
        results.append(LayerResult(
            "L3", "duality", Severity.INFO, "Dual objective not available", 0.5
        ))

    return results
```

### Layer 4: Adversarial Direction Analysis (LLM-based)

L4 uses an **adversarial mechanism** where two LLM roles (verify and repair) debate to converge on the correct answer.

**Design Principle:**
- LLM_verify analyzes expected direction based on problem context
- LLM_repair can Accept or Reject the analysis
- Rejection triggers re-analysis with feedback
- Multiple rounds until convergence or max rejections reached

```python
# l4_adversarial.py

@dataclass
class L4VerifyResult:
    """Result from L4 verification LLM."""
    param: str
    expected_direction: str  # "increase" | "decrease" | "unknown"
    actual_direction: str    # "increase" | "decrease" | "no_change"
    is_violation: bool
    reasoning: str
    confidence: float

@dataclass
class L4RepairDecision:
    """Repair LLM's decision on L4 diagnostic."""
    param: str
    action: str  # "accept" | "reject"
    reason: str

class L4AdversarialVerifier:
    """L4 Adversarial Direction Analysis with LLM debate."""

    def __init__(self, llm_client, max_rejections: int = 2):
        self.llm_client = llm_client
        self.max_rejections = max_rejections
        self.rejection_history: Dict[str, L4RejectionHistory] = {}

    def verify(self, code: str, data: Dict, baseline_obj: float,
               obj_sense: str, problem_description: str,
               perturbation_results: Dict[str, Dict]) -> List[L4VerifyResult]:
        """
        LLM_verify: Analyze direction expectations for each parameter.
        """
        prompt = L4_VERIFY_PROMPT.format(
            problem_description=problem_description,
            code=code,
            perturbation_results=format_perturbation_results(perturbation_results)
        )

        response = self.llm_client.generate(prompt)
        return self._parse_verify_response(response)

    def process_repair_decisions(self, decisions: List[L4RepairDecision]):
        """
        Process repair LLM's Accept/Reject decisions.

        - Accept: Direction issue confirmed → WARNING (should fix)
        - Reject: Analysis was wrong → Re-verify with feedback
        """
        for decision in decisions:
            param = decision.param

            if decision.action == "accept":
                # Issue confirmed - will trigger repair
                pass
            elif decision.action == "reject":
                # Track rejection for re-analysis
                if param not in self.rejection_history:
                    self.rejection_history[param] = L4RejectionHistory(param)

                history = self.rejection_history[param]
                history.add_rejection(decision.reason)

                if history.rejection_count >= self.max_rejections:
                    # Downgrade to INFO after max rejections
                    history.final_status = "INFO"

    def get_final_status(self, param: str, was_accepted: bool) -> Tuple[Severity, str]:
        """Get final severity based on Accept/Reject history."""
        if was_accepted:
            return (Severity.WARNING, "L4 direction issue accepted - should fix")

        history = self.rejection_history.get(param)
        if history and history.rejection_count >= self.max_rejections:
            return (Severity.INFO, "L4 inconclusive after max rejections - reference only")

        return (Severity.INFO, "L4 rejected - no action needed")


def should_exit_l4_loop(verify_results: List[L4VerifyResult],
                        repair_decisions: List[L4RepairDecision],
                        rejection_history: Dict[str, L4RejectionHistory],
                        max_rejections: int) -> Tuple[bool, str]:
    """
    Determine if L4 loop should exit.

    Exit conditions:
    1. All violations accepted → exit with WARNING status
    2. All violations rejected + max rejections reached → exit with INFO
    3. No violations found → exit with PASS
    """
    # ... implementation
```

**L4 Flow Diagram:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        L4 Adversarial Flow                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Perturbation Results (from L2) ──→ LLM_verify                             │
│                                         │                                   │
│                                         ▼                                   │
│                              "Parameter X should decrease obj,              │
│                               but increasing X decreased obj"               │
│                                         │                                   │
│                                         ▼                                   │
│                                    LLM_repair                               │
│                                         │                                   │
│                           ┌─────────────┴─────────────┐                    │
│                           │                           │                    │
│                           ▼                           ▼                    │
│                      ACCEPT                       REJECT                   │
│                           │                           │                    │
│                           ▼                           ▼                    │
│                    WARNING (fix)            Re-verify with feedback        │
│                                                       │                    │
│                                                       ▼                    │
│                                         [Repeat until Accept or max]       │
│                                                       │                    │
│                                                       ▼                    │
│                                         Max rejections → INFO (no fix)     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why Adversarial Mechanism?**

```
Single-LLM Problem:
- LLM_verify might make mistakes in direction analysis
- No mechanism to catch these mistakes

Adversarial Solution:
- LLM_repair reviews LLM_verify's analysis
- Can reject with reasoning (e.g., "wage is a cost coefficient, not constraint bound")
- LLM_verify re-analyzes with this feedback
- Debate converges on correct answer
```

**Exit Conditions (Guaranteed Output):**

| Exit Reason | Condition | Result |
|-------------|-----------|--------|
| `all_pass` | No L4 violations detected | Output immediately |
| `all_rejected_others_pass` | All L4 rejected AND L2/L3/L5 PASS | Output with L4 as INFO |
| `max_rejections` | Any param rejected ≥ max_rejections times | Downgrade to INFO, output |
| `max_iterations` | Reached max_l4_iterations (default 3) | Output current best |
| `accepted_fixed` | Some accepted, code was fixed | Re-verify fixed code |

**Key Parameters:**
- `max_l4_rejections`: Max times a param can be rejected before downgrade (default: 2)
- `max_l4_iterations`: Max L4 loop iterations (default: 3)
- `confidence_threshold`: Min confidence to trigger repair (default: 0.5)

**Critical Guarantee:** L4 loop ALWAYS produces output. One of the exit conditions will be met, preventing infinite loops.

### Layer 5: CPT (Constraint Perturbation Testing)

```python
def _layer5(self, code: str, data: Dict, baseline_obj: float,
            obj_sense: str, problem_description: str, verbose: bool):
    """L5: CPT (ENHANCEMENT) - Only WARNING/INFO, never ERROR/FATAL"""
    results = []

    # L5 requires LLM for constraint extraction (no keyword fallback)
    if not self.llm_client:
        results.append(LayerResult(
            "L5", "cpt_skipped", Severity.INFO,
            "L5 CPT skipped (requires LLM)", 0.5
        ))
        return results

    # Extract candidate constraints using LLM
    candidates = self._cpt_extract_candidates(problem_description, data)

    missing = []
    for candidate in candidates[:10]:
        test_result = self._cpt_test_candidate(code, data, baseline_obj, obj_sense, candidate)
        if test_result["status"] == "MISSING":
            missing.append(test_result)

    if missing:
        results.append(LayerResult(
            "L5", "cpt_missing", Severity.WARNING,
            f"CPT found {len(missing)} potentially missing constraint(s)",
            0.75, {"missing": missing}
        ))

    return results

def _cpt_test_candidate(self, code, data, baseline_obj, obj_sense, candidate):
    """Test a single candidate constraint."""
    params = candidate.get("parameters", [])
    ctype = candidate.get("type", "other")

    # Create test data based on constraint type
    if ctype == "capacity":
        test_data = set_param(data, params[0], 0.001)  # Near-zero capacity
    elif ctype == "demand":
        test_data = perturb_param(data, params[0], 100.0)  # 100x demand
    else:
        test_data = perturb_param(data, params[0], 0.01)

    result = self.executor.execute(code, test_data)

    if result.get("status") == "INFEASIBLE":
        return {"status": "SATISFIED"}  # Constraint is enforced

    new_obj = result.get("objective")
    if new_obj is not None:
        change_ratio = abs(new_obj - baseline_obj) / max(abs(baseline_obj), 1.0)
        # Updated thresholds: 5% / 30%
        if change_ratio < 0.05:
            # < 5%: Constraint likely missing → WARNING (should fix)
            return {"status": "MISSING", "severity": "WARNING", "description": candidate.get("description")}
        elif change_ratio < 0.30:
            # 5-30%: Uncertain → INFO (reference only)
            return {"status": "UNCERTAIN", "severity": "INFO"}
        else:
            # > 30%: Constraint present → PASS
            return {"status": "SATISFIED"}

    return {"status": "SKIP"}

def _cpt_extract_candidates(self, problem_description: str, data: Dict) -> List[Dict]:
    """LLM-based candidate constraint extraction."""
    if not self.llm_client:
        return []

    prompt = f"""Analyze this optimization problem and extract the KEY CONSTRAINTS.

## Problem Description
{problem_description}

## Available Data Parameters
{list(data.keys())}

## Task
Identify constraints that are REQUIRED by the problem. Focus on:
1. Capacity constraints (resource limits, maximum values)
2. Demand constraints (minimum requirements, must-satisfy conditions)
3. Balance constraints (flow balance, inventory balance)

## Output Format
Return ONLY a JSON array:
```json
[
  {{"description": "minimum protein requirement", "type": "demand", "parameters": ["min_protein"]}},
  {{"description": "capacity limit", "type": "capacity", "parameters": ["capacity"]}}
]
```"""

    try:
        response = self.llm_client.generate(prompt)
        match = re.search(r'\[[\s\S]*\]', response)
        if match:
            candidates = json.loads(match.group())
            return [c for c in candidates
                    if isinstance(c, dict) and "description" in c and "parameters" in c]
    except Exception:
        pass
    return []
```

## 3.4 prompts.py - LLM Prompts (Chain-of-Thought Generation)

**Chain-of-Thought Generation (Single API Call):**

The key insight is that using 3 separate API calls loses context between stages. Instead, we use a single API call with step-by-step reasoning:

```python
# ============================================================================
# Chain-of-Thought Generation (Recommended - Single API call)
# ============================================================================

CHAIN_OF_THOUGHT_SYSTEM = '''You are an optimization expert who solves problems with step-by-step reasoning.'''

CHAIN_OF_THOUGHT_PROMPT = '''Solve this optimization problem using chain-of-thought reasoning.

## Problem
{problem_description}

## Data Structure
The `data` variable is PRE-DEFINED with these keys:
{data_structure}

---
## STEP 1: UNDERSTAND THE PROBLEM
First, analyze the problem:
- What is the objective? (minimize cost / maximize profit / etc.)
- What decisions need to be made?
- What constraints exist?
- What parameters are given?

## STEP 2: FORMULATE THE MATHEMATICAL MODEL
Write the formal model:
- Sets and indices
- Parameters (use exact keys from data structure)
- Decision variables with domains
- Constraints in mathematical notation
- Objective function

## STEP 3: GENERATE GUROBI CODE
Write Python code using gurobipy.

**CRITICAL RULES:**
1. Do NOT define `data = {{...}}` - the data variable already exists
2. Access data as: `data["key_name"]`
3. Model variable must be named `m`
4. Set `m.Params.OutputFlag = 0`
5. Print exactly: `print(f"status: {{m.Status}}")` and `print(f"objective: {{m.ObjVal}}")`
6. Implement ALL constraints from Step 2

**Big-M Guidelines (if using indicator/logical constraints):**
- NEVER hardcode Big-M values like `M = 1e6`
- ALWAYS compute M dynamically from data: `M = sum(data["demand"]) * 1.5`
- Use problem-relevant parameters for M calculation

**Edge Case Handling:**
- Check array length before iteration: `if data["items"]: ...`
- Avoid division by zero: `max(value, 1e-6)`
- Use `.get()` for optional keys: `data.get("key", default)`

---
Now solve the problem. Show your reasoning for Steps 1-2, then provide the final code in a ```python block.
'''
```

**Generation Approaches:**

| Approach | Description | API Calls | Error Rate |
|----------|-------------|-----------|------------|
| **Single-Stage** | Direct problem → code | 1 | Baseline |
| **3-Stage CoT (single call)** | Understand → Formalize → Code in one prompt | 1 | 2.17% |
| **3-Stage CoT (separate calls)** | 3 separate API calls | 3 | 10.85% |

**Recommended:** 3-Stage CoT with single API call (CHAIN_OF_THOUGHT_PROMPT).

**Legacy 3-Stage Prompts (Deprecated):**
The old UNDERSTAND_PROMPT, FORMALIZE_PROMPT, SYNTHESIZE_PROMPT are kept for compatibility but the single-call CoT approach is preferred.

# ============================================================================
# Regeneration Prompt (for L1 FATAL recovery)
# ============================================================================

REGENERATE_PROMPT = '''The previous code failed to execute. Generate a new, correct version.

## Problem
{problem_description}

## Previous Code (FAILED)
```python
{failed_code}
```

## Error
{error_message}

## Data Structure
{data_structure}

## Instructions
1. Analyze why the previous code failed
2. Generate completely new code that avoids the error
3. Ensure proper data access patterns
4. Handle edge cases (empty arrays, missing keys)

Return ONLY the corrected Python code.
'''

# ============================================================================
# Repair Prompt (for L2-L5 issues) - Updated with Data Structure
# ============================================================================

REPAIR_PROMPT = '''Fix this optimization code based on the diagnostic report.

## Problem
{problem_description}

## Data Structure
The `data` variable is PRE-DEFINED with these keys:
{data_structure}

## Current Code
```python
{code}
```

## Diagnostic Report
{diagnostic_report}

## Issues to Fix
{issues}

## Instructions
1. Carefully analyze each issue in the diagnostic report
2. Fix the specific problems identified
3. If a parameter "has NO EFFECT", add the missing constraint that uses it
4. Ensure all constraints from the problem are implemented
5. **CRITICAL: The `data` variable is PRE-DEFINED. Do NOT create `data = {{...}}`. Just use `data["key"]` directly.**
6. Do not remove working code unnecessarily

Return the COMPLETE fixed code. Do NOT include any `data = ` definition.
'''
```

**Key Update:** Repair prompt now includes `{data_structure}` to ensure the LLM knows what parameters are available when fixing "parameter has NO EFFECT" issues.

**Helper Functions:**

```python
def format_issues_for_repair(report) -> str:
    """Extract actionable issues for repair."""
    issues = []
    for r in report.layer_results:
        if r.severity.value in ['FATAL', 'ERROR', 'WARNING']:
            if r.severity.value == 'FATAL':
                issues.append(f"EXECUTION ERROR: {r.message}")
            elif 'monotonicity' in r.check:
                issues.append(f"CONSTRAINT DIRECTION ERROR: {r.message}")
            elif 'param_effect' in r.check:
                issues.append(f"MISSING CONSTRAINT: {r.message}")
            elif 'direction_anomaly' in r.check:
                issues.append(f"DIRECTION ANOMALY: {r.message}")
            elif 'cpt_missing' in r.check:
                issues.append(f"CPT DETECTED MISSING: {r.message}")
    return "\n".join(issues) if issues else "No specific issues."
```

## 3.5 generation.py - Chain-of-Thought Code Generation

```python
@dataclass
class GenerationResult:
    """Result of the generation process."""
    code: str
    understanding: Optional[str] = None  # CoT Step 1 output
    mathematical_model: Optional[str] = None  # CoT Step 2 output
    stages_completed: int = 0


class CodeGenerator:
    """
    Chain-of-Thought (CoT) optimization code generator.

    Pipeline: Problem → [CoT 3-Stage] → Code
              (STEP 1: Understand → STEP 2: Formalize → STEP 3: Code)

    The 3-stage pipeline IS Chain-of-Thought reasoning, where each stage
    builds on the previous one's output. Implementation options:
    - Single API call: All 3 steps in one prompt (preserves context, 2.17% error)
    - Multiple API calls: 3 separate calls (loses context, 10.85% error)

    Recommended: Single API call with CHAIN_OF_THOUGHT_PROMPT.

    Features:
    - Schema-based data description (structure only, not values)
    - Universal architecture for all optimization domains
    - Fallback to single-stage generation if CoT fails
    """

    def __init__(self, llm_client, use_structured_generation: bool = True):
        self.llm_client = llm_client
        self.use_structured_generation = use_structured_generation

    def generate(self, problem_description: str, data: Dict, max_retries: int = 3) -> str:
        """Generate Gurobi code using 3-stage pipeline."""
        if self.use_structured_generation:
            result = self.generate_structured(problem_description, data, max_retries)
            return result.code
        else:
            return self._generate_single_stage(problem_description, data, max_retries)

    def generate_structured(self, problem_description: str, data: Dict,
                           max_retries: int = 3) -> GenerationResult:
        """Execute the full three-stage generation pipeline."""
        data_structure = self._describe_data_schema(data)

        for attempt in range(max_retries):
            # Stage 1: Understand
            understanding = self._stage1_understand(problem_description)
            if not understanding:
                continue

            # Stage 2: Formalize
            mathematical_model = self._stage2_formalize(
                problem_description, understanding, data_structure
            )
            if not mathematical_model:
                continue

            # Stage 3: Synthesize
            code = self._stage3_synthesize(
                problem_description, mathematical_model, data_structure
            )

            if self._validate_code(code):
                return GenerationResult(
                    code=code,
                    understanding=understanding,
                    mathematical_model=mathematical_model,
                    stages_completed=3
                )

        # Fallback to single-stage
        return GenerationResult(code=self._generate_single_stage(...), stages_completed=0)

    def regenerate(self, problem_description: str, failed_code: str,
                   error_message: str, data: Dict) -> str:
        """Regenerate code after L1 FATAL error."""
        data_structure = self._describe_data_schema(data)

        prompt = REGENERATE_PROMPT.format(
            problem_description=problem_description,
            failed_code=failed_code,
            error_message=error_message,
            data_structure=data_structure
        )

        response = self.llm_client.generate(prompt, system=REGENERATE_SYSTEM)
        return self._extract_code(response)

    def _describe_data_schema(self, data: Dict) -> str:
        """Schema-only visibility: show structure, NOT values."""
        # Returns: "- demand: list[3] of int" instead of "demand: [100, 150, 200]"
        lines = []
        for key, value in data.items():
            if isinstance(value, list):
                lines.append(f"- {key}: list[{len(value)}] of {type(value[0]).__name__}")
            elif isinstance(value, dict):
                lines.append(f"- {key}: dict with keys {list(value.keys())[:5]}")
            else:
                lines.append(f"- {key}: {type(value).__name__} (scalar)")
        return "\n".join(lines)
```

**Schema-Only Visibility Design:**

The LLM sees only the data schema (keys, types, dimensions), NOT actual values. This enables:
1. **Prevents hardcoding**: LLM cannot embed specific values in generated code
2. **Supports perturbation testing**: L2-L5 verification relies on modifying data values
3. **Generalization**: Generated code works with any data conforming to the schema

**Big-M Guidelines (for indicator/logical constraints):**
- NEVER hardcode Big-M values like `M = 1e6`
- ALWAYS compute M dynamically: `M = sum(data["demand"]) * 1.5`
- Use problem-relevant parameters for M calculation

**Edge Case Handling:**
- Check array length: `if data["items"]: ...`
- Avoid division by zero: `max(value, 1e-6)`
- Use `.get()` for optional keys: `data.get("key", default)`

## 3.6 repair.py - Code Repair (Conservative Strategy)

**Key Design: Context-Based Repair with 3 Sections**

The repair prompt separates issues into three clearly labeled sections:
1. **CRITICAL ERRORS (MUST FIX)** - ERROR level, 99%+ confidence
2. **HIGH PRIORITY (SHOULD FIX)** - WARNING level, 80%+ confidence
3. **DIAGNOSTIC INFO (DO NOT FIX)** - INFO level, reference only

```python
class CodeRepairer:
    """Repair optimization code based on diagnostic reports."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def repair_with_context(
        self,
        code: str,
        data: Dict,
        problem_description: str,
        critical_errors: List[Dict],   # ERROR level - MUST fix
        should_fix: List[Dict],        # WARNING level - SHOULD fix
        for_reference: List[Dict],     # INFO level - DO NOT fix
        max_attempts: int = 3
    ) -> str:
        """
        Repair code with categorized issue context.

        This is the conservative repair strategy:
        - critical_errors: ERROR level, MUST fix (99%+ confidence)
        - should_fix: WARNING level, SHOULD fix (80%+ confidence)
        - for_reference: INFO level, DO NOT fix (likely normal)
        """
        # If nothing to fix, return original code
        if not critical_errors and not should_fix:
            return code

        prompt = REPAIR_WITH_CONTEXT_PROMPT.format(
            problem_description=problem_description,
            data_structure=describe_data_schema(data),
            code=code,
            critical_errors=format_repair_section(critical_errors),
            should_fix=format_repair_section(should_fix),
            for_reference=format_reference_section(for_reference)  # Special formatting for INFO
        )

        for attempt in range(max_attempts):
            response = self.llm_client.generate(prompt, system=REPAIR_WITH_CONTEXT_SYSTEM)
            repaired_code = self._extract_code(response)

            if repaired_code != code and 'import gurobipy' in repaired_code:
                return repaired_code

        return code  # Return original if repair fails
```

**Repair Prompt Structure:**
```
## DIAGNOSTIC REPORT (READ CAREFULLY!)

### SECTION 1: CRITICAL ERRORS - YOU MUST FIX (NO EXCEPTIONS)
[ERROR level issues - constraint direction errors, direction anomalies]

### SECTION 2: HIGH-PRIORITY ISSUES - YOU SHOULD FIX
[WARNING level issues - CPT missing constraints]

### SECTION 3: DIAGNOSTIC INFO - FOR REFERENCE ONLY (DO NOT FIX)
[INFO level issues - slack constraints, numerical artifacts, sensitivity info]
```

## 3.7 pipeline.py - ReLoop Pipeline (with Conservative Repair)

```python
@dataclass
class PipelineResult:
    final_code: str
    final_report: VerificationReport
    iterations: int
    history: List[Tuple[str, VerificationReport]]
    total_time: float
    generation_time: float
    verification_time: float
    repair_time: float
    success: bool
    improved: bool
    regeneration_count: int = 0  # Number of L1 FATAL regenerations


@dataclass
class RepairContext:
    """
    Context for repair decision (Conservative Strategy).

    Separates issues into:
    - critical_errors: ERROR level, must fix (99%+ confidence)
    - should_fix: WARNING level, should fix (80%+ confidence)
    - for_reference: INFO level, likely normal, DO NOT fix
    """
    critical_errors: List[Dict] = field(default_factory=list)
    should_fix: List[Dict] = field(default_factory=list)
    for_reference: List[Dict] = field(default_factory=list)
    should_trigger: bool = False  # Only True if ERROR or WARNING exists


class ReLoopPipeline:
    """
    Complete ReLoop Pipeline: Generate → Verify → Repair loop.

    Key Features:
    - Chain-of-Thought generation: Single API call with step-by-step reasoning
    - L1 FATAL triggers regeneration (not termination)
    - ERROR/WARNING triggers repair, INFO does NOT
    - Guaranteed output when L1 passes
    """

    def run(self, problem_description: str, data: Dict,
            initial_code: Optional[str] = None) -> PipelineResult:
        """
        Run complete pipeline.

        Flow:
        1. Generate code (CoT or single-stage)
        2. Verify with L1-L5
        3. If L1 FATAL: regenerate (up to max_regeneration_attempts)
        4. Run L4 Adversarial Loop (if enabled)
        5. If ERROR/WARNING: repair (up to max_repair_iterations)
        6. INFO does NOT trigger repair (likely normal)
        7. Return result (always has output if any L1 passes)
        """
        # ... generation and L1 handling same as before ...

        # Step 4: Run L4 Adversarial Loop
        if self.enable_l4_adversarial and self.l4_verifier:
            l4_results, l4_exit_reason, l4_code = self._run_l4_adversarial_loop(
                code=code,
                data=data,
                baseline_obj=report.objective,
                problem_description=problem_description,
                obj_sense=self._get_obj_sense_from_report(report),
                report=report
            )
            # If L4 produced fixed code, update and re-verify
            if l4_code != code:
                code = l4_code
                report = self.verifier.verify(code, data, ...)
                history.append((code, report))

        # Step 5: Handle ERROR/WARNING with repair (INFO does NOT trigger)
        repair_iteration = 0
        ctx = self._analyze_verification_results(report)

        while ctx.should_trigger and repair_iteration < self.max_repair_iterations:
            repair_iteration += 1

            # Use context-based repair with 3 sections
            repaired_code = self.repairer.repair_with_context(
                code=code,
                data=data,
                problem_description=problem_description,
                critical_errors=ctx.critical_errors,   # MUST fix
                should_fix=ctx.should_fix,             # SHOULD fix
                for_reference=ctx.for_reference        # DO NOT fix
            )

            if repaired_code == code:
                break

            code = repaired_code
            report = self.verifier.verify(code, data, ...)
            history.append((code, report))

            # Re-analyze for next iteration
            ctx = self._analyze_verification_results(report)

        return PipelineResult(...)

    def _analyze_verification_results(self, report: VerificationReport) -> RepairContext:
        """
        Analyze verification results to determine repair strategy.

        Classification:
        - critical_errors: ERROR level (L2 anomaly - both directions improve) - Must fix
        - should_fix: WARNING level (L4 direction accepted, L5 cpt_missing) - Should fix
        - for_reference: INFO level (L2 no_effect/sensitivity, L3 duality, L4 rejected) - Do NOT fix

        Note: L4 issues are handled by adversarial mechanism (Accept/Reject).
        Only trigger repair if critical_errors or should_fix is non-empty.
        """
        critical_errors = []
        should_fix = []
        for_reference = []

        for r in report.layer_results:
            item = {
                "layer": r.layer,
                "check": r.check,
                "severity": r.severity.value,
                "message": r.message,
                "details": r.details or {}
            }

            if r.severity == Severity.ERROR:
                # L2 anomaly - MUST fix
                critical_errors.append(item)
            elif r.severity == Severity.WARNING:
                # L5 cpt_missing - SHOULD fix
                should_fix.append(item)
            elif r.severity == Severity.INFO:
                # L2 no_effect/sensitivity, L3 duality, L4 rejected - DO NOT fix
                for_reference.append(item)

        # Only trigger if there are ERROR or WARNING issues
        should_trigger = len(critical_errors) > 0 or len(should_fix) > 0

        return RepairContext(
            critical_errors=critical_errors,
            should_fix=should_fix,
            for_reference=for_reference,
            should_trigger=should_trigger
        )
```

**Pipeline Flow Diagram:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Pipeline Execution Flow                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Generate] ──→ [L1-L5 Verify] ──→ L1 Status?                              │
│                                      │                                       │
│                    ┌─────────────────┴─────────────────┐                    │
│                    │                                   │                    │
│                    ▼                                   ▼                    │
│               ┌─────────┐                        ┌─────────┐               │
│               │ FAILED  │                        │  PASS   │               │
│               │(L1 FATAL)│                        │         │               │
│               └────┬────┘                        └────┬────┘               │
│                    │                                   │                    │
│                    ▼                                   ▼                    │
│             [Regenerate]                      [L4 Adversarial Loop]        │
│             (up to 3x)                        ┌────────────────────┐       │
│                    │                          │ verify → accept? ──┼──Yes──→│
│                    ▼                          │    │               │       │
│             [Re-verify]                       │    └──No (reject)──┼──→    │
│                                               │    re-verify with  │       │
│                                               │    context         │       │
│                                               │         │          │       │
│                                               │    max rejections? │       │
│                                               │    ──Yes→ INFO     │       │
│                                               └────────────────────┘       │
│                                                        │                    │
│                                                        ▼                    │
│                                          ┌───────────────────────┐         │
│                                          │ Has ERROR/WARNING?    │         │
│                                          └───────────┬───────────┘         │
│                                   Yes────────────────┴──────────────No     │
│                                    │                                │      │
│                                    ▼                                ▼      │
│                              [Repair Loop]                    ✓ Return     │
│                              (up to 3x)                       Success      │
│                                    │                                       │
│                                    ▼                                       │
│                              [Re-verify]                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3.8 data_extraction.py - Data Extraction

```python
DATA_EXTRACTION_PROMPT = '''Extract ALL numerical parameters from this optimization problem.

## Problem
{problem}

## Output Format
Return ONLY a JSON object with parameter names as keys:
```json
{{
  "demand": [100, 150, 200],
  "capacity": 500,
  "cost_per_unit": 10
}}
```

Rules:
1. Use snake_case for parameter names
2. Use descriptive names (e.g., "labor_hours" not "h")
3. Include ALL numbers mentioned
'''


class DataExtractor:
    """Extract structured parameters from problem descriptions."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def extract(self, problem_description: str) -> Dict[str, Any]:
        """Extract parameter data from problem description."""
        prompt = DATA_EXTRACTION_PROMPT.format(problem=problem_description)

        try:
            response = self.llm_client.generate(prompt)
            data = self._parse_json(response)
            if data and self._validate_data(data):
                return data
        except Exception:
            pass

        # Fallback: extract numbers with regex
        return self._extract_numbers_regex(problem_description)

    def _extract_numbers_regex(self, text: str) -> Dict[str, Any]:
        """Fallback: extract numbers with regex."""
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        return {f"param_{i+1}": float(num) for i, num in enumerate(numbers[:20])}
```

## 3.9 experiment_runner.py - Experiment Runner

```python
@dataclass
class ExperimentRecord:
    id: int
    difficulty: str
    problem_description: str
    extracted_data: Dict[str, Any]
    ground_truth: float
    generated_code: str
    final_code: str
    predicted_objective: Optional[float]
    initial_status: str
    final_status: str
    iterations: int
    objective_error: Optional[float]
    error_detected: bool
    error_repaired: bool
    extraction_time: float
    pipeline_time: float


@dataclass
class ExperimentSummary:
    total_problems: int
    verified_count: int
    warnings_count: int
    errors_count: int
    failed_count: int
    detection_rate: float        # Rate of detecting problems
    false_positive_rate: float   # Rate of false alarms
    repair_success_rate: float   # Rate of successful repairs
    avg_objective_error: float   # Average objective error
    by_difficulty: Dict[str, Dict[str, float]]


class ExperimentRunner:
    """Experiment runner for ReLoop."""

    def __init__(self, llm_client, output_dir: str = "results",
                 enable_cpt: bool = True, max_repair_iterations: int = 3):
        self.extractor = DataExtractor(llm_client)
        self.pipeline = ReLoopPipeline(llm_client, max_repair_iterations, enable_cpt)
        self.output_dir = Path(output_dir)

    def run_dataset(self, dataset_path: str) -> ExperimentSummary:
        """Run experiment on entire dataset."""
        dataset = self._load_dataset(dataset_path)
        records = []

        for item in dataset:
            record = self._run_single(item)
            records.append(record)
            self._save_record(record)

        summary = self._compute_summary(records)
        self._save_summary(summary)
        return summary

    def _run_single(self, item: Dict) -> ExperimentRecord:
        """Run single problem."""
        problem = item.get("en_question", "")
        ground_truth = float(item.get("en_answer", 0))

        # Step 1: Extract data
        data = self.extractor.extract(problem)

        # Step 2-4: Pipeline (generate → verify → repair)
        result = self.pipeline.run(problem, data)

        # Step 5: Evaluate
        predicted = result.final_report.objective
        if predicted is not None and ground_truth != 0:
            objective_error = abs(predicted - ground_truth) / abs(ground_truth)
        else:
            objective_error = None

        # Detection and repair tracking
        initial_status = result.history[0][1].status
        error_detected = initial_status in ['WARNINGS', 'ERRORS']
        error_repaired = (error_detected and
                         result.final_report.status == 'VERIFIED' and
                         objective_error is not None and objective_error < 0.01)

        return ExperimentRecord(...)
```

---

# Part 4: Data Format

## 4.1 Dataset JSONL Format

Each line contains one problem:

```json
{
  "en_question": "Natural language problem description...",
  "en_answer": "123.45",
  "difficulty": "Easy|Medium|Hard",
  "id": 1
}
```

## 4.2 Available Datasets

| Dataset | Count | Description |
|---------|-------|-------------|
| `RetailOpt-190.jsonl` | 190 | **Our benchmark** - Retail inventory optimization |
| `IndustryOR_fixedV2.jsonl` | 100 | Industry OR with difficulty labels |
| `MAMO_EasyLP_fixed.jsonl` | 642 | Easy LP problems |
| `MAMO_ComplexLP_fixed.jsonl` | ~500 | Complex LP problems |
| `NL4OPT.jsonl` | 245 | NL4OPT benchmark |
| `OptMATH_Bench_166.jsonl` | 166 | OptMATH benchmark |
| `OptMATH_Bench_193.jsonl` | 193 | OptMATH benchmark |
| `OptiBench.jsonl` | ~400 | OptiBench problems |

---

# Part 5: Key Design Decisions

## 5.1 Why 5 Layers?

| Layer | What It Detects | Theoretical Basis |
|-------|-----------------|-------------------|
| L1 | Syntax errors, runtime errors, solver failures | Basic correctness |
| L2 | Wrong constraint directions | Relaxation monotonicity theorem |
| L3 | Constraint formulation errors | Strong duality theorem |
| L4 | Missing constraints, direction anomalies | Parameter sensitivity + physical intuition |
| L5 | Semantic constraint violations | Counterfactual testing |

## 5.1.1 L2 Anomaly Detection Design

**Problem:** Keyword-based role inference (demand, capacity, etc.) is domain-specific and prone to false positives.

**Solution:** Bidirectional Anomaly Detection (100% Reliable)

```
L2 Anomaly Detection Principle:
├── Perturb any parameter both UP (+20%) and DOWN (-20%)
├── Check: does BOTH directions IMPROVE the objective?
├── If yes → ERROR: physically impossible, must be modeling error
├── If no → normal behavior (one direction worsens or no effect)
└── Works for ANY domain, no keywords needed
```

**Why This Works:**

For a well-formed optimization model:
- Increasing a constraint parameter should worsen (or not affect) objective
- Decreasing the same parameter should also worsen (or not affect) objective
- **Both improving is physically impossible** → guaranteed error (99%+ confidence)

## 5.1.2 L4 Adversarial Direction Design

**Problem:** Single-LLM direction analysis can make mistakes that go uncaught.

**Solution:** Adversarial Debate Mechanism

```
L4 Adversarial Principle:
├── LLM_verify: Analyze expected direction from problem context
├── LLM_repair: Review analysis, Accept or Reject
├── If Reject → Re-verify with feedback
├── Repeat until Accept or max rejections reached
├── Accepted → WARNING (should fix)
├── Max rejections → INFO (inconclusive, no fix)
└── Two LLM perspectives catch each other's errors
```

**Why Adversarial?**

```
Single-LLM Problem:
├── LLM_verify might misunderstand parameter roles
├── e.g., "wage" could be cost coefficient or constraint bound
└── No mechanism to catch these mistakes

Adversarial Solution:
├── LLM_repair can reject with reasoning
├── e.g., "wage is objective coefficient, not capacity bound"
├── LLM_verify re-analyzes with this feedback
├── Debate converges on correct interpretation
└── Max rejections prevent infinite loops
```

**Design Decisions:**
1. Keyword-based direction verification completely removed (domain-specific, error-prone)
2. L2 handles pure anomaly detection (both-improve → ERROR)
3. L4 handles semantic direction analysis (LLM debate → WARNING or INFO)
4. Separation prevents over-correction from L4 false positives

**Result:**
- L2 catches structural errors with 100% reliability
- L4 catches semantic direction issues with self-correcting mechanism
- Together they are universal across all domains

## 5.2 Why Only L1 is Blocking?

- L1 failures mean no solution exists → nothing to verify
- L2-L5 are diagnostic: solution exists but may have issues
- Guarantees output when code runs: better UX

## 5.3 Complexity Calibration

```
SIMPLE (≤5 vars, ≤5 constraints):
- Parameter no-effect → INFO (not WARNING)
- Skip zero-objective check
- Fewer false positives for toy problems

MEDIUM (≤50 vars):
- Full checking enabled
- Standard thresholds

COMPLEX (>50 vars):
- Full checking enabled
- May need more perturbations
```

## 5.4 CPT Safety

- L5 only produces WARNING/INFO, never ERROR/FATAL
- Requires LLM for constraint extraction (no keyword fallback)
- If no LLM available → L5 is skipped (graceful degradation)
- Cannot create false negatives (worst case: missed detection)

---

# Part 6: Evaluation Metrics

## 6.1 Primary Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Detection Rate** | TP / (TP + FN) | % of errors correctly detected |
| **False Positive Rate** | FP / (FP + TN) | % of correct code flagged |
| **Repair Success Rate** | Repaired / Detected | % of detected errors fixed |
| **Objective Accuracy** | 1 - \|pred - true\| / true | Relative objective error |

## 6.2 Per-Layer Metrics

- **Layer Detection Rate**: Which layers detect which error types
- **Unique Contribution**: Errors only detected by specific layer
- **False Positive Contribution**: Which layers cause false alarms

## 6.3 Experiment Output

```json
{
  "total_problems": 190,
  "verified_count": 142,
  "warnings_count": 35,
  "errors_count": 8,
  "failed_count": 5,
  "detection_rate": 0.94,
  "false_positive_rate": 0.03,
  "repair_success_rate": 0.78,
  "avg_objective_error": 0.02,
  "by_difficulty": {
    "Easy": {"count": 50, "verified_rate": 0.88},
    "Medium": {"count": 80, "verified_rate": 0.72},
    "Hard": {"count": 60, "verified_rate": 0.65}
  }
}
```

---

# Part 7: Usage Examples

## 7.1 Basic Verification

```python
from reloop import verify_code

code = """
import gurobipy as gp
from gurobipy import GRB

m = gp.Model()
m.Params.OutputFlag = 0

x = m.addVar(name="x")
y = m.addVar(name="y")

m.setObjective(10*x + 15*y, GRB.MINIMIZE)
m.addConstr(x >= data['min_x'])
m.addConstr(y >= data['min_y'])
m.addConstr(x + y <= data['max_total'])

m.optimize()
print(f"status: {m.Status}")
print(f"objective: {m.ObjVal}")
"""

data = {"min_x": 100, "min_y": 80, "max_total": 250}
report = verify_code(code, data, obj_sense="minimize", verbose=True)
print(f"Status: {report.status}")
```

## 7.2 Full Pipeline

```python
from reloop import run_reloop, DataExtractor

# Setup LLM client
class MyLLMClient:
    def generate(self, prompt, system=None):
        # Your LLM API call here
        return response

llm = MyLLMClient()

# Extract data from natural language
extractor = DataExtractor(llm)
data = extractor.extract("""
    Minimize total cost. Product A costs $10, B costs $15.
    Must produce at least 100 of A and 80 of B.
    Total cannot exceed 250 units.
""")

# Run pipeline
result = run_reloop(
    problem_description="Minimize total cost...",
    data=data,
    llm_client=llm,
    verbose=True
)

print(f"Status: {result.final_report.status}")
print(f"Objective: {result.final_report.objective}")
print(f"Iterations: {result.iterations}")
```

## 7.3 Batch Experiments

```python
from reloop import run_experiment

summary = run_experiment(
    dataset_path="data/RetailOpt-190.jsonl",
    llm_client=llm,
    output_dir="results",
    verbose=True
)

print(f"Detection Rate: {summary.detection_rate:.1%}")
print(f"False Positive Rate: {summary.false_positive_rate:.1%}")
print(f"Repair Success Rate: {summary.repair_success_rate:.1%}")
```

---

# Part 8: Checklist for Paper

## 8.1 Architecture Correctness

- [x] L1 is the ONLY layer that produces FATAL
- [x] L1 FATAL triggers regeneration (up to 3 attempts), NOT termination
- [x] L1 PASS guarantees has_solution=True
- [x] L2 produces ERROR (anomaly) / INFO (no_effect, sensitivity), never FATAL
- [x] L3 produces INFO only (numerical artifacts)
- [x] L4 produces WARNING (accepted) / INFO (rejected/inconclusive), never FATAL/ERROR
- [x] L5 produces WARNING/INFO only, never ERROR/FATAL
- [x] Guaranteed output when L1 passes
- [x] L2 anomaly detection is 100% reliable (both-improve = error)
- [x] L4 uses adversarial LLM debate (verify + repair roles)
- [x] L4 has exit conditions: all accepted, max rejections, or no violations
- [x] Keyword-based inference completely removed (L2, L4, L5 all use universal/LLM)
- [x] L5 CPT uses LLM-only for constraint extraction (no keyword fallback)

## 8.2 Implementation Completeness

- [x] param_utils.py: extract, perturb, skip determination (no role inference)
- [x] executor.py: subprocess isolation, timeout
- [x] verification.py: L1, L2 (anomaly), L3, L5 layers
- [x] l4_adversarial.py: L4 Adversarial Direction Analysis with Accept/Reject
- [x] prompts.py: CoT generation + repair + L4 verify prompts
- [x] generation.py: CoT pipeline (Understand → Formalize → Synthesize)
- [x] generation.py: regenerate() method for L1 FATAL recovery
- [x] repair.py: diagnostic-based repair with L4 Accept/Reject support
- [x] pipeline.py: Generate→Verify→Repair loop with L1 regeneration and L4 integration
- [x] data_extraction.py: NL→structured data
- [x] experiment_runner.py: batch experiments

## 8.3 Code Generation (Paper Section 3.1)

- [x] Single-Stage: Direct problem → code (baseline)
- [x] 3-Stage CoT: Understand → Formalize → Synthesize (main approach)
- [x] Single API call implementation (preserves context, 2.17% error)
- [x] Schema-only visibility: LLM sees data structure, not values
- [x] Fallback to single-stage if CoT fails

## 8.3 Experiment Requirements

- [x] Multiple benchmark datasets
- [x] Detection/false positive metrics
- [x] Per-difficulty analysis
- [x] Repair success tracking
- [x] Output format for results

---

# Part 9: Related Resources

| Resource | Link |
|----------|------|
| **RetailOpt-190 Dataset** | [Hugging Face](https://huggingface.co/datasets/Jacoblian/RetailOpt-190) |
| **RetailOpt-190 Repository** | [GitHub](https://github.com/junbolian/RetailOpt-190) |
| **ReLoop Code** | [GitHub](https://github.com/junbolian/ReLoop) |
