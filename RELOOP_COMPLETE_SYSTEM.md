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
1. **Relaxation Monotonicity**: Tightening constraints should worsen objective
2. **Strong Duality**: Primal and dual objectives should match at optimum
3. **Parameter Effectiveness**: Meaningful parameters should affect objective
4. **Constraint Presence Testing**: Missing constraints can be detected via counterfactual testing

## 1.3 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ReLoop Complete Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌───────────┐ │
│  │   Problem   │     │    Code     │     │  Behavior   │     │  Repair   │ │
│  │ Description │ ──> │ Generation  │ ──> │ Verification│ ──> │  Module   │ │
│  │   + Data    │     │   (LLM)     │     │  (ReLoop)   │     │   (LLM)   │ │
│  └─────────────┘     └─────────────┘     └─────────────┘     └───────────┘ │
│         │                   │                   │                   │       │
│         │                   ▼                   ▼                   ▼       │
│         │            ┌─────────────┐     ┌─────────────┐     ┌───────────┐ │
│         │            │   Gurobi    │     │ Diagnostic  │     │  Repaired │ │
│         │            │    Code     │     │   Report    │     │   Code    │ │
│         │            └─────────────┘     └─────────────┘     └───────────┘ │
│         │                                                                   │
│         ▼                                                                   │
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
│  │ L2: Relaxation Monotonicity (DIAGNOSTIC)                            │   │
│  │     • Tighten parameters by 20% (delta=0.2)                         │   │
│  │     • Check: objective should worsen                                │   │
│  │     • Violation → WARNING (constraint direction error)              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ L3: Dual Consistency (DIAGNOSTIC)                                   │   │
│  │     • Check primal-dual gap (threshold: 1%)                         │   │
│  │     • Check shadow price signs                                      │   │
│  │     • Large gap → WARNING (numerical or constraint issue)           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ L4: Solution Freedom Analysis (DIAGNOSTIC)                          │   │
│  │     • Perturb parameters ±20%                                       │   │
│  │     • Check: should have effect on objective                        │   │
│  │     • No effect → WARNING (constraint may be missing)               │   │
│  │     • Zero objective check (threshold: 1e-2)                        │   │
│  │     • Complexity calibration: SIMPLE problems get INFO not WARNING  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ L5: CPT - Constraint Perturbation Testing (ENHANCEMENT, Optional)   │   │
│  │     • LLM or rule-based candidate constraint extraction             │   │
│  │     • Test each: perturb parameters, re-solve, compare objectives   │   │
│  │     • Objective unchanged → WARNING (constraint was missing)        │   │
│  │     • Safe: only WARNING/INFO, never ERROR/FATAL                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 1.4 Severity Levels

| Level | Meaning | Source Layers | Effect |
|-------|---------|---------------|--------|
| **FATAL** | Definite error | L1 only | Blocks output, no solution |
| **ERROR** | High-confidence issue | L2-L4 | Has output, low confidence |
| **WARNING** | Potential issue | All layers | Has output, needs review |
| **INFO** | Informational | All layers | Reference only |
| **PASS** | Check passed | All layers | No issue |

## 1.5 Guaranteed Output Mechanism

```
CORE GUARANTEE:

If L1 passes (code runs + solver has solution):
├── Always return objective value
├── Always return solution vector
├── Always return diagnostics
└── L2-L5 findings do NOT block output

Diagnostic layers ADD information, never DELETE output.
```

---

# Part 2: Project Structure

## 2.1 Actual File Structure

```
reloop/
├── reloop/                       # Core package
│   ├── __init__.py               # Package exports (all public APIs)
│   ├── param_utils.py            # Parameter extraction & perturbation
│   ├── executor.py               # Isolated subprocess execution
│   ├── verification.py           # 5-layer verification engine
│   ├── prompts.py                # LLM prompt templates
│   ├── generation.py             # Code generation
│   ├── repair.py                 # Diagnostic-based repair
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
| `param_utils.py` | ~200 | Parameter extraction, perturbation, role inference |
| `executor.py` | ~120 | Subprocess-isolated code execution with timeout |
| `verification.py` | ~650 | Core 5-layer verification engine |
| `prompts.py` | ~75 | LLM prompt templates for generation and repair |
| `generation.py` | ~100 | Code generation from problem description |
| `repair.py` | ~90 | Diagnostic-guided code repair |
| `pipeline.py` | ~200 | Generate→Verify→Repair pipeline orchestration |
| `data_extraction.py` | ~130 | Extract structured data from natural language |
| `experiment_runner.py` | ~330 | Batch experiment runner with metrics |

---

# Part 3: Core Module Implementations

## 3.1 param_utils.py - Parameter Utilities

**Key Classes and Functions:**

```python
class ParameterRole(Enum):
    CAPACITY = "capacity"      # Upper bound (e.g., machine_capacity)
    REQUIREMENT = "requirement" # Lower bound (e.g., demand)
    COST = "cost"              # Objective coefficient
    OTHER = "other"

def extract_numeric_params(data: Dict, prefix: str = "") -> List[str]:
    """Recursively extract all numeric parameters from data dict."""
    # Returns: ["demand", "capacity", "cost.production", ...]

def perturb_param(data: Dict, param_path: str, factor: float) -> Dict:
    """Create copy of data with param multiplied by factor."""
    # factor > 1: increase, factor < 1: decrease

def infer_param_role(param_name: str) -> ParameterRole:
    """Infer parameter role from name keywords."""
    # "capacity", "max", "limit" → CAPACITY
    # "demand", "min", "requirement" → REQUIREMENT
    # "cost", "price" → COST

def should_skip_param(param_name: str, value: Any) -> bool:
    """Skip zero values and Big-M parameters."""
    # Skip if value is 0 or name contains "big_m", "bigm"

def get_expected_direction(param_name: str, obj_sense: str) -> str:
    """Get expected objective change direction when tightening param."""
    # CAPACITY tightened → objective should worsen
    # REQUIREMENT tightened → objective should worsen
```

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

### Layer 2: Relaxation Monotonicity

```python
def _layer2(self, code: str, data: Dict, baseline_obj: float,
            obj_sense: str, params: List[str], verbose: bool):
    """L2: Relaxation Monotonicity (DIAGNOSTIC)"""
    results = []

    for param in params[:self.max_params]:
        if should_skip_param(param, get_param_value(data, param)):
            continue

        # Tighten parameter by 20%
        tightened_data = perturb_param(data, param, 1.0 - self.delta)
        tight_result = self.executor.execute(code, tightened_data)
        tight_obj = tight_result.get("objective")

        if tight_obj is not None:
            # Check direction
            expected = get_expected_direction(param, obj_sense)

            if obj_sense == "minimize":
                # Tightening should WORSEN (increase) objective
                if tight_obj < baseline_obj * (1 - self.epsilon):
                    results.append(LayerResult(
                        "L2", "monotonicity", Severity.WARNING,
                        f"Tightening '{param}' improved objective (wrong direction?)",
                        0.75, {"param": param, "baseline": baseline_obj, "tight": tight_obj}
                    ))

    if not results:
        results.append(LayerResult("L2", "monotonicity", Severity.PASS, "OK", 0.9))

    return results
```

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
                "L3", "duality_gap", Severity.WARNING,
                f"Primal-dual gap {gap:.2%} exceeds threshold",
                0.7, {"gap": gap}
            ))
        else:
            results.append(LayerResult("L3", "duality", Severity.PASS, "OK", 0.9))
    else:
        results.append(LayerResult(
            "L3", "duality", Severity.INFO, "Dual objective not available", 0.5
        ))

    return results
```

### Layer 4: Solution Freedom Analysis

```python
def _layer4(self, code: str, data: Dict, params: List[str],
            baseline_obj: float, obj_sense: str, complexity: Complexity, verbose: bool):
    """L4: Solution Freedom Analysis (DIAGNOSTIC)"""
    results = []

    # Zero objective check (only for non-SIMPLE)
    if complexity != Complexity.SIMPLE and abs(baseline_obj) < 1e-2:
        results.append(LayerResult(
            "L4", "zero_objective", Severity.WARNING,
            f"Objective is ~0 ({baseline_obj:.4f}). Missing cost terms?",
            0.7
        ))

    # Parameter effect testing
    for param in params[:self.max_params]:
        if should_skip_param(param, get_param_value(data, param)):
            continue

        # Test both directions
        up_data = perturb_param(data, param, 1.0 + self.delta)
        down_data = perturb_param(data, param, 1.0 - self.delta)

        up_result = self.executor.execute(code, up_data)
        down_result = self.executor.execute(code, down_data)

        up_obj = up_result.get("objective")
        down_obj = down_result.get("objective")

        if up_obj is not None and down_obj is not None:
            # Check if parameter has effect
            change = max(abs(up_obj - baseline_obj), abs(down_obj - baseline_obj))
            relative_change = change / max(abs(baseline_obj), 1.0)

            if relative_change < 0.001:  # No effect
                severity = Severity.INFO if complexity == Complexity.SIMPLE else Severity.WARNING
                results.append(LayerResult(
                    "L4", "param_effect", severity,
                    f"Parameter '{param}' has no effect on objective",
                    0.6, {"param": param}
                ))

    if not results:
        results.append(LayerResult("L4", "freedom", Severity.PASS, "OK", 0.85))

    return results
```

### Layer 5: CPT (Constraint Perturbation Testing)

```python
def _layer5(self, code: str, data: Dict, baseline_obj: float,
            obj_sense: str, problem_description: str, verbose: bool):
    """L5: CPT (ENHANCEMENT) - Only WARNING/INFO, never ERROR/FATAL"""
    results = []

    # Extract candidate constraints
    if self.llm_client:
        candidates = self._cpt_extract_candidates(problem_description, data)
    else:
        candidates = self._cpt_extract_candidates_rule_based(data)

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

def _cpt_extract_candidates_rule_based(self, data: Dict) -> List[Dict]:
    """Rule-based extraction when no LLM available."""
    candidates = []
    for param in extract_numeric_params(data):
        role = infer_param_role(param)
        if role == ParameterRole.CAPACITY:
            candidates.append({
                "description": f"Should not exceed {param}",
                "type": "capacity",
                "parameters": [param]
            })
        elif role == ParameterRole.REQUIREMENT:
            candidates.append({
                "description": f"Should meet {param}",
                "type": "demand",
                "parameters": [param]
            })
    return candidates

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
        if change_ratio > 0.5:
            return {"status": "SATISFIED"}
        else:
            return {"status": "MISSING", "description": candidate.get("description")}

    return {"status": "SKIP"}
```

## 3.4 prompts.py - LLM Prompts

**Actual Implementation:**

```python
CODE_GENERATION_PROMPT = '''You are an optimization expert. Generate Gurobi Python code.

## Problem
{problem_description}

## Data Structure
{data_structure}

## Requirements
1. Use `gurobipy` library, model named `m`
2. Access data via `data` dictionary
3. Set `m.Params.OutputFlag = 0`
4. Print output format:
   print(f"status: {{m.Status}}")
   print(f"objective: {{m.ObjVal}}")

Return ONLY Python code.
'''

CODE_GENERATION_SYSTEM = '''You write correct Gurobi Python code. Include ALL constraints.'''

REPAIR_PROMPT = '''Fix this optimization code based on the diagnostic report.

## Problem
{problem_description}

## Code
```python
{code}
```

## Diagnostic Report
{diagnostic_report}

## Issues
{issues}

Return the COMPLETE fixed code.
'''

REPAIR_SYSTEM = '''You debug optimization models. Fix each issue in the report.'''


def format_diagnostic_report(report) -> str:
    """Format diagnostic report for display."""
    lines = [f"Status: {report.status}", f"Confidence: {report.confidence:.2f}", "", "Issues:"]
    for r in report.layer_results:
        if r.severity.value in ['FATAL', 'ERROR', 'WARNING']:
            lines.append(f"  [{r.layer}] {r.severity.value}: {r.message}")
    return "\n".join(lines)


def format_issues_for_repair(report) -> str:
    """Extract actionable issues for repair."""
    issues = []
    for r in report.layer_results:
        if r.severity.value == 'WARNING':
            if 'monotonicity' in r.check:
                issues.append(f"CONSTRAINT DIRECTION ERROR: {r.message}")
            elif 'param_effect' in r.check:
                issues.append(f"MISSING CONSTRAINT: {r.message}")
            elif 'cpt_missing' in r.check:
                issues.append(f"CPT DETECTED MISSING: {r.message}")
    return "\n".join(issues) if issues else "No specific issues."
```

## 3.5 generation.py - Code Generation

```python
class CodeGenerator:
    """Generate optimization code using LLM."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate(self, problem_description: str, data: Dict, max_retries: int = 3) -> str:
        """Generate Gurobi code for the given problem."""
        data_structure = self._describe_data(data)

        prompt = CODE_GENERATION_PROMPT.format(
            problem_description=problem_description,
            data_structure=data_structure
        )

        for attempt in range(max_retries):
            response = self.llm_client.generate(prompt, system=CODE_GENERATION_SYSTEM)
            code = self._extract_code(response)

            if self._validate_code(code):
                return code

        raise ValueError("Failed to generate valid code")

    def _validate_code(self, code: str) -> bool:
        """Basic validation of generated code."""
        required = [r'import\s+gurobipy', r'\.addVar|\.addVars',
                   r'\.setObjective', r'\.optimize\s*\(\s*\)']
        return all(re.search(p, code) for p in required)
```

## 3.6 repair.py - Code Repair

```python
class CodeRepairer:
    """Repair optimization code based on diagnostic reports."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def repair(self, code: str, data: Dict, report: VerificationReport,
               problem_description: str, max_attempts: int = 3) -> str:
        """Repair code based on diagnostic report."""
        if report.status == 'VERIFIED':
            return code

        diagnostic_report = format_diagnostic_report(report)
        issues = format_issues_for_repair(report)

        prompt = REPAIR_PROMPT.format(
            problem_description=problem_description,
            code=code,
            diagnostic_report=diagnostic_report,
            issues=issues
        )

        for attempt in range(max_attempts):
            response = self.llm_client.generate(prompt, system=REPAIR_SYSTEM)
            repaired_code = self._extract_code(response)

            # Validate repair is different and valid
            if repaired_code != code and 'import gurobipy' in repaired_code:
                return repaired_code

        return code  # Return original if repair fails
```

## 3.7 pipeline.py - ReLoop Pipeline

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


class ReLoopPipeline:
    """Complete ReLoop Pipeline: Generate → Verify → Repair loop."""

    def __init__(self, llm_client, max_repair_iterations: int = 3,
                 enable_cpt: bool = True, verbose: bool = False):
        self.generator = CodeGenerator(llm_client)
        self.verifier = ReLoopVerifier(llm_client=llm_client)
        self.repairer = CodeRepairer(llm_client)
        self.max_repair_iterations = max_repair_iterations
        self.enable_cpt = enable_cpt
        self.verbose = verbose

    def run(self, problem_description: str, data: Dict,
            initial_code: Optional[str] = None) -> PipelineResult:
        """Run complete pipeline."""

        # Step 1: Generate code (or use provided)
        if initial_code:
            code = initial_code
        else:
            code = self.generator.generate(problem_description, data)

        # Step 2: Initial verification
        report = self.verifier.verify(
            code, data,
            problem_description=problem_description,
            enable_cpt=self.enable_cpt
        )
        history = [(code, report)]

        # Step 3: Repair loop
        iteration = 0
        while self._needs_repair(report) and iteration < self.max_repair_iterations:
            iteration += 1

            repaired_code = self.repairer.repair(code, data, report, problem_description)

            if repaired_code == code:
                break  # No changes, stop

            code = repaired_code
            report = self.verifier.verify(code, data, ...)
            history.append((code, report))

        return PipelineResult(
            final_code=code,
            final_report=report,
            iterations=len(history),
            history=history,
            success=report.status in ['VERIFIED', 'WARNINGS'],
            improved=self._is_improved(history[0][1], report)
        )

    def _needs_repair(self, report):
        return report.status in ['WARNINGS', 'ERRORS']

    def _is_improved(self, before, after):
        order = {'FAILED': 0, 'ERRORS': 1, 'WARNINGS': 2, 'VERIFIED': 3}
        return order.get(after.status, 0) > order.get(before.status, 0)
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
| L4 | Missing constraints | Parameter sensitivity analysis |
| L5 | Semantic constraint violations | Counterfactual testing |

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
- LLM extraction failure → silently skip (graceful degradation)
- Rule-based fallback when no LLM available
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
- [x] L1 PASS guarantees has_solution=True
- [x] L2-L4 produce WARNING/ERROR, never FATAL
- [x] L5 produces WARNING/INFO only, never ERROR/FATAL
- [x] Guaranteed output when L1 passes

## 8.2 Implementation Completeness

- [x] param_utils.py: extract, perturb, role inference
- [x] executor.py: subprocess isolation, timeout
- [x] verification.py: all 5 layers
- [x] prompts.py: generation and repair prompts
- [x] generation.py: code generation
- [x] repair.py: diagnostic-based repair
- [x] pipeline.py: Generate→Verify→Repair loop
- [x] data_extraction.py: NL→structured data
- [x] experiment_runner.py: batch experiments

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
