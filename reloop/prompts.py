"""ReLoop LLM Prompt Templates

Generation Approaches:
  1. Chain-of-Thought (CoT): Single API call with 3-stage reasoning
  2. Single-Stage: Direct problem → code generation
  3. Legacy 3-Stage: Separate API calls (deprecated, kept for compatibility)
"""

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
  **Variable Type**: For each variable, explicitly decide CONTINUOUS, INTEGER, or BINARY.
  Look for context where fractional values would be physically meaningless
  (e.g., number of trucks, workers to hire, items to select).
  State your choice and reasoning.
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

# ============================================================================
# Single-Stage Generation (Simple direct approach)
# ============================================================================

CODE_GENERATION_SYSTEM = '''You write correct Gurobi Python code. Include ALL constraints from the problem.'''

CODE_GENERATION_PROMPT = '''Generate Gurobi Python code for this optimization problem.

## Problem
{problem_description}

## Data Structure
The `data` variable is PRE-DEFINED with these keys:
{data_structure}

## Requirements
1. Use `gurobipy` library, model named `m`
2. **CRITICAL: Do NOT create `data = {{...}}`. The data variable already exists. Just use `data["key"]`.**
3. Set `m.Params.OutputFlag = 0`
4. Print: `print(f"status: {{m.Status}}")` and `print(f"objective: {{m.ObjVal}}")`
5. Implement ALL constraints mentioned in the problem
6. For Big-M constraints: compute M dynamically from data (e.g., `M = sum(data["demand"]) * 1.5`), NEVER hardcode
7. Handle edge cases: check array length, avoid division by zero

Return ONLY Python code. Do NOT include any `data = ` definition.
'''

# ============================================================================
# Legacy: Separate Stage Prompts (deprecated, kept for compatibility)
# ============================================================================

UNDERSTAND_SYSTEM = '''You are an optimization expert. Analyze optimization problems to identify key components.'''

UNDERSTAND_PROMPT = '''Analyze this optimization problem and extract the key components.

## Problem
{problem_description}

## Task
Identify and describe:
1. **Objective**: What are we optimizing? (minimize cost, maximize profit, etc.)
2. **Decision Scope**: What decisions need to be made? (quantities, assignments, schedules, etc.)
3. **Constraints**: What limitations or requirements exist?
4. **Key Parameters**: What numerical data is provided?

## Output Format
```yaml
objective:
  sense: minimize|maximize
  description: "brief description of what to optimize"

decisions:
  - name: "decision variable description"
    type: continuous|integer|binary
    indexed_by: "what dimensions (products, periods, locations, etc.)"

constraints:
  - name: "constraint name"
    type: capacity|demand|balance|logical|bound
    description: "what this constraint enforces"

parameters:
  - name: "parameter name"
    role: cost|revenue|capacity|requirement|other
    description: "what this parameter represents"
```

Return ONLY the YAML, no explanation.
'''

# ============================================================================
# Stage 2: Formalize
# ============================================================================

FORMALIZE_SYSTEM = '''You are a mathematical optimization modeler. Transform problem understanding into formal mathematical specifications.'''

FORMALIZE_PROMPT = '''Transform this problem understanding into a formal mathematical specification.

## Problem Description
{problem_description}

## Problem Understanding
{understanding}

## Data Structure
{data_structure}

## Task
Create a formal mathematical model with these 5 components:

### S1: Index Sets (I)
Define all index sets (products, periods, locations, etc.)

### S2: Parameters (P)
List all parameters with their dimensions and meanings.
Use exact keys from the data structure.

### S3: Decision Variables (V)
Define decision variables with domains and bounds.
**Variable Type**: For each variable, explicitly decide CONTINUOUS, INTEGER, or BINARY.
Look for context where fractional values would be physically meaningless
(e.g., number of trucks, workers to hire, items to select).
State your choice and reasoning.

### S4: Constraints (C)
Write each constraint in mathematical notation.
Use LaTeX-style notation: \\sum, \\forall, \\leq, \\geq

### S5: Objective Function (f)
Write the objective function to minimize or maximize.

## Output Format
```
SETS:
  I = {{...}}  # description

PARAMETERS:
  param_name[indices]: description

VARIABLES:
  var_name[indices] >= 0: description

CONSTRAINTS:
  constraint_name: mathematical_expression \\forall conditions

OBJECTIVE:
  minimize/maximize: expression
```

Return ONLY the mathematical specification.
'''

# ============================================================================
# Stage 3: Synthesize
# ============================================================================

SYNTHESIZE_SYSTEM = '''You write correct Gurobi Python code. Implement the mathematical model exactly as specified.'''

SYNTHESIZE_PROMPT = '''Generate Gurobi Python code from this mathematical specification.

## Problem Description
{problem_description}

## Mathematical Model
{mathematical_model}

## Data Structure
{data_structure}

## Requirements
1. Use `gurobipy` library
2. Model variable named `m`
3. **CRITICAL: The `data` variable is PRE-DEFINED. Do NOT create `data = {{...}}`. Just use `data["key"]` directly.**
4. Set `m.Params.OutputFlag = 0`
5. Print output format:
   ```
   print(f"status: {{m.Status}}")
   print(f"objective: {{m.ObjVal}}")
   ```
6. Implement ALL constraints from the mathematical model
7. Handle edge cases (empty lists, missing keys)

Return ONLY Python code, no explanation. Do NOT include any `data = ` definition.
'''

# ============================================================================
# Legacy: Separate 3-stage prompts (deprecated)
# ============================================================================
# These are kept for backward compatibility but Chain-of-Thought is preferred.

# ============================================================================
# Repair prompts
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

REPAIR_SYSTEM = '''You debug optimization models. Fix each issue in the report while preserving correct code.'''

# ============================================================================
# Regeneration prompt (for L1 FATAL)
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
3. **CRITICAL: The `data` variable is PRE-DEFINED. Do NOT create `data = {{...}}`. Just use `data["key"]` directly.**
4. Handle edge cases (empty arrays, missing keys)

Return ONLY the corrected Python code. Do NOT include any `data = ` definition.
'''

REGENERATE_SYSTEM = '''You fix broken optimization code. Ensure the new code is syntactically correct and handles all edge cases.'''


# ============================================================================
# L2 Anomaly Repair Prompt
# ============================================================================

L2_ANOMALY_REPAIR_PROMPT = '''Fix the structural anomaly in this optimization code.

## Problem
{problem_description}

## Data Structure
{data_structure}

## Current Code
```python
{code}
```

## CRITICAL ERROR - L2 Anomaly Detection

**Parameter**: {param}
**Baseline objective**: {z_baseline:.4f}
**After {param} increased by {delta}%**: z = {z_plus:.4f}
**After {param} decreased by {delta}%**: z = {z_minus:.4f}

**Problem**: BOTH increasing AND decreasing '{param}' IMPROVE the objective.
This is mathematically impossible for any valid optimization model.

**Likely causes**:
1. Constraint direction is wrong (>= should be <=, or vice versa)
2. Parameter is used with wrong sign in constraint
3. Parameter is applied to wrong constraint
4. Parameter is multiplied where it should be divided (or vice versa)

**Action Required**:
Review all constraints and objective terms involving '{param}' and fix the structural error.

## Instructions
1. Find where '{param}' is used in the code
2. Check if the constraint direction or sign is correct
3. Fix the structural error
4. **CRITICAL: The `data` variable is PRE-DEFINED. Do NOT create `data = {{...}}`.**

Return the COMPLETE fixed code in a ```python block.
'''

L2_ANOMALY_REPAIR_SYSTEM = '''You fix structural errors in optimization models.
Focus on constraint directions (>= vs <=) and parameter signs.'''


# ============================================================================
# L2 Direction Consistency Analysis Prompts
# ============================================================================
# Note: L2 prompts are defined in l2_direction.py to keep the module self-contained.
# Import them from there if needed:
#   from .l2_direction import L2_VERIFY_PROMPT, L2_REPAIR_PROMPT


# ============================================================================
# Helper functions
# ============================================================================

def describe_data_schema(data: dict, prefix: str = "", depth: int = 0) -> str:
    """
    Create a schema description of the data dictionary.

    Shows structure (keys, types, dimensions) but NOT actual values.
    This is the "schema-only visibility" design from the paper.
    """
    if depth > 3:
        return ""

    lines = []
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key

        if isinstance(value, (int, float)):
            lines.append(f"- {path}: {type(value).__name__} (scalar)")
        elif isinstance(value, list):
            if len(value) > 0:
                elem_type = type(value[0]).__name__
                if isinstance(value[0], list):
                    # 2D array
                    inner_len = len(value[0]) if value[0] else 0
                    lines.append(f"- {path}: list[{len(value)}][{inner_len}] (2D array)")
                elif isinstance(value[0], dict):
                    lines.append(f"- {path}: list[{len(value)}] of dict")
                else:
                    lines.append(f"- {path}: list[{len(value)}] of {elem_type}")
            else:
                lines.append(f"- {path}: list (empty)")
        elif isinstance(value, dict):
            keys_sample = list(value.keys())[:5]
            if len(value) > 5:
                keys_sample.append("...")
            lines.append(f"- {path}: dict with keys {keys_sample}")
            # Recurse for nested dicts
            nested = describe_data_schema(value, path, depth + 1)
            if nested:
                lines.append(nested)
        elif isinstance(value, str):
            lines.append(f"- {path}: str")
        else:
            lines.append(f"- {path}: {type(value).__name__}")

    return "\n".join(lines)


def format_diagnostic_report(report) -> str:
    """Format diagnostic report for display."""
    lines = [
        f"Status: {report.status}",
        f"Confidence: {report.confidence:.2f}",
        "",
        "Issues:"
    ]
    for r in report.layer_results:
        if r.severity.value in ['FATAL', 'ERROR', 'WARNING']:
            lines.append(f"  [{r.layer}] {r.severity.value}: {r.message}")
    return "\n".join(lines)


def format_issues_for_repair(report) -> str:
    """Extract actionable issues for repair."""
    issues = []
    for r in report.layer_results:
        if r.severity.value in ['FATAL', 'ERROR', 'WARNING']:
            if r.severity.value == 'FATAL':
                issues.append(f"EXECUTION ERROR: {r.message}")
            elif r.check == 'anomaly':
                # L2 Anomaly: both directions improve (physically impossible)
                issues.append(f"STRUCTURAL ANOMALY: {r.message}")
                if r.details and r.details.get("repair_hint"):
                    issues.append(f"   Hint: {r.details['repair_hint']}")
            elif 'monotonicity' in r.check:
                issues.append(f"CONSTRAINT DIRECTION ERROR: {r.message}")
            elif 'param_effect' in r.check or 'no_effect' in r.check:
                issues.append(f"MISSING CONSTRAINT: {r.message}")
            elif 'direction_anomaly' in r.check:
                issues.append(f"DIRECTION ANOMALY: {r.message}")
            elif 'cpt_missing' in r.check:
                issues.append(f"CPT DETECTED MISSING: {r.message}")
            else:
                issues.append(f"{r.severity.value}: {r.message}")
    return "\n".join(issues) if issues else "No specific issues identified."


# ============================================================================
# Context-Based Repair Prompts (Conservative Strategy)
# ============================================================================

REPAIR_WITH_CONTEXT_SYSTEM = '''You are an optimization code repair expert.

CRITICAL RULES:
1. ONLY fix issues in Sections 1 and 2 (CRITICAL ERRORS and HIGH PRIORITY)
2. Section 3 (DIAGNOSTIC INFO) is for REFERENCE ONLY - DO NOT modify code based on it
3. Be conservative - only make changes that are clearly necessary
4. Do NOT add constraints unless there is clear evidence they are missing
5. Preserve all working code - only change what is broken'''

REPAIR_WITH_CONTEXT_PROMPT = '''Fix this optimization code based on the categorized diagnostic report.

## Problem
{problem_description}

## Data Structure
The `data` variable is PRE-DEFINED with these keys:
{data_structure}

## Current Code
```python
{code}
```

---
## DIAGNOSTIC REPORT (READ CAREFULLY!)

### SECTION 1: CRITICAL ERRORS - YOU MUST FIX (NO EXCEPTIONS)
{critical_errors}

### SECTION 2: HIGH-PRIORITY ISSUES - YOU SHOULD FIX
{should_fix}

### SECTION 3: DIAGNOSTIC INFO - FOR REFERENCE ONLY (DO NOT FIX)
{for_reference}

---
## REPAIR INSTRUCTIONS

1. **MUST FIX** all issues in Section 1 (CRITICAL ERRORS)
   - These are mathematically certain errors (99%+ confidence)
   - Constraint direction errors, impossible behaviors, etc.

2. **SHOULD FIX** issues in Section 2 (HIGH PRIORITY)
   - These are high-confidence issues (80%+ confidence)
   - Missing constraints detected by CPT, etc.

3. **DO NOT FIX** issues in Section 3 (DIAGNOSTIC INFO)
   - These are likely normal behavior or numerical artifacts
   - Slack constraints, normal optimization behavior, etc.
   - Provided for context only - ignore for repair purposes

4. **CRITICAL**: The `data` variable is PRE-DEFINED. Do NOT create `data = {{...}}`.

Return the COMPLETE fixed code in a ```python block.
'''


def build_repair_prompt(
    diagnostics: list,
    code: str,
    problem_desc: str,
    data_structure: str,
    current_obj=None,
):
    """
    Assemble repair prompt from unified Diagnostic objects.

    Only includes diagnostics where triggers_repair=True.

    Args:
        diagnostics: List of Diagnostic objects
        code: Current optimization code
        problem_desc: Original problem description
        data_structure: Data schema string (from describe_data_schema)
        current_obj: Current objective value (None if model didn't solve)

    Returns:
        Formatted repair prompt string, or None if nothing to repair.
    """
    actionable = [d for d in diagnostics if d.triggers_repair]
    if not actionable:
        return None

    # Build issues section
    issues_lines = []
    for i, d in enumerate(actionable, 1):
        issues_lines.append(
            f"=== Issue {i} [{d.layer}] [{d.severity}] ===\n"
            f"Type: {d.issue_type}\n"
            f"Target: {d.target_name}\n"
            f"Evidence: {d.evidence}\n"
        )
    issues_text = "\n".join(issues_lines)

    # Build reference section (non-actionable diagnostics)
    info_items = [d for d in diagnostics if not d.triggers_repair]
    if info_items:
        ref_lines = [
            "+" + "-" * 65 + "+",
            "| Below items are NORMAL in 80%+ of cases.  DO NOT CHANGE.       |",
            "+" + "-" * 65 + "+",
            "",
        ]
        for j, d in enumerate(info_items, 1):
            ref_lines.append(
                f"{j}. [{d.layer}] {d.issue_type} — {d.target_name}\n"
                f"   {d.evidence}\n"
                f"   Action: DO NOT FIX (unless 100% certain this is an error)\n"
            )
        reference_text = "\n".join(ref_lines)
    else:
        reference_text = "[OK] No additional diagnostic information."

    # Objective line
    if current_obj is not None:
        obj_line = f"Current objective value: {current_obj}"
    else:
        obj_line = "Model did not produce an objective value."

    prompt = f"""Fix this optimization code based on the behavioral verification report.

## Problem
{problem_desc}

## Data Structure
The `data` variable is PRE-DEFINED with these keys:
{data_structure}

## Current Code
```python
{code}
```

## {obj_line}

---
## ISSUES DETECTED ({len(actionable)} actionable)

{issues_text}

---
## REFERENCE ONLY (DO NOT FIX)

{reference_text}

---
## REPAIR INSTRUCTIONS

1. Read each Issue carefully, especially the Evidence field
2. Identify the root cause in your code for each actionable issue
3. Fix ALL actionable issues above
4. DO NOT fix items in the REFERENCE section — they are likely normal
5. **CRITICAL**: The `data` variable is PRE-DEFINED as a Python dict. Do NOT create `data = {{{{...}}}}`.
6. **CRITICAL**: Do NOT use `data = json.loads(...)`. The `data` variable is ALREADY a dict — access it directly as `data["key"]`.
7. Preserve all working code — only change what is broken

**SAFETY RULES (violations will cause your repair to be rejected):**
- Do NOT redefine the `data` variable. Data is provided externally as a Python dict.
- Do NOT use `json.loads()` to re-parse data. Access `data["key"]` directly.
- Do NOT modify data contents (no `data[key] = new_value`).
- Only modify the optimization model: variables, constraints, objective function.

Return the COMPLETE fixed code in a ```python block."""

    return prompt


REPAIR_PROMPT_SYSTEM = """You are an optimization code repair expert.

CRITICAL RULES:
1. ONLY fix the actionable issues listed in the ISSUES DETECTED section
2. Items in REFERENCE ONLY are for context — DO NOT modify code based on them
3. Be conservative — only make changes that are clearly necessary
4. Preserve all working code — only change what is broken
5. NEVER redefine or modify the `data` variable — it is provided externally as a Python dict
6. NEVER use `json.loads()` — `data` is already a dict, access keys directly with `data["key"]`"""


def format_repair_section(items: list, empty_message: str = "None") -> str:
    """Format a list of issues for repair prompt section (ERROR/WARNING)."""
    if not items:
        return empty_message

    lines = []
    for i, item in enumerate(items, 1):
        layer = item.get("layer", "?")
        severity = item.get("severity", "?")
        message = item.get("message", "No message")
        action = item.get("action", "")

        lines.append(f"{i}. [{layer}] {severity}: {message}")
        if action:
            lines.append(f"   Action: {action}")

        # Add relevant details if present
        details = item.get("details", {})
        if details:
            if "param" in details:
                lines.append(f"   Parameter: {details['param']}")
            if "expected" in details and "actual" in details:
                lines.append(f"   Expected: {details['expected']}, Actual: {details['actual']}")

    return "\n".join(lines)


def format_reference_section(items: list) -> str:
    """
    Format INFO-level items for reference section.

    These are CONTEXT ONLY - the LLM should NOT fix these unless 100% certain.
    Uses emphatic formatting to prevent over-correction.
    """
    if not items:
        return "[OK] No diagnostic information."

    lines = []

    # Emphatic reminder header
    lines.append("+" + "-" * 65 + "+")
    lines.append("| REMINDER: Below items are NORMAL in 80%+ of cases.              |")
    lines.append("| DEFAULT ACTION: Confirm code is correct. DO NOT CHANGE.         |")
    lines.append("+" + "-" * 65 + "+")
    lines.append("")

    for i, item in enumerate(items, 1):
        layer = item.get("layer", "?")
        check = item.get("check", "unknown")
        message = item.get("message", "")

        lines.append(f"{i}. [{layer}] {check}")
        lines.append(f"   Observation: {message}")
        lines.append(f"   Status: [OK] PROBABLY NORMAL (slack constraint, numerical artifact, etc.)")
        lines.append(f"   Action: DO NOT FIX (unless you are 100% certain this is an error)")
        lines.append("")

    # Final reminder
    lines.append("-" * 65)
    lines.append("[!] FINAL REMINDER: If you're not 100% sure, DO NOT TOUCH.")
    lines.append("    Wrong 'fixes' to normal behavior can break correct code.")
    lines.append("-" * 65)

    return "\n".join(lines)
