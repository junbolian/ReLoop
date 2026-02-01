"""ReLoop LLM Prompt Templates"""

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
   ```
   print(f"status: {{m.Status}}")
   print(f"objective: {{m.ObjVal}}")
   ```

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
        if r.severity.value == 'WARNING':
            if 'monotonicity' in r.check:
                issues.append(f"CONSTRAINT DIRECTION ERROR: {r.message}")
            elif 'param_effect' in r.check:
                issues.append(f"MISSING CONSTRAINT: {r.message}")
            elif 'cpt_missing' in r.check:
                issues.append(f"CPT DETECTED MISSING: {r.message}")
    return "\n".join(issues) if issues else "No specific issues."
