"""
Module 3: Diagnosis-Guided Repair

Takes verification failure diagnosis and guides LLM to fix the code.

Key principles:
- MINIMAL changes: Only fix the specific issue
- NO REGRESSION: Do not break working parts (L1/L2)
- TARGETED: Focus on the failing constraint/parameter
"""

from typing import List
from dataclasses import dataclass
from .error_patterns import get_repair_hints, format_repair_guidance


# Layer descriptions for context
LAYER_DESCRIPTIONS = {
    1: "Execution - Code runs without errors",
    2: "Feasibility - Model returns OPTIMAL status",
    3: "Monotonicity - Each parameter affects the objective value",
    4: "Sensitivity Direction - Parameter changes affect objective in expected direction",
    5: "Boundary - Extreme values cause expected behavior",
    6: "Domain Probes - Domain-specific behavior is correct"
}


@dataclass
class RepairContext:
    """Context for repair operation"""
    code: str
    diagnosis: str
    failed_layer: int
    history: List[str]
    repair_hints: List[str]


class DiagnosisRepairer:
    """Generates repair prompts based on verification diagnosis."""

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def create_repair_context(self, code: str, diagnosis: str, failed_layer: int,
                               history: List[str] = None) -> RepairContext:
        hints = get_repair_hints(failed_layer, diagnosis)
        return RepairContext(code=code, diagnosis=diagnosis, failed_layer=failed_layer,
                            history=history or [], repair_hints=hints)

    def _get_layer_context(self, failed_layer: int) -> str:
        """Generate context about what passed and what failed."""
        passed = [f"  L{i}: {LAYER_DESCRIPTIONS[i]} - PASSED"
                  for i in range(1, failed_layer)]
        failed = f"  L{failed_layer}: {LAYER_DESCRIPTIONS[failed_layer]} - FAILED"

        if passed:
            return "Verification Status:\n" + "\n".join(passed) + "\n" + failed
        return f"Verification Status:\n{failed}"

    def _get_preservation_rules(self, failed_layer: int) -> str:
        """Generate rules about what NOT to change."""
        if failed_layer == 1:
            return ""  # Nothing passed yet

        rules = ["\n## CRITICAL: Preservation Rules"]
        rules.append("The following parts are WORKING - DO NOT modify them:")

        if failed_layer >= 2:
            rules.append("- Model creation, variable definitions, and basic structure")
            rules.append("- Import statements and data loading")
        if failed_layer >= 3:
            rules.append("- Objective function (it returns OPTIMAL)")
            rules.append("- Existing constraints that make the model feasible")

        rules.append("\nONLY add or fix the SPECIFIC constraint mentioned in the diagnosis.")
        return "\n".join(rules)

    def _get_targeted_guidance(self, diagnosis: str, failed_layer: int) -> str:
        """Extract specific guidance based on diagnosis."""
        guidance = []

        if failed_layer == 3 and "NO EFFECT" in diagnosis.upper():
            # Extract parameter name from diagnosis
            import re
            param_match = re.search(r"Parameter '([^']+)'", diagnosis)
            if param_match:
                param = param_match.group(1)
                guidance.append(f"\n## Specific Fix Required")
                guidance.append(f"Parameter '{param}' is NOT affecting the objective.")
                guidance.append(f"")
                guidance.append(f"YOU MUST ADD A CONSTRAINT that connects '{param}' to decision variables.")
                guidance.append(f"")
                guidance.append(f"Common constraint patterns (pick the appropriate one):")
                guidance.append(f"")
                # Infer constraint type from parameter name
                param_lower = param.lower()
                if "capacity" in param_lower or "cap" in param_lower:
                    guidance.append(f"  # {param} is likely a CAPACITY constraint:")
                    guidance.append(f"  for key in data['{param}']:")
                    guidance.append(f"      m.addConstr(sum_of_relevant_vars <= data['{param}'][key])")
                elif "usage" in param_lower or "consumption" in param_lower:
                    guidance.append(f"  # {param} is likely used in a resource constraint:")
                    guidance.append(f"  m.addConstr(quicksum(data['{param}'][p] * var[p,...] for p in ...) <= capacity)")
                elif "demand" in param_lower:
                    guidance.append(f"  # {param} is likely a DEMAND constraint:")
                    guidance.append(f"  m.addConstr(supply_var >= data['{param}'][key])")
                else:
                    guidance.append(f"  # Generic constraint pattern:")
                    guidance.append(f"  m.addConstr(decision_var <= data['{param}'][key])")
                    guidance.append(f"  # OR")
                    guidance.append(f"  m.addConstr(decision_var >= data['{param}'][key])")

        return "\n".join(guidance) if guidance else ""

    def repair(self, code: str, diagnosis: str, failed_layer: int,
               history: List[str] = None) -> str:
        """Generate repaired code using LLM."""
        if self.llm is None:
            raise ValueError("LLM client not provided")

        context = self.create_repair_context(code, diagnosis, failed_layer, history)

        history_section = ""
        if context.history:
            history_section = "\n## Previous Repair Attempts (avoid repeating these):\n"
            history_section += "\n".join(f"  {i}. {d}" for i, d in enumerate(context.history, 1))

        layer_context = self._get_layer_context(context.failed_layer)
        preservation_rules = self._get_preservation_rules(context.failed_layer)
        targeted_guidance = self._get_targeted_guidance(context.diagnosis, context.failed_layer)
        repair_guidance = format_repair_guidance(context.failed_layer, context.diagnosis)

        prompt = f"""[CODE REPAIR - Layer {context.failed_layer}]

{layer_context}

## Current Diagnosis
{context.diagnosis}
{targeted_guidance}
{repair_guidance}
{preservation_rules}
{history_section}

## Original Code
```python
{context.code}
```

## Task
Fix ONLY the specific issue. Make the MINIMAL change needed.
- If L3 failed: ADD the missing constraint, do NOT rewrite the whole model
- Keep all imports, variable definitions, and working constraints UNCHANGED
- The fix should be 1-5 lines of code change, not a complete rewrite

Output ONLY the corrected Python code. No explanations."""

        response = self.llm.generate(prompt)
        return self._extract_code(response)

    def _extract_code(self, response: str) -> str:
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        lines = response.strip().split("\n")
        code_lines, in_code = [], False
        for line in lines:
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                in_code = True
            if in_code:
                code_lines.append(line)
        return "\n".join(code_lines) if code_lines else response.strip()
