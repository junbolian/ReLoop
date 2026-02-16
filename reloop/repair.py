"""ReLoop Diagnostic Repair Module

Conservative Repair Strategy:
- Only repair ERROR and WARNING level issues
- INFO level issues are for reference only (do NOT trigger repair)

Issue Classification:
- critical_errors: ERROR level (99%+ confidence) - MUST fix
- should_fix: WARNING level (80%+ confidence) - SHOULD fix
- for_reference: INFO level (likely normal) - DO NOT fix

L2 Adversarial Mechanism:
- Repair LLM can Accept or Reject L2 diagnostics
- Rejection triggers re-analysis with feedback
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .verification import VerificationReport, Severity
from .prompts import (
    REPAIR_PROMPT, REPAIR_SYSTEM,
    REPAIR_WITH_CONTEXT_PROMPT, REPAIR_WITH_CONTEXT_SYSTEM,
    L2_ANOMALY_REPAIR_PROMPT, L2_ANOMALY_REPAIR_SYSTEM,
    format_diagnostic_report, format_issues_for_repair,
    format_repair_section, format_reference_section,
    describe_data_schema, format_data_instructions
)


@dataclass
class RepairResult:
    """Result of repair operation."""
    code: str
    changed: bool
    l2_decisions: Optional[List[Dict]] = None  # L2 Accept/Reject decisions


class CodeRepairer:
    """Repair optimization code based on verification diagnostics."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def repair(
        self,
        code: str,
        data: Dict[str, Any],
        report: VerificationReport,
        problem_description: str,
        max_attempts: int = 3
    ) -> str:
        """Repair code based on verification report."""
        if report.status == 'VERIFIED':
            return code

        diagnostic_report = format_diagnostic_report(report)
        issues = format_issues_for_repair(report)
        data_structure = describe_data_schema(data) if data else ""
        data_instructions = format_data_instructions(data_structure)

        prompt = REPAIR_PROMPT.format(
            problem_description=problem_description,
            data_instructions=data_instructions,
            code=code,
            diagnostic_report=diagnostic_report,
            issues=issues
        )

        for _ in range(max_attempts):
            try:
                response = self.llm_client.generate(prompt, system=REPAIR_SYSTEM)
                repaired = self._extract_code(response)

                # Verify the repair is different and valid
                if repaired != code and 'import gurobipy' in repaired:
                    return repaired
            except Exception:
                continue

        return code  # Return original if repair fails

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        match = re.search(r'```python\s*([\s\S]*?)\s*```', response)
        if match:
            return match.group(1).strip()

        match = re.search(r'```\s*([\s\S]*?)\s*```', response)
        if match:
            return match.group(1).strip()

        return response.strip()

    def repair_with_context(
        self,
        code: str,
        data: Dict[str, Any],
        problem_description: str,
        critical_errors: List[Dict],
        should_fix: List[Dict],
        for_reference: List[Dict],
        max_attempts: int = 3
    ) -> str:
        """
        Repair code with categorized issue context.

        This is the conservative repair strategy:
        - critical_errors: ERROR level, MUST fix (99%+ confidence)
        - should_fix: WARNING level, SHOULD fix (80%+ confidence)
        - for_reference: INFO level, DO NOT fix (likely normal)

        Args:
            code: Current code to repair
            data: Problem data dictionary
            problem_description: Natural language problem description
            critical_errors: List of ERROR-level issues (must fix)
            should_fix: List of WARNING-level issues (should fix)
            for_reference: List of INFO-level issues (reference only)
            max_attempts: Max repair attempts

        Returns:
            Repaired code (or original if repair fails)
        """
        # If nothing to fix, return original code
        if not critical_errors and not should_fix:
            return code

        data_structure = describe_data_schema(data) if data else ""
        data_instructions = format_data_instructions(data_structure)

        # Format each section
        critical_section = format_repair_section(
            critical_errors,
            "None - no critical errors detected."
        )
        should_fix_section = format_repair_section(
            should_fix,
            "None - no high-priority issues detected."
        )
        # Use special formatting for INFO items - emphasize "DO NOT FIX"
        reference_section = format_reference_section(for_reference)

        prompt = REPAIR_WITH_CONTEXT_PROMPT.format(
            problem_description=problem_description,
            data_instructions=data_instructions,
            code=code,
            critical_errors=critical_section,
            should_fix=should_fix_section,
            for_reference=reference_section
        )

        for _ in range(max_attempts):
            try:
                response = self.llm_client.generate(
                    prompt,
                    system=REPAIR_WITH_CONTEXT_SYSTEM
                )
                repaired = self._extract_code(response)

                # Verify the repair is different and valid
                if repaired != code and 'import gurobipy' in repaired:
                    return repaired
            except Exception:
                continue

        return code  # Return original if repair fails

    def repair_l2_anomaly(
        self,
        code: str,
        data: Dict[str, Any],
        problem_description: str,
        anomaly_result: Dict,
        max_attempts: int = 3
    ) -> str:
        """
        Repair code for L2 anomaly (both directions improve).

        Args:
            anomaly_result: Dict with param, z_baseline, z_plus, z_minus, etc.

        Returns:
            Repaired code
        """
        data_structure = describe_data_schema(data) if data else ""
        data_instructions = format_data_instructions(data_structure)

        prompt = L2_ANOMALY_REPAIR_PROMPT.format(
            problem_description=problem_description,
            data_instructions=data_instructions,
            code=code,
            param=anomaly_result.get("param", "unknown"),
            z_baseline=anomaly_result.get("baseline_obj", 0),
            z_plus=anomaly_result.get("obj_up", 0),
            z_minus=anomaly_result.get("obj_down", 0),
            delta=20  # Default delta percentage
        )

        for _ in range(max_attempts):
            try:
                response = self.llm_client.generate(
                    prompt, system=L2_ANOMALY_REPAIR_SYSTEM
                )
                repaired = self._extract_code(response)

                if repaired != code and 'import gurobipy' in repaired:
                    return repaired
            except Exception:
                continue

        return code

    def repair_with_l2(
        self,
        code: str,
        data: Dict[str, Any],
        problem_description: str,
        l2_diagnostics: str,
        l1_errors: List[Dict],
        l3_warnings: List[Dict],
        for_reference: List[Dict],
        max_attempts: int = 3
    ) -> RepairResult:
        """
        Repair with L2 adversarial mechanism (Accept/Reject).

        Args:
            l2_diagnostics: Formatted L2 diagnostics string
            l1_errors: L1 ERROR items
            l3_warnings: L3 WARNING items
            for_reference: INFO items

        Returns:
            RepairResult with code, changed flag, and L2 decisions
        """
        from .l2_direction import L2_REPAIR_PROMPT

        data_structure = describe_data_schema(data) if data else ""
        data_instructions = format_data_instructions(data_structure)

        # Format sections
        l1_section = format_repair_section(l1_errors, "None") if l1_errors else "None"
        l3_section = format_repair_section(l3_warnings, "None") if l3_warnings else "None"
        ref_section = format_reference_section(for_reference)

        # Build combined prompt
        prompt = f'''Fix this optimization code based on the diagnostic report.

## Problem
{problem_description}

{data_instructions}

## Current Code
```python
{code}
```

---
## DIAGNOSTIC REPORT

### SECTION 1: L1 CRITICAL ERRORS (MUST FIX)
{l1_section}

### SECTION 2: L2 DIRECTION ANALYSIS (Review & Decide)
{l2_diagnostics}

**For each L2 diagnostic, you must either:**
- **ACCEPT**: Fix the code accordingly
- **REJECT**: Explain why the analysis is wrong (your feedback will trigger re-analysis)

### SECTION 3: L3 WARNINGS (SHOULD FIX)
{l3_section}

### SECTION 4: INFO (FOR REFERENCE ONLY)
{ref_section}

---
## Response Format

Return JSON with your decisions:
```json
{{
    "l2_decisions": [
        {{
            "param": "parameter_name",
            "action": "accept" | "reject",
            "reason": "Explanation"
        }}
    ],
    "fixed_code": "Complete fixed Python code (if any changes needed)"
}}
```

If no changes needed, set "fixed_code" to null.
'''

        for _ in range(max_attempts):
            try:
                response = self.llm_client.generate(prompt)
                result = self._parse_l2_repair_response(response, code)
                if result:
                    return result
            except Exception:
                continue

        return RepairResult(code=code, changed=False, l2_decisions=[])

    # Backward compatibility alias
    def repair_with_l4(self, code, data, problem_description, l4_diagnostics,
                        l2_errors, l5_warnings, for_reference, max_attempts=3):
        return self.repair_with_l2(
            code, data, problem_description, l4_diagnostics,
            l2_errors, l5_warnings, for_reference, max_attempts
        )

    def _parse_l2_repair_response(
        self, response: str, original_code: str
    ) -> Optional[RepairResult]:
        """Parse repair response with L2 decisions."""
        # Try to find JSON
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                parsed = None
        else:
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                parsed = None

        if parsed:
            l2_decisions = parsed.get("l2_decisions", parsed.get("l4_decisions", []))
            fixed_code = parsed.get("fixed_code")

            if fixed_code and fixed_code != "null":
                # Extract code if wrapped in markdown
                if "```python" in fixed_code:
                    match = re.search(r'```python\s*([\s\S]*?)\s*```', fixed_code)
                    if match:
                        fixed_code = match.group(1).strip()
                elif "```" in fixed_code:
                    match = re.search(r'```\s*([\s\S]*?)\s*```', fixed_code)
                    if match:
                        fixed_code = match.group(1).strip()

                if fixed_code and 'import gurobipy' in fixed_code:
                    return RepairResult(
                        code=fixed_code,
                        changed=True,
                        l2_decisions=l2_decisions
                    )

            return RepairResult(
                code=original_code,
                changed=False,
                l2_decisions=l2_decisions
            )

        # Fallback: try to extract code directly
        code = self._extract_code(response)
        if code != original_code and 'import gurobipy' in code:
            return RepairResult(code=code, changed=True, l2_decisions=[])

        return None

    # Backward compatibility alias
    _parse_l4_repair_response = _parse_l2_repair_response

    def get_repair_suggestions(self, report: VerificationReport) -> List[str]:
        """Generate repair suggestions from verification report."""
        suggestions = []

        for r in report.layer_results:
            if r.severity in [Severity.ERROR, Severity.WARNING]:
                if 'direction' in r.check:
                    # L2 direction violation
                    suggestions.append(
                        f"Unexpected direction change: {r.message}"
                    )
                elif 'cpt_missing' in r.check:
                    # L3 CPT
                    suggestions.append(
                        f"CPT detected missing constraint: {r.message}"
                    )
                else:
                    suggestions.append(f"{r.severity.value}: {r.message}")

        return suggestions
