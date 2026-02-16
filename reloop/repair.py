"""ReLoop Diagnostic Repair Module

Conservative Repair Strategy:
- Only repair WARNING level issues (from L2 behavioral testing)
- INFO level issues are for reference only (do NOT trigger repair)

Issue Classification:
- should_fix: WARNING level (80%+ confidence) - SHOULD fix
- for_reference: INFO level (likely normal) - DO NOT fix
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .verification import VerificationReport, Severity
from .prompts import (
    REPAIR_PROMPT, REPAIR_SYSTEM,
    REPAIR_WITH_CONTEXT_PROMPT, REPAIR_WITH_CONTEXT_SYSTEM,
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

    def get_repair_suggestions(self, report: VerificationReport) -> List[str]:
        """Generate repair suggestions from verification report."""
        suggestions = []

        for r in report.layer_results:
            if r.severity in [Severity.ERROR, Severity.WARNING]:
                if 'cpt_missing' in r.check:
                    suggestions.append(
                        f"CPT detected missing constraint: {r.message}"
                    )
                elif 'opt_missing' in r.check:
                    suggestions.append(
                        f"OPT detected missing objective term: {r.message}"
                    )
                else:
                    suggestions.append(f"{r.severity.value}: {r.message}")

        return suggestions
