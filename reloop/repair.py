"""ReLoop Diagnostic Repair Module"""

import re
from typing import Dict, Any, List
from .verification import VerificationReport, Severity
from .prompts import (
    REPAIR_PROMPT, REPAIR_SYSTEM,
    format_diagnostic_report, format_issues_for_repair
)


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

        prompt = REPAIR_PROMPT.format(
            problem_description=problem_description,
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

    def get_repair_suggestions(self, report: VerificationReport) -> List[str]:
        """Generate repair suggestions from verification report."""
        suggestions = []

        for r in report.layer_results:
            if r.severity == Severity.WARNING:
                if 'monotonicity' in r.check:
                    suggestions.append(
                        f"Check constraint direction: {r.message}"
                    )
                elif 'param_effect' in r.check:
                    param = r.details.get('param', '') if r.details else ''
                    suggestions.append(
                        f"Missing constraint for parameter: {param}"
                    )
                elif 'direction' in r.check:
                    suggestions.append(
                        f"Unexpected direction change: {r.message}"
                    )
                elif 'cpt_missing' in r.check:
                    suggestions.append(
                        f"CPT detected missing constraint: {r.message}"
                    )

        return suggestions
