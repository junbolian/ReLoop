"""
ReLoop L2: Semantic Audit (Adversarial)

LLM-based semantic verification with adversarial rejection mechanism.
Compares problem description against generated code to identify
missing terms, wrong coefficients, and missing constraints.

Design:
- LLM_audit: Compares problem description vs code using 3-part checklist
- LLM_repair: Can Accept (fix) or Reject (with reason) diagnostics
- Rejection triggers re-audit with context
- Exit when: all PASS, all rejected + others PASS, or max iterations
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .param_utils import extract_numeric_params, perturb_param, should_skip_param, get_param_value
from .perturbation import (
    get_source_code_param_names, extract_perturbable_params,
    perturb_code, _match_param,
)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class L2VerifyResult:
    """Result of L2 semantic audit for a single finding."""
    param: str                   # Target element name (cost term, coefficient, constraint)
    param_role: str              # "missing_term" | "wrong_coefficient" | "missing_constraint"
    expected_direction: str      # "n/a" (no perturbation in semantic audit)
    actual_direction: str        # "n/a" (no perturbation in semantic audit)
    is_violation: bool           # Whether a semantic issue was found
    confidence: float            # 0.0 - 1.0
    reason: str                  # Evidence and suggested fix
    z_plus: float               # 0.0 (unused, kept for API compatibility)
    z_minus: float              # 0.0 (unused, kept for API compatibility)
    z_baseline: float           # Baseline objective


@dataclass
class L2RepairDecision:
    """Decision from repair LLM for a single parameter."""
    param: str
    action: str                  # "accept" | "reject"
    rejection_reason: str        # Detailed reason if rejected
    fix_description: str         # Description of fix if accepted


@dataclass
class L2RejectionHistory:
    """History of rejections for re-analysis context."""
    param: str
    round: int
    verify_reason: str
    rejection_reason: str


@dataclass
class L2Result:
    """Aggregated L2 direction analysis result."""
    status: str                  # "PASS" | "ISSUES" | "INFO"
    verify_results: List[L2VerifyResult]
    decisions: List[L2RepairDecision]
    rejection_history: Dict[str, List[L2RejectionHistory]]


# Backward compatibility aliases
L4VerifyResult = L2VerifyResult
L4RepairDecision = L2RepairDecision
L4RejectionHistory = L2RejectionHistory
L4Result = L2Result


# =============================================================================
# Prompts
# =============================================================================

L2_VERIFY_SYSTEM = """You are an optimization model auditor. Your task is to compare a problem description against generated code and identify semantic errors: missing cost/revenue terms, wrong coefficient interpretations, and missing constraints."""

L2_VERIFY_PROMPT = '''Audit this optimization code against the problem description. Identify any semantic discrepancies.

## Problem Description
{problem_description}

## Generated Code
```python
{code}
```

## Audit Checklist

### 1. Objective Function Completeness
List EVERY cost or revenue term mentioned in the problem description, then check if each is present in the code's objective function.
- Common miss: purchasing/procurement cost, holding cost, shortage/backorder cost, transportation cost, setup/fixed cost
- If a term from the description is missing from the objective, flag it.

### 2. Coefficient Semantic Correctness
For each numeric coefficient used in the code, verify its interpretation matches the problem description:
- **Rate vs total**: Is `r` a rate (e.g., 5% = 0.05) or a multiplier (e.g., 1.05)? Does the code use `r * x` when it should use `(1 + r) * x`, or vice versa?
- **Per-unit vs aggregate**: Is the cost per-unit or total? Does the code multiply correctly?
- **Percentage vs absolute**: Is the value a percentage (divide by 100) or already a fraction?

### 3. Constraint Completeness
Check whether all constraints mentioned in the problem description are present in the code:
- Supply/capacity limits
- Demand satisfaction requirements
- Budget constraints
- Logical/linking constraints
- Variable bounds (non-negativity, upper bounds)

## Confidence Guidelines
- 0.9+: Clear evidence from problem text that something is missing or wrong
- 0.7-0.9: Likely issue based on domain knowledge
- 0.5-0.7: Possible issue, ambiguous problem text
- <0.5: Speculative, do not flag

## Response Format
Return ONLY a JSON array. If no issues found, return an empty array `[]`.
```json
[
  {{
    "issue_type": "missing_term" | "wrong_coefficient" | "missing_constraint",
    "target": "name of the missing/wrong element",
    "evidence": "quote from problem description vs what code does",
    "confidence": 0.0 to 1.0,
    "suggested_fix": "specific fix description"
  }}
]
```
Only include issues with confidence >= 0.5. Be precise — false positives waste repair budget.
'''

L2_REJECTION_CONTEXT = """
## Previous Analysis Was Rejected
**Your previous analysis**: {previous_reason}
**Rejection reason from repair agent**: {rejection_reason}

Please reconsider your analysis in light of this feedback.
If the rejection reason is valid, adjust your conclusion.
If you still believe your original analysis was correct, provide stronger justification.
"""

L2_REPAIR_PROMPT = '''Review these semantic audit findings and decide how to handle each.

## Problem Description
{problem_description}

## Current Code
```python
{code}
```

## Semantic Audit Findings

{l4_diagnostics}

## For Each Finding, Choose One:

### Option 1: ACCEPT
If you agree the finding is correct and the code has this issue, fix the code accordingly.

### Option 2: REJECT
If you believe the finding is WRONG (the code is actually correct), provide a detailed rejection reason.

**When to REJECT (examples):**
1. "The audit says purchasing cost is missing, but it IS included in variable `procurement_cost` on line X"
2. "The coefficient is correctly interpreted as a rate — the problem says 'annual rate of 5%' and code uses 0.05"
3. "The constraint is present but named differently than what the audit expected"

**When to ACCEPT:**
1. A cost/revenue term from the problem description is genuinely missing from the objective
2. A coefficient is demonstrably misinterpreted (rate vs total, per-unit vs aggregate)
3. A constraint mentioned in the problem is genuinely absent from the model

**SAFETY RULES (violations will cause your repair to be rejected):**
- Do NOT redefine the `data` variable. Data is provided externally as a Python dict.
- Do NOT use `json.loads()` — `data` is already a dict, access keys directly with `data["key"]`.
- Do NOT modify data contents (no `data[key] = new_value`).
- Only modify the optimization model: variables, constraints, objective function.

## Response Format
Return JSON only (no other text):
```json
{{
    "decisions": [
        {{
            "param": "parameter_name",
            "action": "accept" | "reject",
            "reason": "If reject: why the finding is wrong. If accept: description of the fix applied"
        }}
    ],
    "fixed_code": "Complete fixed code if any accepts, otherwise null"
}}
```
'''

# Backward compatibility aliases
L4_VERIFY_SYSTEM = L2_VERIFY_SYSTEM
L4_VERIFY_PROMPT = L2_VERIFY_PROMPT
L4_REJECTION_CONTEXT = L2_REJECTION_CONTEXT
L4_REPAIR_PROMPT = L2_REPAIR_PROMPT


# =============================================================================
# L2 Semantic Audit Verifier (Adversarial)
# =============================================================================

class L2DirectionVerifier:
    """
    L2: Semantic Audit (Adversarial)

    Uses LLM to compare problem description against generated code,
    identifying missing terms, wrong coefficients, and missing constraints.
    Features adversarial mechanism where repair LLM can reject diagnostics.
    """

    def __init__(
        self,
        llm_client,
        delta: float = 0.1,
        max_rejections: int = 2,
        confidence_threshold: float = 0.5
    ):
        """
        Args:
            llm_client: LLM client with generate(prompt, system=None) -> str method
            delta: Perturbation factor (default 10%)
            max_rejections: Max times a param can be rejected before downgrade to INFO
            confidence_threshold: Min confidence to trigger repair (default 0.5)
        """
        self.llm_client = llm_client
        self.delta = delta
        self.max_rejections = max_rejections
        self.confidence_threshold = confidence_threshold
        self.rejection_history: Dict[str, List[L2RejectionHistory]] = {}

    def reset(self):
        """Reset rejection history for new verification session."""
        self.rejection_history = {}

    def verify(
        self,
        code: str,
        data: Dict,
        baseline_obj: float,
        problem_description: str,
        params: Optional[List[str]] = None,
        exclude_params: Optional[List[str]] = None,
        executor=None,
        mode: str = "data_dict"
    ) -> List[L2VerifyResult]:
        """
        Semantic audit: compare problem description against generated code
        to identify missing terms, wrong coefficients, and missing constraints.

        No perturbation is performed. Uses a single LLM call for the audit.

        Args:
            code: Generated optimization code
            data: Problem data dictionary
            baseline_obj: Baseline objective value
            problem_description: Natural language problem description
            params: (unused, kept for API compatibility)
            exclude_params: (unused, kept for API compatibility)
            executor: (unused, kept for API compatibility)
            mode: (unused, kept for API compatibility)

        Returns:
            List of L2VerifyResult for each identified issue
        """
        return self._semantic_audit(
            code=code,
            problem_description=problem_description,
            baseline_obj=baseline_obj,
        )

    def _perturb_and_solve(
        self, executor, code: str, data: Dict, param: str,
        mode: str = "data_dict", baseline_obj: Optional[float] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        """Perturb parameter and get objective values."""
        if mode == "source_code":
            code_params = extract_perturbable_params(code)
            matched = _match_param(code_params, param)
            if not matched:
                return None, None
            code_plus = perturb_code(code, matched['access_path'], 1 + self.delta)
            code_minus = perturb_code(code, matched['access_path'], 1 - self.delta)
            result_plus = executor.execute(code_plus, data)
            result_minus = executor.execute(code_minus, data)
        else:
            # data_dict or hybrid
            data_plus = perturb_param(data, param, 1 + self.delta)
            data_minus = perturb_param(data, param, 1 - self.delta)
            result_plus = executor.execute(code, data_plus)
            result_minus = executor.execute(code, data_minus)

        obj_plus = result_plus.get("objective")
        obj_minus = result_minus.get("objective")

        # Hybrid fallback: if data perturbation had no effect, try source-code
        if (mode == "hybrid" and baseline_obj is not None
                and obj_plus is not None and obj_minus is not None):
            threshold = 1e-4 * max(abs(baseline_obj), 1.0)
            if (abs(obj_plus - baseline_obj) <= threshold
                    and abs(obj_minus - baseline_obj) <= threshold):
                code_params = extract_perturbable_params(code)
                matched = _match_param(code_params, param)
                if matched:
                    code_plus = perturb_code(code, matched['access_path'], 1 + self.delta)
                    code_minus = perturb_code(code, matched['access_path'], 1 - self.delta)
                    if code_plus != code or code_minus != code:
                        result_plus = executor.execute(code_plus, data)
                        result_minus = executor.execute(code_minus, data)
                        obj_plus = result_plus.get("objective")
                        obj_minus = result_minus.get("objective")

        return obj_plus, obj_minus

    def _build_rejection_context(self, param: str) -> str:
        """Build rejection context for re-analysis."""
        if param not in self.rejection_history or not self.rejection_history[param]:
            return ""

        latest = self.rejection_history[param][-1]
        return L2_REJECTION_CONTEXT.format(
            previous_reason=latest.verify_reason,
            rejection_reason=latest.rejection_reason
        )

    def _semantic_audit(
        self,
        code: str,
        problem_description: str,
        baseline_obj: float,
    ) -> List[L2VerifyResult]:
        """Run semantic audit: single LLM call comparing problem desc vs code."""
        # Build rejection context from any previous rounds
        rejection_ctx = ""
        if self.rejection_history:
            parts = []
            for param, history in self.rejection_history.items():
                latest = history[-1]
                parts.append(
                    L2_REJECTION_CONTEXT.format(
                        previous_reason=latest.verify_reason,
                        rejection_reason=latest.rejection_reason,
                    )
                )
            rejection_ctx = "\n".join(parts)

        prompt = L2_VERIFY_PROMPT.format(
            problem_description=problem_description,
            code=code,
        )
        if rejection_ctx:
            prompt += f"\n\n## Previous Audit Feedback\n{rejection_ctx}"

        try:
            response = self.llm_client.generate(prompt, system=L2_VERIFY_SYSTEM)
            analysis_list = self._parse_json_response(response)
            if not analysis_list or not isinstance(analysis_list, list):
                return []

            results: List[L2VerifyResult] = []
            for analysis in analysis_list:
                confidence = analysis.get("confidence", 0.5)
                if confidence < self.confidence_threshold:
                    continue

                issue_type = analysis.get("issue_type", "unknown")
                target = analysis.get("target", "unknown")
                # Map semantic audit fields to L2VerifyResult
                evidence = analysis.get("evidence", "")
                suggested_fix = analysis.get("suggested_fix", "")
                reason = f"{evidence} | Fix: {suggested_fix}" if suggested_fix else evidence
                results.append(L2VerifyResult(
                    param=target,
                    param_role=issue_type,
                    expected_direction="n/a",       # No perturbation
                    actual_direction="n/a",          # No perturbation
                    is_violation=True,               # All returned issues are violations
                    confidence=confidence,
                    reason=reason,
                    z_plus=0.0,                      # No perturbation data
                    z_minus=0.0,
                    z_baseline=baseline_obj,
                ))
            return results
        except Exception:
            return []

    def _get_param_value(self, data: Dict, param: str) -> Any:
        """Get parameter value from data."""
        keys = param.split(".")
        obj = data
        for key in keys:
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                return None
        return obj

    def _determine_actual_direction(
        self,
        z_plus: float,
        z_minus: float,
        baseline: float
    ) -> str:
        """Determine actual objective direction from perturbation."""
        threshold = 1e-4 * max(abs(baseline), 1.0)

        # Always minimize: lower objective is better
        if z_plus < baseline - threshold:
            return "decrease_on_increase"  # Improving when param increases
        elif z_plus > baseline + threshold:
            return "increase_on_increase"  # Worsening when param increases
        else:
            return "no_effect"

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from LLM response."""
        import re
        # Try to find JSON block
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find any JSON object
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return None

    def process_repair_decisions(
        self,
        decisions: List[L2RepairDecision],
        verify_results: List[L2VerifyResult]
    ) -> Dict[str, Any]:
        """
        Process repair LLM's Accept/Reject decisions.

        Returns:
            {
                "accepted": List of accepted decisions,
                "rejected": List of rejected decisions,
                "should_reverify": Whether to re-run L2 with rejection context
            }
        """
        accepted = []
        rejected = []

        for decision in decisions:
            if decision.action == "accept":
                accepted.append(decision)
            else:
                # Record rejection history
                param = decision.param
                if param not in self.rejection_history:
                    self.rejection_history[param] = []

                # Find corresponding verify result
                verify_result = next(
                    (r for r in verify_results if r.param == param), None
                )
                verify_reason = verify_result.reason if verify_result else ""

                self.rejection_history[param].append(L2RejectionHistory(
                    param=param,
                    round=len(self.rejection_history[param]) + 1,
                    verify_reason=verify_reason,
                    rejection_reason=decision.rejection_reason
                ))
                rejected.append(decision)

        # Determine if we should re-verify
        should_reverify = (
            len(rejected) > 0 and
            not self._all_max_rejections_reached(rejected)
        )

        return {
            "accepted": accepted,
            "rejected": rejected,
            "should_reverify": should_reverify
        }

    def _all_max_rejections_reached(self, rejected: List[L2RepairDecision]) -> bool:
        """Check if all rejected params have reached max rejection count."""
        for decision in rejected:
            history = self.rejection_history.get(decision.param, [])
            if len(history) < self.max_rejections:
                return False
        return True

    def get_final_status(self, verify_results: List[L2VerifyResult]) -> str:
        """
        Determine L2 final status.

        Returns:
            "PASS": No violations or all low confidence
            "INFO": Violations exist but all max-rejected (downgraded)
            "ISSUES": Violations exist and need attention
        """
        violations = [
            r for r in verify_results
            if r.is_violation and r.confidence >= self.confidence_threshold
        ]

        if not violations:
            return "PASS"

        # Check if all violations have been max-rejected
        all_max_rejected = all(
            len(self.rejection_history.get(v.param, [])) >= self.max_rejections
            for v in violations
        )

        if all_max_rejected:
            return "INFO"  # Downgrade to INFO, don't trigger repair

        return "ISSUES"

    def format_diagnostics_for_repair(
        self,
        verify_results: List[L2VerifyResult]
    ) -> str:
        """Format L2 semantic audit results for repair prompt."""
        lines = []

        violations = [
            r for r in verify_results
            if r.is_violation and r.confidence >= self.confidence_threshold
        ]

        if not violations:
            return "No semantic issues detected."

        for i, v in enumerate(violations, 1):
            rejection_count = len(self.rejection_history.get(v.param, []))
            lines.append(f"""
### Diagnostic {i}: {v.param}
- **Issue Type**: {v.param_role}
- **Confidence**: {v.confidence:.0%}
- **Evidence & Fix**: {v.reason}
- **Previous Rejections**: {rejection_count}
""")

        return "\n".join(lines)

    def parse_repair_response(self, response: str) -> Tuple[List[L2RepairDecision], Optional[str]]:
        """Parse repair LLM response into decisions and fixed code."""
        parsed = self._parse_json_response(response)

        if not parsed:
            return [], None

        decisions = []
        for d in parsed.get("decisions", []):
            decisions.append(L2RepairDecision(
                param=d.get("param", ""),
                action=d.get("action", "reject"),
                rejection_reason=d.get("reason", "") if d.get("action") == "reject" else "",
                fix_description=d.get("reason", "") if d.get("action") == "accept" else ""
            ))

        fixed_code = parsed.get("fixed_code")

        return decisions, fixed_code


# Backward compatibility alias
L4AdversarialVerifier = L2DirectionVerifier


# =============================================================================
# Convenience Functions
# =============================================================================

def should_exit_l2_loop(
    l2_status: str,
    decisions: List[L2RepairDecision],
    other_layers_status: Dict[str, str]
) -> Tuple[bool, str]:
    """
    Determine if L2 adversarial loop should exit.

    Returns:
        (should_exit, reason)
    """
    # Condition 1: All PASS
    if l2_status == "PASS":
        return True, "all_pass"

    # Condition 2: All violations rejected + other layers PASS
    all_rejected = all(d.action == "reject" for d in decisions)
    other_pass = all(
        s in ("PASS", "INFO") for s in other_layers_status.values()
    )

    if all_rejected and other_pass:
        return True, "all_rejected_others_pass"

    # Condition 3: L2 downgraded to INFO (max rejections reached)
    if l2_status == "INFO":
        return True, "downgraded_to_info"

    return False, "continue"


# Backward compatibility alias
should_exit_l4_loop = should_exit_l2_loop
