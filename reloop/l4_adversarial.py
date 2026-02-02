"""
ReLoop L4: Adversarial Direction Analysis

LLM-based parameter direction verification with adversarial rejection mechanism.

Design:
- LLM_verify: Analyzes each parameter's expected direction
- LLM_repair: Can Accept (fix) or Reject (with reason) diagnostics
- Rejection triggers re-analysis with context
- Exit when: all PASS, all rejected + others PASS, or max iterations
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .param_utils import extract_numeric_params, perturb_param, should_skip_param


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class L4VerifyResult:
    """Result of L4 verification for a single parameter."""
    param: str
    param_role: str              # "constraint_bound" | "objective_coef" | "other"
    expected_direction: str      # "increase" | "decrease" | "no_effect" | "uncertain"
    actual_direction: str        # Observed direction from perturbation
    is_violation: bool           # Whether behavior violates expectation
    confidence: float            # 0.0 - 1.0
    reason: str                  # Explanation of the analysis
    z_plus: float               # Objective after param increase
    z_minus: float              # Objective after param decrease
    z_baseline: float           # Baseline objective


@dataclass
class L4RepairDecision:
    """Decision from repair LLM for a single parameter."""
    param: str
    action: str                  # "accept" | "reject"
    rejection_reason: str        # Detailed reason if rejected
    fix_description: str         # Description of fix if accepted


@dataclass
class L4RejectionHistory:
    """History of rejections for re-analysis context."""
    param: str
    round: int
    verify_reason: str
    rejection_reason: str


@dataclass
class L4Result:
    """Aggregated L4 result."""
    status: str                  # "PASS" | "ISSUES" | "INFO"
    verify_results: List[L4VerifyResult]
    decisions: List[L4RepairDecision]
    rejection_history: Dict[str, List[L4RejectionHistory]]


# =============================================================================
# Prompts
# =============================================================================

L4_VERIFY_SYSTEM = """You are an optimization expert analyzing parameter behavior in mathematical models.
Your task is to determine what behavior we SHOULD expect from each parameter, and whether the observed behavior matches."""

L4_VERIFY_PROMPT = '''Analyze the parameter behavior in this optimization problem.

## Problem Description
{problem_description}

## Optimization Sense
{sense}

## Generated Code
```python
{code}
```

## Parameter to Analyze
**Parameter name**: {param}
**Current value**: {param_value}

## Perturbation Test Results
- Baseline objective: {z_baseline:.4f}
- After {param} increased by {delta_pct}%: z = {z_plus:.4f} (change: {change_plus:+.2f}%)
- After {param} decreased by {delta_pct}%: z = {z_minus:.4f} (change: {change_minus:+.2f}%)

{rejection_context}

## Your Task
Analyze what behavior we SHOULD expect from this parameter, and whether the observed behavior matches.

Think step by step:
1. What role does '{param}' play in this problem? (constraint bound? objective coefficient? other?)
2. Based on the problem description and optimization sense ({sense}), what should happen to the objective when '{param}' increases?
3. Does the observed behavior match this expectation?

## Important Notes
- For MINIMIZATION: increasing a cost coefficient -> objective INCREASES (worse)
- For MAXIMIZATION: increasing a revenue coefficient -> objective INCREASES (better)
- Constraint bounds vs objective coefficients behave differently!
- Consider whether the parameter appears in constraints or the objective function

## Confidence Guidelines
- 0.9+: Very certain (clear constraint bound or objective coefficient)
- 0.7-0.9: Fairly certain (typical parameter role)
- 0.5-0.7: Uncertain (ambiguous role)
- <0.5: Low confidence (don't trigger repair based on this)

Return your analysis as JSON only (no other text):
```json
{{
    "param_role": "constraint_bound" | "objective_coef" | "other",
    "role_explanation": "Why you classified it this way",
    "expected_direction": "increase" | "decrease" | "no_effect" | "uncertain",
    "direction_explanation": "Expected relationship between {param} and objective",
    "actual_behavior": "What the test results show",
    "is_violation": true | false,
    "violation_explanation": "If violation, what's wrong. If not, why behavior is correct.",
    "confidence": 0.0 to 1.0,
    "suggested_fix": "If violation, what might be wrong in the code"
}}
```
'''

L4_REJECTION_CONTEXT = """
## Previous Analysis Was Rejected
**Your previous analysis**: {previous_reason}
**Rejection reason from repair agent**: {rejection_reason}

Please reconsider your analysis in light of this feedback.
If the rejection reason is valid, adjust your conclusion.
If you still believe your original analysis was correct, provide stronger justification.
"""

L4_REPAIR_PROMPT = '''Review these L4 direction analysis diagnostics and decide how to handle each.

## Problem Description
{problem_description}

## Current Code
```python
{code}
```

## L4 Diagnostics

{l4_diagnostics}

## For Each Diagnostic, Choose One:

### Option 1: ACCEPT
If you agree the analysis is correct, the code will be fixed accordingly.

### Option 2: REJECT
If you believe the analysis is WRONG, provide a detailed rejection reason.
Your rejection will trigger re-analysis with your feedback.

**When to REJECT (examples):**
1. "wage is classified as constraint_bound, but it's actually an objective coefficient"
2. "The expected direction assumes minimization, but context suggests maximization"
3. "The parameter is correctly used; the 'violation' is actually expected behavior"

**When to ACCEPT:**
1. The analysis correctly identifies a constraint direction error
2. The parameter role is correctly identified AND behavior violates expectation
3. You cannot find a reason to reject the analysis

## Response Format
Return JSON only (no other text):
```json
{{
    "decisions": [
        {{
            "param": "parameter_name",
            "action": "accept" | "reject",
            "reason": "If reject: why L4 analysis is wrong. If accept: description of the fix needed"
        }}
    ],
    "fixed_code": "Complete fixed code if any accepts, otherwise null"
}}
```
'''


# =============================================================================
# L4 Adversarial Verifier
# =============================================================================

class L4AdversarialVerifier:
    """
    L4: Adversarial Direction Analysis

    Uses LLM to analyze parameter direction expectations.
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
        self.rejection_history: Dict[str, List[L4RejectionHistory]] = {}

    def reset(self):
        """Reset rejection history for new verification session."""
        self.rejection_history = {}

    def verify(
        self,
        code: str,
        data: Dict,
        baseline_obj: float,
        problem_description: str,
        obj_sense: str = "minimize",
        params: Optional[List[str]] = None,
        exclude_params: Optional[List[str]] = None,
        executor=None
    ) -> List[L4VerifyResult]:
        """
        Verify parameter directions using LLM analysis.

        Args:
            code: Generated optimization code
            data: Problem data dictionary
            baseline_obj: Baseline objective value
            problem_description: Natural language problem description
            obj_sense: "minimize" or "maximize"
            params: Parameters to analyze (if None, extract from data)
            exclude_params: Parameters to skip (e.g., already flagged by L2)
            executor: Code executor (for perturbation tests)

        Returns:
            List of L4VerifyResult for each analyzed parameter
        """
        results = []

        # Get parameters to analyze
        if params is None:
            params = extract_numeric_params(data)

        exclude_params = exclude_params or []
        params_to_analyze = [p for p in params if p not in exclude_params]

        for param in params_to_analyze[:10]:  # Limit to 10 params
            skip, reason = should_skip_param(data, param)
            if skip:
                continue

            # Get perturbation results
            if executor:
                z_plus, z_minus = self._perturb_and_solve(
                    executor, code, data, param
                )
            else:
                # If no executor, we can't do perturbation tests
                continue

            if z_plus is None or z_minus is None:
                continue

            # Build rejection context if this param was previously rejected
            rejection_ctx = self._build_rejection_context(param)

            # Call LLM for analysis
            result = self._analyze_param(
                code, data, param, baseline_obj,
                z_plus, z_minus, problem_description,
                obj_sense, rejection_ctx
            )

            if result:
                results.append(result)

        return results

    def _perturb_and_solve(
        self, executor, code: str, data: Dict, param: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Perturb parameter and get objective values."""
        data_plus = perturb_param(data, param, 1 + self.delta)
        data_minus = perturb_param(data, param, 1 - self.delta)

        result_plus = executor.execute(code, data_plus)
        result_minus = executor.execute(code, data_minus)

        return result_plus.get("objective"), result_minus.get("objective")

    def _build_rejection_context(self, param: str) -> str:
        """Build rejection context for re-analysis."""
        if param not in self.rejection_history or not self.rejection_history[param]:
            return ""

        latest = self.rejection_history[param][-1]
        return L4_REJECTION_CONTEXT.format(
            previous_reason=latest.verify_reason,
            rejection_reason=latest.rejection_reason
        )

    def _analyze_param(
        self,
        code: str,
        data: Dict,
        param: str,
        baseline_obj: float,
        z_plus: float,
        z_minus: float,
        problem_description: str,
        obj_sense: str,
        rejection_ctx: str
    ) -> Optional[L4VerifyResult]:
        """Analyze a single parameter using LLM."""
        # Calculate changes
        change_plus = ((z_plus - baseline_obj) / abs(baseline_obj) * 100
                       if abs(baseline_obj) > 1e-6 else z_plus - baseline_obj)
        change_minus = ((z_minus - baseline_obj) / abs(baseline_obj) * 100
                        if abs(baseline_obj) > 1e-6 else z_minus - baseline_obj)

        # Get param value for context
        param_value = self._get_param_value(data, param)

        prompt = L4_VERIFY_PROMPT.format(
            problem_description=problem_description,
            sense=obj_sense,
            code=code,
            param=param,
            param_value=param_value,
            z_baseline=baseline_obj,
            z_plus=z_plus,
            z_minus=z_minus,
            delta_pct=int(self.delta * 100),
            change_plus=change_plus,
            change_minus=change_minus,
            rejection_context=rejection_ctx
        )

        try:
            response = self.llm_client.generate(prompt, system=L4_VERIFY_SYSTEM)
            analysis = self._parse_json_response(response)

            if not analysis:
                return None

            # Determine actual direction
            actual_direction = self._determine_actual_direction(
                z_plus, z_minus, baseline_obj, obj_sense
            )

            return L4VerifyResult(
                param=param,
                param_role=analysis.get("param_role", "other"),
                expected_direction=analysis.get("expected_direction", "uncertain"),
                actual_direction=actual_direction,
                is_violation=analysis.get("is_violation", False),
                confidence=analysis.get("confidence", 0.5),
                reason=analysis.get("direction_explanation", ""),
                z_plus=z_plus,
                z_minus=z_minus,
                z_baseline=baseline_obj
            )
        except Exception as e:
            return None

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
        baseline: float,
        sense: str
    ) -> str:
        """Determine actual objective direction from perturbation."""
        threshold = 1e-4 * max(abs(baseline), 1.0)

        if sense == "minimize":
            if z_plus < baseline - threshold:
                return "decrease_on_increase"  # Improving when param increases
            elif z_plus > baseline + threshold:
                return "increase_on_increase"  # Worsening when param increases
            else:
                return "no_effect"
        else:  # maximize
            if z_plus > baseline + threshold:
                return "increase_on_increase"  # Improving when param increases
            elif z_plus < baseline - threshold:
                return "decrease_on_increase"  # Worsening when param increases
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
        decisions: List[L4RepairDecision],
        verify_results: List[L4VerifyResult]
    ) -> Dict[str, Any]:
        """
        Process repair LLM's Accept/Reject decisions.

        Returns:
            {
                "accepted": List of accepted decisions,
                "rejected": List of rejected decisions,
                "should_reverify": Whether to re-run L4 with rejection context
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

                self.rejection_history[param].append(L4RejectionHistory(
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

    def _all_max_rejections_reached(self, rejected: List[L4RepairDecision]) -> bool:
        """Check if all rejected params have reached max rejection count."""
        for decision in rejected:
            history = self.rejection_history.get(decision.param, [])
            if len(history) < self.max_rejections:
                return False
        return True

    def get_final_status(self, verify_results: List[L4VerifyResult]) -> str:
        """
        Determine L4 final status.

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
        verify_results: List[L4VerifyResult]
    ) -> str:
        """Format L4 results for repair prompt."""
        lines = []

        violations = [
            r for r in verify_results
            if r.is_violation and r.confidence >= self.confidence_threshold
        ]

        if not violations:
            return "No direction violations detected."

        for i, v in enumerate(violations, 1):
            rejection_count = len(self.rejection_history.get(v.param, []))
            lines.append(f"""
### Diagnostic {i}: {v.param}
- **Role**: {v.param_role}
- **Expected Direction**: {v.expected_direction}
- **Actual Behavior**: {v.actual_direction}
- **Confidence**: {v.confidence:.0%}
- **Reason**: {v.reason}
- **Perturbation Results**: baseline={v.z_baseline:.4f}, +{int(self.delta*100)}%={v.z_plus:.4f}, -{int(self.delta*100)}%={v.z_minus:.4f}
- **Previous Rejections**: {rejection_count}
""")

        return "\n".join(lines)

    def parse_repair_response(self, response: str) -> Tuple[List[L4RepairDecision], Optional[str]]:
        """Parse repair LLM response into decisions and fixed code."""
        parsed = self._parse_json_response(response)

        if not parsed:
            return [], None

        decisions = []
        for d in parsed.get("decisions", []):
            decisions.append(L4RepairDecision(
                param=d.get("param", ""),
                action=d.get("action", "reject"),
                rejection_reason=d.get("reason", "") if d.get("action") == "reject" else "",
                fix_description=d.get("reason", "") if d.get("action") == "accept" else ""
            ))

        fixed_code = parsed.get("fixed_code")

        return decisions, fixed_code


# =============================================================================
# Convenience Functions
# =============================================================================

def should_exit_l4_loop(
    l4_status: str,
    decisions: List[L4RepairDecision],
    other_layers_status: Dict[str, str]
) -> Tuple[bool, str]:
    """
    Determine if L4 loop should exit.

    Returns:
        (should_exit, reason)
    """
    # Condition 1: All PASS
    if l4_status == "PASS":
        return True, "all_pass"

    # Condition 2: All violations rejected + other layers PASS
    all_rejected = all(d.action == "reject" for d in decisions)
    other_pass = all(
        s in ("PASS", "INFO") for s in other_layers_status.values()
    )

    if all_rejected and other_pass:
        return True, "all_rejected_others_pass"

    # Condition 3: L4 downgraded to INFO (max rejections reached)
    if l4_status == "INFO":
        return True, "downgraded_to_info"

    return False, "continue"
