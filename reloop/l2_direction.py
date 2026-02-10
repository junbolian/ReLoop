"""
ReLoop L2: Direction Consistency Analysis (Adversarial)

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
    """Result of L2 direction verification for a single parameter."""
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

L2_VERIFY_SYSTEM = """You are an optimization expert analyzing parameter behavior in mathematical models.
Your task is to determine what behavior we SHOULD expect from each parameter, and whether the observed behavior matches."""

L2_VERIFY_PROMPT = '''Analyze the parameter behavior in this optimization problem.

## Problem Description
{problem_description}

## Optimization Sense
{sense}

## Generated Code
```python
{code}
```

## Parameters to Analyze (batched)
For EACH entry below, first reason about the expected direction, then decide violation.

{param_block}

## Important Notes
- For MINIMIZATION: increasing a cost coefficient -> objective INCREASES (worse)
- For MAXIMIZATION: increasing a revenue coefficient -> objective INCREASES (better)
- Constraint bounds vs objective coefficients behave differently.
- Consider whether the parameter appears in constraints or the objective function.

## Confidence Guidelines
- 0.9+: Very certain (clear constraint bound or objective coefficient)
- 0.7-0.9: Fairly certain (typical parameter role)
- 0.5-0.7: Uncertain (ambiguous role)
- <0.5: Low confidence (don't trigger repair based on this)

Return ONLY JSON (no text) in this format:
```json
[
  {{
    "param": "param_name",
    "param_role": "constraint_bound" | "objective_coef" | "other",
    "role_explanation": "...",
    "expected_direction": "increase" | "decrease" | "no_effect" | "uncertain",
    "direction_explanation": "...",
    "is_violation": true | false,
    "violation_explanation": "...",
    "confidence": 0.0 to 1.0,
    "suggested_fix": "..."
  }}
]
```
Include one object per parameter, in the same order.
'''

L2_REJECTION_CONTEXT = """
## Previous Analysis Was Rejected
**Your previous analysis**: {previous_reason}
**Rejection reason from repair agent**: {rejection_reason}

Please reconsider your analysis in light of this feedback.
If the rejection reason is valid, adjust your conclusion.
If you still believe your original analysis was correct, provide stronger justification.
"""

L2_REPAIR_PROMPT = '''Review these L2 direction analysis diagnostics and decide how to handle each.

## Problem Description
{problem_description}

## Current Code
```python
{code}
```

## L2 Diagnostics

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
            "reason": "If reject: why L2 analysis is wrong. If accept: description of the fix needed"
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
# L2 Direction Consistency Verifier (Adversarial)
# =============================================================================

class L2DirectionVerifier:
    """
    L2: Direction Consistency Analysis (Adversarial)

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
        Verify parameter directions using a single batched LLM analysis call.

        Args:
            code: Generated optimization code
            data: Problem data dictionary
            baseline_obj: Baseline objective value
            problem_description: Natural language problem description
            params: Parameters to analyze (if None, extract from data)
            exclude_params: Parameters to skip (e.g., already flagged)
            executor: Code executor (for perturbation tests)

        Returns:
            List of L2VerifyResult for each analyzed parameter
        """
        # Get parameters to analyze
        if params is None:
            if mode == "source_code":
                params = get_source_code_param_names(code)
            elif mode == "hybrid":
                data_params = extract_numeric_params(data)
                code_params = get_source_code_param_names(code)
                params = data_params + [p for p in code_params if p not in data_params]
            else:
                params = extract_numeric_params(data)

        exclude_params = exclude_params or []
        # Analyze all numeric params (excluding those flagged elsewhere)
        params_to_analyze = [p for p in params if p not in exclude_params]

        param_entries = []
        for param in params_to_analyze:
            if mode != "source_code":
                param_in_data = get_param_value(data, param) is not None
                if param_in_data:
                    skip, _ = should_skip_param(data, param)
                    if skip:
                        continue

            if executor is None:
                continue

            z_plus, z_minus = self._perturb_and_solve(
                executor, code, data, param, mode, baseline_obj
            )
            if z_plus is None or z_minus is None:
                continue

            change_plus = ((z_plus - baseline_obj) / abs(baseline_obj) * 100
                           if abs(baseline_obj) > 1e-6 else z_plus - baseline_obj)
            change_minus = ((z_minus - baseline_obj) / abs(baseline_obj) * 100
                            if abs(baseline_obj) > 1e-6 else z_minus - baseline_obj)

            param_entries.append({
                "param": param,
                "value": self._get_param_value(data, param),
                "z_plus": z_plus,
                "z_minus": z_minus,
                "change_plus": change_plus,
                "change_minus": change_minus,
                "rejection_context": self._build_rejection_context(param)
            })

        if not param_entries:
            return []

        analyses = self._analyze_params_batch(
            code=code,
            problem_description=problem_description,
            baseline_obj=baseline_obj,
            param_entries=param_entries
        )

        return analyses

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

    def _analyze_params_batch(
        self,
        code: str,
        problem_description: str,
        baseline_obj: float,
        param_entries: List[Dict[str, Any]]
    ) -> List[L2VerifyResult]:
        """Analyze multiple parameters in one LLM call."""
        lines = []
        for i, p in enumerate(param_entries, 1):
            lines.append(
                f"""### Param {i}: {p['param']}
- Current value: {p['value']}
- Baseline objective: {baseline_obj:.4f}
- +{int(self.delta*100)}% => {p['z_plus']:.4f} (change: {p['change_plus']:+.2f}%)
- -{int(self.delta*100)}% => {p['z_minus']:.4f} (change: {p['change_minus']:+.2f}%)
{p['rejection_context']}
"""
            )
        param_block = "\n".join(lines)

        prompt = L2_VERIFY_PROMPT.format(
            problem_description=problem_description,
            sense="minimize",
            code=code,
            param_block=param_block
        )

        try:
            response = self.llm_client.generate(prompt, system=L2_VERIFY_SYSTEM)
            analysis_list = self._parse_json_response(response)
            if not analysis_list or not isinstance(analysis_list, list):
                return []

            results: List[L2VerifyResult] = []
            for analysis in analysis_list:
                param_name = analysis.get("param")
                entry = next((p for p in param_entries if p["param"] == param_name), None)
                if not entry:
                    continue

                actual_direction = self._determine_actual_direction(
                    entry["z_plus"], entry["z_minus"], baseline_obj
                )

                results.append(L2VerifyResult(
                    param=param_name,
                    param_role=analysis.get("param_role", "other"),
                    expected_direction=analysis.get("expected_direction", "uncertain"),
                    actual_direction=actual_direction,
                    is_violation=analysis.get("is_violation", False),
                    confidence=analysis.get("confidence", 0.5),
                    reason=analysis.get("direction_explanation", ""),
                    z_plus=entry["z_plus"],
                    z_minus=entry["z_minus"],
                    z_baseline=baseline_obj
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
        """Format L2 results for repair prompt."""
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
