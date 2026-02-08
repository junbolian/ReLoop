"""
ReLoop Core Verification Module - 5 Layer Architecture

Layers:
- L1: Execution & Solver (Blocking Layer) → FATAL
- L2: Anomaly Detection (Diagnostic Layer) → ERROR/INFO
- L3: Dual Consistency (Diagnostic Layer) → INFO only
- L4: Adversarial Direction Analysis (Diagnostic Layer) → WARNING/INFO
- L5: CPT (Enhancement Layer, Optional) → WARNING/INFO

Severity Levels:
- FATAL: Code cannot run (L1 only)
- ERROR: Mathematically certain error, must fix (L2 anomaly - both directions improve)
- WARNING: High-confidence issue, should fix (L4 accepted, L5 cpt_missing)
- INFO: Likely normal, for reference only (L2 no_effect, L3 duality, L4 rejected)
- PASS: Check passed
"""

import time
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .executor import CodeExecutor
from .param_utils import (
    extract_numeric_params, get_param_value,
    perturb_param, set_param, should_skip_param
)


class Severity(Enum):
    FATAL = "FATAL"      # L1 only: code cannot run
    ERROR = "ERROR"      # Mathematically certain error, must fix
    WARNING = "WARNING"  # High-confidence issue, should fix
    INFO = "INFO"        # Likely normal, for reference only
    PASS = "PASS"        # Check passed


class Complexity(Enum):
    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"


@dataclass
class LayerResult:
    layer: str
    check: str
    severity: Severity
    message: str
    confidence: float
    details: Optional[Dict[str, Any]] = None


@dataclass
class VerificationReport:
    status: str
    has_solution: bool
    objective: Optional[float]
    solution: Optional[Dict[str, float]]
    confidence: float
    complexity: Optional[Complexity]
    layer_results: List[LayerResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    execution_time: float = 0.0


class ReLoopVerifier:
    """ReLoop Behavioral Verifier"""

    def __init__(
        self,
        delta: float = 0.2,
        epsilon: float = 1e-4,
        timeout: int = 60,
        llm_client=None
    ):
        self.delta = delta
        self.epsilon = epsilon
        self.executor = CodeExecutor(timeout=timeout)
        self.llm_client = llm_client
        self.max_params = 10

    def verify(
        self,
        code: str,
        data: Dict[str, Any],
        problem_description: str = "",
        enable_cpt: bool = False,
        verbose: bool = False
    ) -> VerificationReport:
        """Run complete verification pipeline."""
        start_time = time.time()
        layer_results = []

        if verbose:
            print("=" * 60)
            print("ReLoop Verification")
            print("=" * 60)

        # L1: Execution & Solver
        if verbose:
            print("\n[L1] Execution & Solver")
        l1_results, baseline = self._layer1(code, data, verbose)
        layer_results.extend(l1_results)

        if any(r.severity == Severity.FATAL for r in l1_results):
            return self._create_failed_report(layer_results, start_time)

        objective = baseline.get("objective")
        solution = baseline.get("solution")

        if objective is None:
            layer_results.append(LayerResult(
                "L1", "objective", Severity.FATAL, "No objective value", 1.0
            ))
            return self._create_failed_report(layer_results, start_time)

        # Estimate complexity
        complexity = self._estimate_complexity(code, data)
        params = extract_numeric_params(data)

        if verbose:
            print(f"\n[Complexity: {complexity.value}]")
            print(f"[Parameters: {len(params)} found]")

        # L2: Anomaly Detection (bidirectional perturbation)
        if verbose:
            print("\n[L2] Anomaly Detection")
        l2_results = self._layer2(code, data, objective, params, verbose)
        layer_results.extend(l2_results)

        # Extract L2 anomaly params to exclude from L4
        l2_anomaly_params = [
            r.details.get("param") for r in l2_results
            if r.severity == Severity.ERROR and r.details and "param" in r.details
        ]

        # L3: Dual Consistency
        if verbose:
            print("\n[L3] Dual Consistency")
        l3_results = self._layer3(objective, baseline, verbose)
        layer_results.extend(l3_results)

        # L4: Adversarial Direction Analysis (LLM-based)
        if verbose:
            print("\n[L4] Adversarial Direction Analysis")
        l4_results = self._layer4(
            code, data, params, objective, complexity, verbose,
            problem_description=problem_description,
            l2_anomaly_params=l2_anomaly_params
        )
        layer_results.extend(l4_results)

        # L5: CPT (Optional)
        if enable_cpt and problem_description:
            if verbose:
                print("\n[L5] CPT")
            l5_results = self._layer5(code, data, objective, problem_description, verbose)
            layer_results.extend(l5_results)

        return self._aggregate(layer_results, objective, solution, complexity, start_time, verbose)

    # =========================================================================
    # L1: Execution & Solver (FATAL only)
    # =========================================================================

    def _layer1(
        self, code: str, data: Dict, verbose: bool
    ) -> Tuple[List[LayerResult], Dict]:
        """L1: Execution and solver status check."""
        results = []

        # L1.1: Syntax check
        syntax_ok, syntax_err = self.executor.check_syntax(code)
        if not syntax_ok:
            results.append(LayerResult(
                "L1", "syntax", Severity.FATAL,
                f"Syntax error: {syntax_err}", 1.0,
                {"trigger_repair": True, "is_likely_normal": False}
            ))
            return results, {}

        # L1.2: Execute
        baseline = self.executor.execute(code, data)
        if baseline.get("exit_code", -1) != 0:
            results.append(LayerResult(
                "L1", "runtime", Severity.FATAL,
                f"Runtime error: {baseline.get('error', '')[:500]}", 1.0,
                {"trigger_repair": True, "is_likely_normal": False}
            ))
            return results, baseline

        # L1.3: Solver status
        status = baseline.get("status")
        objective = baseline.get("objective")

        if status is None:
            results.append(LayerResult(
                "L1", "output", Severity.FATAL,
                "No solver status output", 1.0,
                {"trigger_repair": True, "is_likely_normal": False}
            ))
            return results, baseline

        if status == "INFEASIBLE":
            results.append(LayerResult(
                "L1", "feasibility", Severity.FATAL,
                "Model is INFEASIBLE", 1.0,
                {"trigger_repair": True, "is_likely_normal": False,
                 "repair_hint": "Remove conflicting constraints or fix constraint directions"}
            ))
            return results, baseline

        if status == "UNBOUNDED":
            results.append(LayerResult(
                "L1", "boundedness", Severity.FATAL,
                "Model is UNBOUNDED", 1.0,
                {"trigger_repair": True, "is_likely_normal": False,
                 "repair_hint": "Add missing variable bounds or constraints"}
            ))
            return results, baseline

        if status == "TIMEOUT" and objective is None:
            results.append(LayerResult(
                "L1", "timeout", Severity.FATAL,
                "Solver timeout with no solution", 1.0,
                {"trigger_repair": True, "is_likely_normal": False}
            ))
            return results, baseline

        results.append(LayerResult(
            "L1", "solver_status", Severity.PASS,
            f"Solver returned {status} with objective {objective}", 1.0
        ))

        if verbose:
            print(f"  [PASS] Status={status}, Objective={objective}")

        return results, baseline

    # =========================================================================
    # L2: Anomaly Detection (bidirectional perturbation)
    # =========================================================================

    def _layer2(
        self, code: str, data: Dict, baseline_obj: float,
        params: List[str], verbose: bool
    ) -> List[LayerResult]:
        """
        L2: Anomaly Detection.

        Detects physically impossible behavior through bidirectional perturbation:
        - anomaly (both directions improve): ERROR (physically impossible)
        - no_effect: INFO (likely slack constraint)
        - high_sensitivity: INFO (for reference)

        Theory: For any valid optimization model, it is impossible for both
        increasing AND decreasing a parameter to improve the objective.
        """
        results = []
        tested = 0
        anomaly_params = []
        no_effect_params = []
        high_sensitivity_params = []
        is_minimize = True

        for param in params[:self.max_params]:
            skip, reason = should_skip_param(data, param)
            if skip:
                continue

            # Bidirectional perturbation
            result_up = self.executor.execute(
                code, perturb_param(data, param, 1 + self.delta)
            )
            result_down = self.executor.execute(
                code, perturb_param(data, param, 1 - self.delta)
            )

            obj_up = result_up.get("objective")
            obj_down = result_down.get("objective")

            # Handle infeasibility caused by perturbation
            if obj_up is None or obj_down is None:
                if verbose:
                    print(f"  [INFO] Parameter '{param}' perturbation caused infeasibility")
                results.append(LayerResult(
                    "L2", "perturbation_infeasible", Severity.INFO,
                    f"Parameter '{param}' perturbation caused infeasibility",
                    0.5,
                    {"param": param, "trigger_repair": False, "is_likely_normal": True}
                ))
                continue

            tested += 1
            threshold = self.epsilon * max(abs(baseline_obj), 1.0)
            change_up = obj_up - baseline_obj
            change_down = obj_down - baseline_obj
            has_effect = abs(change_up) > threshold or abs(change_down) > threshold

            # ────────────────────────────────────────────────────
            # Check 1: No-effect → INFO (very likely normal)
            # ────────────────────────────────────────────────────
            if not has_effect:
                no_effect_params.append({
                    "param": param,
                    "baseline": baseline_obj,
                    "obj_up": obj_up,
                    "obj_down": obj_down
                })
                continue

            # ────────────────────────────────────────────────────
            # Check 2: Anomaly (both directions improve) → ERROR
            # ────────────────────────────────────────────────────
            if is_minimize:
                up_better = change_up < -threshold
                down_better = change_down < -threshold
            else:
                up_better = change_up > threshold
                down_better = change_down > threshold

            if up_better and down_better:
                # ERROR: Physically impossible
                anomaly_params.append({
                    "param": param,
                    "baseline": baseline_obj,
                    "obj_up": obj_up,
                    "obj_down": obj_down,
                    "change_up": change_up,
                    "change_down": change_down
                })
                continue

            # ────────────────────────────────────────────────────
            # Check 3: High sensitivity → INFO
            # ────────────────────────────────────────────────────
            max_change = max(abs(change_up), abs(change_down))
            if abs(baseline_obj) > self.epsilon and max_change > 0.5 * abs(baseline_obj):
                sensitivity = max_change / abs(baseline_obj)
                high_sensitivity_params.append({
                    "param": param,
                    "sensitivity": sensitivity,
                    "change_up": change_up,
                    "change_down": change_down
                })

        # Report anomalies → ERROR
        for item in anomaly_params:
            results.append(LayerResult(
                "L2", "anomaly", Severity.ERROR,
                f"ANOMALY: Both increasing AND decreasing '{item['param']}' "
                f"IMPROVE the objective ({item['baseline']:.4f} -> "
                f"{item['obj_up']:.4f} / {item['obj_down']:.4f}). "
                f"This is physically impossible - model has structural error.",
                0.95,
                {
                    "param": item["param"],
                    "baseline_obj": item["baseline"],
                    "obj_up": item["obj_up"],
                    "obj_down": item["obj_down"],
                    "change_up": item["change_up"],
                    "change_down": item["change_down"],
                    "trigger_repair": True,
                    "is_likely_normal": False,
                    "repair_hint": f"Parameter '{item['param']}' is used incorrectly. "
                                   f"Check constraint signs, directions, and coefficients."
                }
            ))

        # Report no-effect parameters → INFO
        for item in no_effect_params:
            results.append(LayerResult(
                "L2", "no_effect", Severity.INFO,
                f"Parameter '{item['param']}' has no measurable effect on objective",
                0.3,
                {
                    "param": item["param"],
                    "baseline": item["baseline"],
                    "obj_up": item["obj_up"],
                    "obj_down": item["obj_down"],
                    "trigger_repair": False,
                    "is_likely_normal": True,
                    "note": "VERY LIKELY NORMAL - slack constraints naturally have no effect."
                }
            ))

        # Report high sensitivity → INFO
        for item in high_sensitivity_params[:3]:
            results.append(LayerResult(
                "L2", "high_sensitivity", Severity.INFO,
                f"Parameter '{item['param']}' shows high sensitivity: "
                f"{item['sensitivity']:.1%} change in objective",
                0.4,
                {
                    "param": item["param"],
                    "sensitivity": item["sensitivity"],
                    "trigger_repair": False,
                    "is_likely_normal": True,
                    "note": "High sensitivity is often normal for binding parameters"
                }
            ))

        if not anomaly_params:
            results.append(LayerResult(
                "L2", "anomaly_ok", Severity.PASS,
                f"Anomaly detection passed ({tested} parameters tested)", 0.9
            ))

        if verbose:
            print(f"  Tested {tested} params: {len(anomaly_params)} anomalies, "
                  f"{len(no_effect_params)} no-effect, {len(high_sensitivity_params)} high-sensitivity")

        return results

    # =========================================================================
    # L3: Dual Consistency (INFO only - likely numerical issues)
    # =========================================================================

    def _layer3(
        self, primal_obj: float, baseline: Dict, verbose: bool
    ) -> List[LayerResult]:
        """
        L3: Dual Consistency Detection.

        Purpose: Check primal-dual gap.
        Severity: INFO only (not WARNING).
        Reason: Large gap is often numerical precision, not modeling error.
        """
        results = []
        dual_obj = baseline.get("dual_objective")

        if dual_obj is not None:
            gap = abs(primal_obj - dual_obj)
            relative_gap = gap / (abs(primal_obj) + self.epsilon)

            if relative_gap > 0.01:
                # INFO: Large gap is likely numerical, not error
                results.append(LayerResult(
                    "L3", "duality_gap", Severity.INFO,
                    f"Primal-dual gap: {relative_gap:.2%}", 0.5,
                    {
                        "primal": primal_obj,
                        "dual": dual_obj,
                        "gap": relative_gap,
                        "trigger_repair": False,
                        "is_likely_normal": True,
                        "note": "Large gap may indicate numerical issues, not modeling errors"
                    }
                ))
            else:
                results.append(LayerResult(
                    "L3", "duality_gap", Severity.PASS,
                    f"Duality gap: {relative_gap:.4%}", 0.9
                ))
        else:
            results.append(LayerResult(
                "L3", "dual_unavailable", Severity.INFO,
                "Dual objective not available (MIP or solver limitation)", 0.5,
                {"trigger_repair": False, "is_likely_normal": True}
            ))

        return results

    # =========================================================================
    # L4: Adversarial Direction Analysis (LLM-based)
    # =========================================================================

    def _layer4(
        self, code: str, data: Dict, params: List[str],
        baseline_obj: float,
        complexity: Complexity, verbose: bool,
        problem_description: str = "",
        l2_anomaly_params: Optional[List[str]] = None
    ) -> List[LayerResult]:
        """
        L4: Adversarial Direction Analysis.

        Uses LLM to analyze parameter direction expectations and detect violations.
        Features adversarial mechanism: Repair LLM can Accept or Reject diagnostics.

        Note: This is a placeholder. Full implementation is in l4_adversarial.py
        and is called from pipeline.py. When LLM client is not available,
        L4 returns PASS (no analysis performed).

        Args:
            l2_anomaly_params: Parameters already flagged by L2 (excluded from L4)
        """
        results = []

        # If no LLM client, skip L4
        if self.llm_client is None:
            results.append(LayerResult(
                "L4", "skipped", Severity.INFO,
                "L4 Adversarial Direction Analysis skipped (no LLM client)",
                0.5,
                {"trigger_repair": False, "is_likely_normal": True}
            ))
            return results

        # Exclude parameters already flagged by L2
        l2_anomaly_params = l2_anomaly_params or []
        params_to_analyze = [p for p in params if p not in l2_anomaly_params]

        if not params_to_analyze:
            results.append(LayerResult(
                "L4", "no_params", Severity.PASS,
                "No parameters to analyze (all handled by L2)",
                0.9
            ))
            return results

        # Full L4 analysis is handled by L4AdversarialVerifier in pipeline
        # This placeholder returns PASS when called directly
        results.append(LayerResult(
            "L4", "placeholder", Severity.INFO,
            f"L4 analysis deferred to pipeline ({len(params_to_analyze)} params pending)",
            0.5,
            {
                "params_pending": params_to_analyze,
                "trigger_repair": False,
                "note": "Full L4 analysis with LLM is handled in pipeline.py"
            }
        ))

        return results

    # =========================================================================
    # L5: CPT (WARNING for missing <5%, INFO for uncertain 5-30%)
    # =========================================================================

    def _layer5(
        self, code: str, data: Dict, baseline_obj: float,
        problem_description: str, verbose: bool
    ) -> List[LayerResult]:
        """
        L5: CPT (Constraint Perturbation Testing).

        Severity grading based on change ratio:
        - < 5%: WARNING (constraint likely missing)
        - 5%-30%: INFO (uncertain, may be normal)
        - > 30%: PASS (constraint present)

        Note: L5 uses WARNING (not ERROR) because LLM-extracted
        candidates may themselves be inaccurate.
        """
        results = []

        try:
            # Extract candidates (LLM-only, no keyword fallback)
            if not self.llm_client:
                results.append(LayerResult(
                    "L5", "cpt_skipped", Severity.INFO,
                    "L5 CPT skipped (requires LLM for constraint extraction)", 0.5,
                    {"trigger_repair": False, "is_likely_normal": True}
                ))
                return results

            candidates = self._cpt_extract_candidates(problem_description, data)

            if not candidates:
                results.append(LayerResult(
                    "L5", "cpt_extraction", Severity.INFO,
                    "No candidate constraints extracted", 0.5,
                    {"trigger_repair": False, "is_likely_normal": True}
                ))
                return results

            if verbose:
                print(f"  Extracted {len(candidates)} candidates")

            for candidate in candidates[:10]:
                try:
                    test_result = self._cpt_test_candidate_v2(
                        code, data, baseline_obj, candidate, verbose
                    )
                    if test_result:
                        results.append(test_result)
                except Exception:
                    pass

        except Exception as e:
            results.append(LayerResult(
                "L5", "cpt_error", Severity.INFO,
                f"CPT skipped: {str(e)[:100]}", 0.5,
                {"trigger_repair": False, "is_likely_normal": True}
            ))

        return results

    def _cpt_test_candidate_v2(
        self, code: str, data: Dict, baseline_obj: float,
        candidate: Dict, verbose: bool
    ) -> Optional[LayerResult]:
        """Test a single candidate constraint with new thresholds."""
        params = candidate.get("parameters", [])
        if not params:
            return None

        param = params[0]
        ctype = candidate.get("type", "other")
        description = candidate.get("description", "")

        # Create extreme perturbation based on constraint type
        if ctype == "capacity":
            test_data = set_param(data, param, 0.001)
            perturbation_desc = "set to near-zero (x0.001)"
        elif ctype == "demand":
            test_data = perturb_param(data, param, 100.0)
            perturbation_desc = "scaled up 100x"
        else:
            test_data = perturb_param(data, param, 0.01)
            perturbation_desc = "scaled to 1%"

        result = self.executor.execute(code, test_data)
        new_status = result.get("status")
        new_obj = result.get("objective")

        # If became infeasible, constraint is present
        if new_status == "INFEASIBLE":
            if verbose:
                print(f"    [PRESENT] {description} - perturbation caused infeasibility")
            return LayerResult(
                "L5", "cpt_present", Severity.PASS,
                f"Constraint '{description}': extreme perturbation caused infeasibility - constraint is present",
                0.9,
                {"trigger_repair": False}
            )

        if new_obj is None:
            return None

        # Calculate change ratio
        if abs(baseline_obj) > self.epsilon:
            change_ratio = abs(new_obj - baseline_obj) / abs(baseline_obj)
        else:
            change_ratio = abs(new_obj - baseline_obj)

        # ────────────────────────────────────────────────────
        # Threshold-based grading
        # ────────────────────────────────────────────────────

        if change_ratio < 0.05:
            # < 5%: WARNING (constraint likely missing)
            if verbose:
                print(f"    [MISSING] {description} - only {change_ratio:.1%} change")
            return LayerResult(
                "L5", "cpt_missing", Severity.WARNING,
                f"Constraint '{description}' is likely MISSING: "
                f"extreme perturbation ({perturbation_desc}) caused only {change_ratio:.1%} change",
                0.75,
                {
                    "constraint_name": description,
                    "constraint_type": ctype,
                    "related_param": param,
                    "change_ratio": change_ratio,
                    "trigger_repair": True,
                    "is_likely_normal": False,
                    "repair_hint": f"Add constraint: {description}"
                }
            )

        elif change_ratio < 0.30:
            # 5%-30%: INFO (uncertain)
            if verbose:
                print(f"    [UNCERTAIN] {description} - {change_ratio:.1%} change")
            return LayerResult(
                "L5", "cpt_uncertain", Severity.INFO,
                f"Constraint '{description}': {change_ratio:.1%} change (uncertain, may be normal)",
                0.5,
                {
                    "constraint_name": description,
                    "change_ratio": change_ratio,
                    "trigger_repair": False,
                    "is_likely_normal": True,
                    "note": "Moderate change - constraint may be partially active"
                }
            )

        else:
            # > 30%: PASS (constraint present)
            if verbose:
                print(f"    [PRESENT] {description} - {change_ratio:.1%} change")
            return LayerResult(
                "L5", "cpt_present", Severity.PASS,
                f"Constraint '{description}': {change_ratio:.1%} change - constraint is active",
                0.85,
                {"trigger_repair": False}
            )

    def _cpt_extract_candidates(self, problem_description: str, data: Dict) -> List[Dict]:
        """LLM-based candidate extraction."""
        if not self.llm_client:
            return []

        prompt = f"""Analyze this optimization problem and extract the KEY CONSTRAINTS that should be present in the model.

## Problem Description
{problem_description}

## Available Data Parameters
{list(data.keys())}

## Task
Identify constraints that are REQUIRED by the problem. Focus on:
1. Capacity constraints (resource limits, maximum values)
2. Demand constraints (minimum requirements, must-satisfy conditions)
3. Balance constraints (flow balance, inventory balance)

## Output Format
Return ONLY a JSON array with this exact format:
```json
[
  {{"description": "minimum protein requirement", "type": "demand", "parameters": ["min_protein"]}},
  {{"description": "capacity limit on production", "type": "capacity", "parameters": ["capacity"]}}
]
```

Return ONLY the JSON array, no explanation."""

        try:
            response = self.llm_client.generate(prompt)

            # Try to find JSON array - use greedy match
            match = re.search(r'\[[\s\S]*\]', response)
            if match:
                json_str = match.group()
                candidates = json.loads(json_str)
                valid = [c for c in candidates
                        if isinstance(c, dict) and "description" in c and "parameters" in c]
                return valid

            # Try parsing entire response as JSON
            try:
                candidates = json.loads(response.strip())
                if isinstance(candidates, list):
                    return [c for c in candidates
                            if isinstance(c, dict) and "description" in c]
            except json.JSONDecodeError:
                pass

        except json.JSONDecodeError as e:
            # JSON parsing failed - log but continue
            pass
        except Exception as e:
            # Other error - log but continue
            pass

        return []

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _estimate_complexity(self, code: str, data: Dict) -> Complexity:
        """Estimate problem complexity."""
        n_addvar = len(re.findall(r'\.addVar\s*\(', code))
        n_addvars = len(re.findall(r'\.addVars\s*\(', code))
        n_addconstr = len(re.findall(r'\.addConstr\s*\(', code))
        n_addconstrs = len(re.findall(r'\.addConstrs\s*\(', code))

        est_vars = n_addvar + n_addvars * 10
        est_constrs = n_addconstr + n_addconstrs * 10

        if est_vars <= 5 and est_constrs <= 5 and len(data) <= 10:
            return Complexity.SIMPLE
        elif est_vars <= 50 and est_constrs <= 50:
            return Complexity.MEDIUM
        else:
            return Complexity.COMPLEX

    def _create_failed_report(
        self, results: List[LayerResult], start_time: float
    ) -> VerificationReport:
        """Create failed report."""
        return VerificationReport(
            status='FAILED',
            has_solution=False,
            objective=None,
            solution=None,
            confidence=1.0,
            complexity=None,
            layer_results=results,
            recommendations=[r.message for r in results if r.severity == Severity.FATAL],
            execution_time=time.time() - start_time
        )

    def _aggregate(
        self, results: List[LayerResult], objective: float,
        solution: Optional[Dict], complexity: Complexity,
        start_time: float, verbose: bool
    ) -> VerificationReport:
        """Aggregate results."""
        severities = [r.severity for r in results]

        if Severity.FATAL in severities:
            status = 'FAILED'
        elif Severity.ERROR in severities:
            status = 'ERRORS'
        elif Severity.WARNING in severities:
            status = 'WARNINGS'
        else:
            status = 'VERIFIED'

        fatal = sum(1 for s in severities if s == Severity.FATAL)
        error = sum(1 for s in severities if s == Severity.ERROR)
        warning = sum(1 for s in severities if s == Severity.WARNING)
        confidence = max(0.1, 1.0 - 0.3*fatal - 0.2*error - 0.1*warning)

        recommendations = [
            f"[{r.layer}.{r.check}] {r.message}"
            for r in results
            if r.severity in (Severity.FATAL, Severity.ERROR, Severity.WARNING)
        ]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Result: {status} (confidence: {confidence:.2f})")
            print(f"Objective: {objective}")

        return VerificationReport(
            status=status,
            has_solution=True,
            objective=objective,
            solution=solution,
            confidence=confidence,
            complexity=complexity,
            layer_results=results,
            recommendations=recommendations,
            execution_time=time.time() - start_time
        )


# =============================================================================
# Convenience Function
# =============================================================================

def verify_code(
    code: str,
    data: Dict[str, Any],
    verbose: bool = False
) -> VerificationReport:
    """Convenience verification function."""
    return ReLoopVerifier().verify(code, data, verbose=verbose)
