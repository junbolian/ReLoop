"""
ReLoop Core Verification Module - 5 Layer Architecture

Layers:
- L1: Execution & Solver (Blocking Layer)
- L2: Relaxation Monotonicity (Diagnostic Layer)
- L3: Dual Consistency (Diagnostic Layer)
- L4: Solution Freedom Analysis (Diagnostic Layer)
- L5: CPT (Enhancement Layer, Optional)
"""

import time
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .executor import CodeExecutor
from .param_utils import (
    extract_numeric_params, get_param_value, infer_param_role,
    get_expected_direction, perturb_param, set_param, should_skip_param,
    ParameterRole
)


class Severity(Enum):
    FATAL = "FATAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    PASS = "PASS"


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
        obj_sense: str = "minimize",
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
        l1_results, baseline = self._layer1(code, data, obj_sense, verbose)
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

        # L2: Relaxation Monotonicity
        if verbose:
            print("\n[L2] Relaxation Monotonicity")
        l2_results = self._layer2(code, data, objective, obj_sense, params, verbose)
        layer_results.extend(l2_results)

        # L3: Dual Consistency
        if verbose:
            print("\n[L3] Dual Consistency")
        l3_results = self._layer3(objective, baseline, verbose)
        layer_results.extend(l3_results)

        # L4: Solution Freedom Analysis
        if verbose:
            print("\n[L4] Solution Freedom Analysis")
        l4_results = self._layer4(code, data, params, objective, obj_sense, complexity, verbose)
        layer_results.extend(l4_results)

        # L5: CPT (Optional)
        if enable_cpt and problem_description:
            if verbose:
                print("\n[L5] CPT")
            l5_results = self._layer5(code, data, objective, obj_sense, problem_description, verbose)
            layer_results.extend(l5_results)

        return self._aggregate(layer_results, objective, solution, complexity, start_time, verbose)

    # =========================================================================
    # L1: Execution & Solver
    # =========================================================================

    def _layer1(
        self, code: str, data: Dict, obj_sense: str, verbose: bool
    ) -> Tuple[List[LayerResult], Dict]:
        """L1: Execution and solver status check."""
        results = []

        # L1.1: Syntax check
        syntax_ok, syntax_err = self.executor.check_syntax(code)
        if not syntax_ok:
            results.append(LayerResult(
                "L1", "syntax", Severity.FATAL, f"Syntax error: {syntax_err}", 1.0
            ))
            return results, {}

        # L1.2: Execute
        baseline = self.executor.execute(code, data)
        if baseline.get("exit_code", -1) != 0:
            results.append(LayerResult(
                "L1", "runtime", Severity.FATAL,
                f"Runtime error: {baseline.get('error', '')[:500]}", 1.0
            ))
            return results, baseline

        # L1.3: Solver status
        status = baseline.get("status")
        objective = baseline.get("objective")

        if status is None:
            results.append(LayerResult(
                "L1", "output", Severity.FATAL, "No solver status output", 1.0
            ))
            return results, baseline

        if status == "INFEASIBLE":
            results.append(LayerResult(
                "L1", "feasibility", Severity.FATAL, "Model is INFEASIBLE", 1.0
            ))
            return results, baseline

        if status == "UNBOUNDED":
            results.append(LayerResult(
                "L1", "boundedness", Severity.FATAL, "Model is UNBOUNDED", 1.0
            ))
            return results, baseline

        if status == "TIMEOUT" and objective is None:
            results.append(LayerResult(
                "L1", "timeout", Severity.FATAL, "Solver timeout with no solution", 1.0
            ))
            return results, baseline

        if objective is not None and obj_sense == "minimize" and objective < -self.epsilon:
            results.append(LayerResult(
                "L1", "objective_sign", Severity.WARNING,
                f"Negative objective ({objective:.4f}) in minimization", 0.7
            ))

        results.append(LayerResult(
            "L1", "solver_status", Severity.PASS,
            f"Solver returned {status} with objective {objective}", 1.0
        ))

        if verbose:
            print(f"  [PASS] Status={status}, Objective={objective}")

        return results, baseline

    # =========================================================================
    # L2: Relaxation Monotonicity
    # =========================================================================

    def _layer2(
        self, code: str, data: Dict, baseline_obj: float,
        obj_sense: str, params: List[str], verbose: bool
    ) -> List[LayerResult]:
        """L2: Relaxation monotonicity detection."""
        results = []
        violations = []
        tested = 0
        is_minimize = obj_sense == "minimize"

        for param in params[:self.max_params]:
            skip, reason = should_skip_param(data, param)
            if skip:
                continue

            tightened = perturb_param(data, param, 1 - self.delta)
            result = self.executor.execute(code, tightened)
            tightened_obj = result.get("objective")

            if tightened_obj is None:
                continue

            tested += 1
            threshold = self.epsilon * max(abs(baseline_obj), 1.0)

            if is_minimize:
                if tightened_obj < baseline_obj - threshold:
                    violations.append({
                        "param": param,
                        "baseline": baseline_obj,
                        "tightened": tightened_obj
                    })
            else:
                if tightened_obj > baseline_obj + threshold:
                    violations.append({
                        "param": param,
                        "baseline": baseline_obj,
                        "tightened": tightened_obj
                    })

        for v in violations:
            results.append(LayerResult(
                "L2", "monotonicity", Severity.WARNING,
                f"Tightening '{v['param']}' IMPROVED objective. Check constraint direction.",
                0.8, v
            ))

        if not violations:
            results.append(LayerResult(
                "L2", "monotonicity", Severity.PASS,
                f"Monotonicity passed ({tested} tested)", 0.9
            ))

        return results

    # =========================================================================
    # L3: Dual Consistency
    # =========================================================================

    def _layer3(
        self, primal_obj: float, baseline: Dict, verbose: bool
    ) -> List[LayerResult]:
        """L3: Dual consistency detection."""
        results = []
        dual_obj = baseline.get("dual_objective")

        if dual_obj is not None:
            gap = abs(primal_obj - dual_obj)
            relative_gap = gap / (abs(primal_obj) + self.epsilon)

            if relative_gap > 0.01:
                results.append(LayerResult(
                    "L3", "duality_gap", Severity.WARNING,
                    f"Primal-dual gap: {relative_gap:.2%}", 0.7,
                    {"primal": primal_obj, "dual": dual_obj, "gap": relative_gap}
                ))
            else:
                results.append(LayerResult(
                    "L3", "duality_gap", Severity.PASS,
                    f"Duality gap: {relative_gap:.4%}", 0.9
                ))
        else:
            results.append(LayerResult(
                "L3", "duality_gap", Severity.INFO,
                "Dual objective not available", 0.5
            ))

        return results

    # =========================================================================
    # L4: Solution Freedom Analysis
    # =========================================================================

    def _layer4(
        self, code: str, data: Dict, params: List[str],
        baseline_obj: float, obj_sense: str,
        complexity: Complexity, verbose: bool
    ) -> List[LayerResult]:
        """L4: Solution freedom analysis."""
        results = []

        if complexity == Complexity.SIMPLE:
            no_effect_severity = Severity.INFO
            check_zero_obj = False
        else:
            no_effect_severity = Severity.WARNING
            check_zero_obj = True

        no_effect_params = []
        direction_violations = []

        for param in params[:self.max_params]:
            skip, reason = should_skip_param(data, param)
            if skip:
                continue

            obj_up = self.executor.execute(
                code, perturb_param(data, param, 1 + self.delta)
            ).get("objective")

            obj_down = self.executor.execute(
                code, perturb_param(data, param, 1 - self.delta)
            ).get("objective")

            if obj_up is None or obj_down is None:
                continue

            threshold = self.epsilon * max(abs(baseline_obj), 1.0)
            has_effect = abs(obj_up - baseline_obj) > threshold or abs(obj_down - baseline_obj) > threshold

            if not has_effect:
                no_effect_params.append(param)
            else:
                role = infer_param_role(param)
                if role != ParameterRole.UNKNOWN:
                    expected = get_expected_direction(role, obj_sense)
                    change = obj_up - baseline_obj
                    actual = "increase" if change > threshold else \
                             "decrease" if change < -threshold else "none"
                    if actual != "none" and actual != expected:
                        direction_violations.append({
                            "param": param,
                            "role": role.value,
                            "expected": expected,
                            "actual": actual
                        })

        for param in no_effect_params:
            results.append(LayerResult(
                "L4", "param_effect", no_effect_severity,
                f"Parameter '{param}' has NO EFFECT. Constraint may be MISSING.",
                0.8 if complexity != Complexity.SIMPLE else 0.3,
                {"param": param}
            ))

        for v in direction_violations:
            results.append(LayerResult(
                "L4", "direction", Severity.WARNING,
                f"Parameter '{v['param']}' shows UNEXPECTED direction.", 0.7, v
            ))

        if check_zero_obj and abs(baseline_obj) < self.epsilon:
            results.append(LayerResult(
                "L4", "zero_objective", Severity.WARNING,
                "Objective is ~0. Check cost terms.", 0.6
            ))

        if not no_effect_params and not direction_violations:
            results.append(LayerResult(
                "L4", "solution_freedom", Severity.PASS,
                "Solution freedom analysis passed", 0.85
            ))

        return results

    # =========================================================================
    # L5: CPT (Constraint Perturbation Testing)
    # =========================================================================

    def _layer5(
        self, code: str, data: Dict, baseline_obj: float,
        obj_sense: str, problem_description: str, verbose: bool
    ) -> List[LayerResult]:
        """L5: CPT - Safety: Only produces WARNING/INFO."""
        results = []

        try:
            # Extract candidates
            if self.llm_client:
                candidates = self._cpt_extract_candidates(problem_description, data)
            else:
                candidates = self._cpt_extract_candidates_rule_based(data)

            if not candidates:
                results.append(LayerResult(
                    "L5", "cpt_extraction", Severity.INFO,
                    "No candidate constraints extracted", 0.5
                ))
                return results

            if verbose:
                print(f"  Extracted {len(candidates)} candidates")

            missing = []
            satisfied = 0
            conflicts = 0
            skipped = 0

            for candidate in candidates[:10]:
                try:
                    test_result = self._cpt_test_candidate(
                        code, data, baseline_obj, obj_sense, candidate
                    )
                    status = test_result.get("status")
                    if status == "MISSING":
                        missing.append(test_result)
                        if verbose:
                            print(f"    [MISSING] {candidate.get('description', '')}")
                    elif status == "SATISFIED":
                        satisfied += 1
                        if verbose:
                            print(f"    [OK] {candidate.get('description', '')}")
                    elif status == "CONFLICT":
                        conflicts += 1
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1

            if missing:
                results.append(LayerResult(
                    "L5", "cpt_missing", Severity.WARNING,
                    f"CPT found {len(missing)} potentially missing constraint(s)",
                    0.75, {"missing": missing}
                ))

            results.append(LayerResult(
                "L5", "cpt_summary",
                Severity.PASS if not missing else Severity.INFO,
                f"CPT: {satisfied} satisfied, {len(missing)} missing, {conflicts} conflicts, {skipped} skipped",
                0.8 if not missing else 0.6
            ))

        except Exception as e:
            results.append(LayerResult(
                "L5", "cpt_error", Severity.INFO,
                f"CPT skipped: {str(e)[:100]}", 0.5
            ))

        return results

    def _cpt_extract_candidates(self, problem_description: str, data: Dict) -> List[Dict]:
        """LLM-based candidate extraction."""
        if not self.llm_client:
            return []

        prompt = f"""Extract ALL constraints from this optimization problem. Return JSON array only.

Problem: {problem_description}
Data keys: {list(data.keys())}

Format: [{{"description": "...", "type": "capacity|demand|balance", "parameters": ["..."]}}]"""

        try:
            response = self.llm_client.generate(prompt)
            match = re.search(r'\[[\s\S]*?\]', response)
            if match:
                return [c for c in json.loads(match.group())
                        if isinstance(c, dict) and "description" in c]
        except Exception:
            pass
        return []

    def _cpt_extract_candidates_rule_based(self, data: Dict) -> List[Dict]:
        """Rule-based candidate extraction."""
        candidates = []
        for param in extract_numeric_params(data):
            role = infer_param_role(param)
            if role == ParameterRole.CAPACITY:
                candidates.append({
                    "description": f"Should not exceed {param}",
                    "type": "capacity",
                    "parameters": [param]
                })
            elif role == ParameterRole.REQUIREMENT:
                candidates.append({
                    "description": f"Should meet {param}",
                    "type": "demand",
                    "parameters": [param]
                })
        return candidates

    def _cpt_test_candidate(
        self, code: str, data: Dict, baseline_obj: float,
        obj_sense: str, candidate: Dict
    ) -> Dict:
        """Test a single candidate constraint."""
        params = candidate.get("parameters", [])
        if not params:
            return {"status": "SKIP", "reason": "No parameters"}

        param = params[0]
        ctype = candidate.get("type", "other")

        # Create test data based on constraint type
        if ctype == "capacity":
            test_data = set_param(data, param, 0.001)
        elif ctype == "demand":
            test_data = perturb_param(data, param, 100.0)
        else:
            test_data = perturb_param(data, param, 0.01)

        result = self.executor.execute(code, test_data)
        new_status = result.get("status")
        new_obj = result.get("objective")

        if new_status == "INFEASIBLE":
            return {"status": "SATISFIED", "description": candidate.get("description", "")}

        if new_obj is not None:
            change_ratio = abs(new_obj - baseline_obj) / max(abs(baseline_obj), 1.0)
            if change_ratio > 0.5:
                return {"status": "SATISFIED", "description": candidate.get("description", "")}
            else:
                return {
                    "status": "MISSING",
                    "description": candidate.get("description", ""),
                    "baseline_obj": baseline_obj,
                    "new_obj": new_obj
                }

        return {"status": "SKIP", "reason": "Execution failed"}

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
    obj_sense: str = "minimize",
    verbose: bool = False
) -> VerificationReport:
    """Convenience verification function."""
    return ReLoopVerifier().verify(code, data, obj_sense, verbose=verbose)
