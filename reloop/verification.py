"""
ReLoop Core Verification Module - 2 Layer Architecture

Layers:
- L1: Execution Verification (Blocking Layer) -> FATAL + duality INFO
- L2: Behavioral Testing (Diagnostic Layer) -> WARNING/INFO
  - CPT: Constraint Presence Testing
  - OPT: Objective Presence Testing

Severity Levels:
- FATAL: Code cannot run (L1 only)
- WARNING: High-confidence issue, should fix (L2 CPT/OPT missing)
- INFO: Likely normal, for reference only (L1 duality, L2 uncertain)
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
from .perturbation import (
    detect_perturbation_mode, run_perturbation,
    get_source_code_param_names, perturb_code,
    extract_perturbable_params, _match_param,
    has_data_override, strip_data_override,
)


class Severity(Enum):
    FATAL = "FATAL"      # L1 only: code cannot run
    ERROR = "ERROR"      # Reserved for future use
    WARNING = "WARNING"  # High-confidence issue, should fix
    INFO = "INFO"        # Likely normal, for reference only
    PASS = "PASS"        # Check passed


class Complexity(Enum):
    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"


@dataclass
class Diagnostic:
    """Unified diagnostic schema for all verification layers."""
    layer: str              # "L1", "L2"
    issue_type: str         # "INFEASIBLE", "UNBOUNDED", "RUNTIME_ERROR",
                            # "SYNTAX_ERROR", "NO_OUTPUT", "TIMEOUT",
                            # "DUALITY_GAP",
                            # "MISSING_CONSTRAINT", "MISSING_OBJECTIVE_TERM"
    severity: str           # "WARNING", "INFO"
    target_name: str        # which parameter/constraint, e.g. "capacity"
    evidence: str           # auto-generated evidence description
    triggers_repair: bool = True


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
        delta: float = 0.1,
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

        # L1: Execution Verification (+ duality check)
        if verbose:
            print("\n[L1] Execution Verification")
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

        # Detect perturbation mode
        mode = detect_perturbation_mode(code, data)
        if mode == "source_code":
            params = get_source_code_param_names(code)
        elif mode == "hybrid":
            data_params = extract_numeric_params(data)
            code_params = get_source_code_param_names(code)
            params = data_params + [p for p in code_params if p not in data_params]
        else:
            params = extract_numeric_params(data)

        if verbose:
            print(f"\n[Complexity: {complexity.value}]")
            print(f"[Parameters: {len(params)} found]")

        # L2: Behavioral Testing (CPT + OPT, Optional)
        if enable_cpt and problem_description:
            if verbose:
                print("\n[L2] Behavioral Testing")

            # L2-CPT: Constraint Presence Testing
            if verbose:
                print("  [L2-CPT] Constraint Presence Testing")
            cpt_results = self._layer2_cpt(code, data, objective, problem_description, verbose, mode)
            layer_results.extend(cpt_results)

            # L2-OPT: Objective Presence Testing
            if verbose:
                print("  [L2-OPT] Objective Presence Testing")
            opt_results = self._layer2_opt(code, data, objective, problem_description, verbose, mode)
            layer_results.extend(opt_results)

        return self._aggregate(layer_results, objective, solution, complexity, start_time, verbose)

    # =========================================================================
    # L1: Execution Verification (Blocking) + Duality Check
    # =========================================================================

    def _layer1(
        self, code: str, data: Dict, verbose: bool
    ) -> Tuple[List[LayerResult], Dict]:
        """
        L1: Execution Verification (blocking).

        Steps:
        1. Syntax check
        2. Execute code
        3. Solver status check (INFEASIBLE/UNBOUNDED/TIMEOUT)
        4. If OPTIMAL: extract objective + run duality check as add-on diagnostic
        """
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
            # Build enhanced message with IIS diagnostics if available
            iis_constrs = baseline.get("iis_constraints")
            iis_bounds = baseline.get("iis_bounds")
            if iis_constrs or iis_bounds:
                n_c = len(iis_constrs) if iis_constrs else 0
                n_b = len(iis_bounds) if iis_bounds else 0
                constr_list = "\n".join(
                    f"  - {c}" for c in (iis_constrs or [])
                )
                bound_list = "\n".join(
                    f"  - {b}" for b in (iis_bounds or [])
                )
                iis_msg = (
                    f"Model is INFEASIBLE. Gurobi IIS analysis identified "
                    f"{n_c} conflicting constraints and {n_b} conflicting "
                    f"variable bounds that cannot be simultaneously satisfied."
                    f"\n\nConflicting constraints:\n{constr_list}"
                )
                if iis_bounds:
                    iis_msg += f"\n\nConflicting variable bounds:\n{bound_list}"
                iis_msg += (
                    "\n\nCommon causes:\n"
                    "1. Missing slack/surplus variable (e.g. no lost_sales "
                    "variable when supply < demand)\n"
                    "2. Constraint direction error (<= written as >=)\n"
                    "3. Overly tight bounds relative to data values\n"
                    "4. Incorrect indexing causing wrong parameter values"
                )
                details = {
                    "trigger_repair": True, "is_likely_normal": False,
                    "iis_constraints": iis_constrs,
                    "iis_bounds": iis_bounds,
                    "repair_hint": iis_msg,
                }
            else:
                iis_msg = (
                    "Model is INFEASIBLE. No IIS details available.\n"
                    "Please check: (1) are slack variables present where needed? "
                    "(2) are constraint directions correct? "
                    "(3) are bounds reasonable?"
                )
                details = {
                    "trigger_repair": True, "is_likely_normal": False,
                    "repair_hint": iis_msg,
                }

            results.append(LayerResult(
                "L1", "feasibility", Severity.FATAL,
                iis_msg, 1.0, details
            ))
            return results, baseline

        if status == "UNBOUNDED":
            ub_vars = baseline.get("unbounded_vars")
            if ub_vars:
                var_list = "\n".join(f"  - {v}" for v in ub_vars)
                ub_msg = (
                    f"Model is UNBOUNDED. The following {len(ub_vars)} "
                    f"variable(s) have unbounded rays:\n{var_list}\n\n"
                    "This means these variables can be increased/decreased "
                    "infinitely to improve the objective. Common causes:\n"
                    "1. A decision variable that should be bounded is not "
                    "(e.g. order quantity has no capacity constraint)\n"
                    "2. A constraint that should limit the objective is missing\n"
                    "3. Wrong sign in the objective function"
                )
            else:
                ub_msg = (
                    "Model is UNBOUNDED. The objective can be improved "
                    "infinitely. Common causes:\n"
                    "1. A decision variable without upper bound or "
                    "capacity constraint\n"
                    "2. A missing constraint that should limit the objective\n"
                    "3. Wrong sign in the objective function"
                )
            results.append(LayerResult(
                "L1", "boundedness", Severity.FATAL,
                ub_msg, 1.0,
                {"trigger_repair": True, "is_likely_normal": False,
                 "unbounded_vars": ub_vars,
                 "repair_hint": ub_msg}
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

        # L1 add-on: Duality check (if OPTIMAL)
        duality_results = self._check_duality(objective, baseline, verbose)
        results.extend(duality_results)

        return results, baseline

    # =========================================================================
    # Duality Check (internal helper, called by L1 after OPTIMAL)
    # =========================================================================

    def _check_duality(
        self, primal_obj: float, baseline: Dict, verbose: bool
    ) -> List[LayerResult]:
        """
        Duality check: verify primal-dual consistency.

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
                    "L1", "duality_gap", Severity.INFO,
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
                    "L1", "duality_gap", Severity.PASS,
                    f"Duality gap: {relative_gap:.4%}", 0.9
                ))
        else:
            results.append(LayerResult(
                "L1", "dual_unavailable", Severity.INFO,
                "Dual objective not available (MIP or solver limitation)", 0.5,
                {"trigger_repair": False, "is_likely_normal": True}
            ))

        return results

    # =========================================================================
    # L2: Behavioral Testing
    # =========================================================================

    # ----- L2-CPT: Constraint Presence Testing -----

    def _layer2_cpt(
        self, code: str, data: Dict, baseline_obj: float,
        problem_description: str, verbose: bool, mode: str = "data_dict"
    ) -> List[LayerResult]:
        """
        L2-CPT: Constraint Presence Testing.

        Severity grading based on change ratio:
        - < 5%: WARNING (constraint likely missing)
        - 5%-30%: INFO (uncertain, may be normal)
        - > 30%: PASS (constraint present)
        """
        results = []

        try:
            if not self.llm_client:
                results.append(LayerResult(
                    "L2", "cpt_skipped", Severity.INFO,
                    "L2 CPT skipped (requires LLM for constraint extraction)", 0.5,
                    {"trigger_repair": False, "is_likely_normal": True}
                ))
                return results

            candidates = self._cpt_extract_candidates(problem_description, data)

            if not candidates:
                results.append(LayerResult(
                    "L2", "cpt_extraction", Severity.INFO,
                    "No candidate constraints extracted", 0.5,
                    {"trigger_repair": False, "is_likely_normal": True}
                ))
                return results

            if verbose:
                print(f"    Extracted {len(candidates)} constraint candidates")

            for candidate in candidates[:10]:
                try:
                    test_result = self._cpt_test_candidate(
                        code, data, baseline_obj, candidate, verbose, mode
                    )
                    if test_result:
                        results.append(test_result)
                except Exception:
                    pass

        except Exception as e:
            results.append(LayerResult(
                "L2", "cpt_error", Severity.INFO,
                f"CPT skipped: {str(e)[:100]}", 0.5,
                {"trigger_repair": False, "is_likely_normal": True}
            ))

        return results

    def _cpt_test_candidate(
        self, code: str, data: Dict, baseline_obj: float,
        candidate: Dict, verbose: bool, mode: str = "data_dict"
    ) -> Optional[LayerResult]:
        """Test a single candidate constraint."""
        params = candidate.get("parameters", [])
        if not params:
            return None

        param = params[0]
        ctype = candidate.get("type", "other")
        description = candidate.get("description", "")

        # Determine perturbation factor based on constraint type
        if ctype == "capacity":
            code_factor = 0.001
            perturbation_desc = "set to near-zero (x0.001)"
        elif ctype == "demand":
            code_factor = 100.0
            perturbation_desc = "scaled up 100x"
        else:
            code_factor = 0.01
            perturbation_desc = "scaled to 1%"

        new_status = None
        new_obj = None

        # Strip json.loads data override so perturbed data dict is used
        exec_code = strip_data_override(code) if has_data_override(code) else code

        # Strategy 1: data-dict perturbation (for data_dict and hybrid modes)
        if mode != "source_code":
            if ctype == "capacity":
                test_data = set_param(data, param, 0.001)
            elif ctype == "demand":
                test_data = perturb_param(data, param, 100.0)
            else:
                test_data = perturb_param(data, param, 0.01)
            result = self.executor.execute(exec_code, test_data)
            new_status = result.get("status")
            new_obj = result.get("objective")

        # Strategy 2: source-code fallback
        use_code_fallback = False
        if mode == "source_code":
            use_code_fallback = True
        elif mode == "hybrid" and new_obj is not None and new_status != "INFEASIBLE":
            if abs(baseline_obj) > self.epsilon:
                pre_change = abs(new_obj - baseline_obj) / abs(baseline_obj)
            else:
                pre_change = abs(new_obj - baseline_obj)
            if pre_change < 0.01:
                use_code_fallback = True

        if use_code_fallback:
            l2_code_params = extract_perturbable_params(code)
            matched = _match_param(l2_code_params, param)
            if matched:
                perturbed_code = perturb_code(code, matched['access_path'], code_factor)
                if perturbed_code != code:
                    result = self.executor.execute(perturbed_code, data)
                    new_status = result.get("status")
                    new_obj = result.get("objective")

        # If became infeasible, constraint is present
        if new_status == "INFEASIBLE":
            if verbose:
                print(f"    [PRESENT] {description} - perturbation caused infeasibility")
            return LayerResult(
                "L2", "cpt_present", Severity.PASS,
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

        # Threshold-based grading
        if change_ratio < 0.05:
            # < 5%: WARNING (constraint likely missing)
            if verbose:
                print(f"    [MISSING] {description} - only {change_ratio:.1%} change")
            return LayerResult(
                "L2", "cpt_missing", Severity.WARNING,
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
                "L2", "cpt_uncertain", Severity.INFO,
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
                "L2", "cpt_present", Severity.PASS,
                f"Constraint '{description}': {change_ratio:.1%} change - constraint is active",
                0.85,
                {"trigger_repair": False}
            )

    def _cpt_extract_candidates(self, problem_description: str, data: Dict) -> List[Dict]:
        """LLM-based constraint candidate extraction."""
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

        except json.JSONDecodeError:
            pass
        except Exception:
            pass

        return []

    # ----- L2-OPT: Objective Presence Testing -----

    def _layer2_opt(
        self, code: str, data: Dict, baseline_obj: float,
        problem_description: str, verbose: bool, mode: str = "data_dict"
    ) -> List[LayerResult]:
        """
        L2-OPT: Objective Presence Testing.

        Tests whether expected objective function terms (cost/revenue components)
        are actually present in the generated code by perturbing related parameters.

        Severity grading based on change ratio:
        - < 5%: WARNING (objective term likely missing)
        - 5%-30%: INFO (uncertain, may be normal)
        - > 30%: PASS (objective term present)
        """
        results = []

        try:
            if not self.llm_client:
                results.append(LayerResult(
                    "L2", "opt_skipped", Severity.INFO,
                    "L2 OPT skipped (requires LLM for objective term extraction)", 0.5,
                    {"trigger_repair": False, "is_likely_normal": True}
                ))
                return results

            candidates = self._opt_extract_candidates(problem_description, data)

            if not candidates:
                results.append(LayerResult(
                    "L2", "opt_extraction", Severity.INFO,
                    "No candidate objective terms extracted", 0.5,
                    {"trigger_repair": False, "is_likely_normal": True}
                ))
                return results

            if verbose:
                print(f"    Extracted {len(candidates)} objective term candidates")

            for candidate in candidates[:10]:
                try:
                    test_result = self._opt_test_candidate(
                        code, data, baseline_obj, candidate, verbose, mode
                    )
                    if test_result:
                        results.append(test_result)
                except Exception:
                    pass

        except Exception as e:
            results.append(LayerResult(
                "L2", "opt_error", Severity.INFO,
                f"OPT skipped: {str(e)[:100]}", 0.5,
                {"trigger_repair": False, "is_likely_normal": True}
            ))

        return results

    def _opt_test_candidate(
        self, code: str, data: Dict, baseline_obj: float,
        candidate: Dict, verbose: bool, mode: str = "data_dict"
    ) -> Optional[LayerResult]:
        """Test a single candidate objective term."""
        params = candidate.get("parameters", [])
        if not params:
            return None

        param = params[0]
        role = candidate.get("role", "other")
        description = candidate.get("description", "")

        # Determine perturbation factor based on objective term role
        if role == "cost":
            code_factor = 0.001
            perturbation_desc = "set to near-zero (x0.001)"
        elif role == "revenue":
            code_factor = 100.0
            perturbation_desc = "scaled up 100x"
        else:
            code_factor = 0.01
            perturbation_desc = "scaled to 1%"

        new_status = None
        new_obj = None

        # Strip json.loads data override so perturbed data dict is used
        exec_code = strip_data_override(code) if has_data_override(code) else code

        # Strategy 1: data-dict perturbation
        if mode != "source_code":
            if role == "cost":
                test_data = perturb_param(data, param, 0.001)
            elif role == "revenue":
                test_data = perturb_param(data, param, 100.0)
            else:
                test_data = perturb_param(data, param, 0.01)
            result = self.executor.execute(exec_code, test_data)
            new_status = result.get("status")
            new_obj = result.get("objective")

        # Strategy 2: source-code fallback
        use_code_fallback = False
        if mode == "source_code":
            use_code_fallback = True
        elif mode == "hybrid" and new_obj is not None and new_status != "INFEASIBLE":
            if abs(baseline_obj) > self.epsilon:
                pre_change = abs(new_obj - baseline_obj) / abs(baseline_obj)
            else:
                pre_change = abs(new_obj - baseline_obj)
            if pre_change < 0.01:
                use_code_fallback = True

        if use_code_fallback:
            opt_code_params = extract_perturbable_params(code)
            matched = _match_param(opt_code_params, param)
            if matched:
                perturbed_code = perturb_code(code, matched['access_path'], code_factor)
                if perturbed_code != code:
                    result = self.executor.execute(perturbed_code, data)
                    new_status = result.get("status")
                    new_obj = result.get("objective")

        if new_obj is None:
            return None

        # Calculate change ratio
        if abs(baseline_obj) > self.epsilon:
            change_ratio = abs(new_obj - baseline_obj) / abs(baseline_obj)
        else:
            change_ratio = abs(new_obj - baseline_obj)

        # Threshold-based grading
        if change_ratio < 0.05:
            # < 5%: WARNING (objective term likely missing)
            if verbose:
                print(f"    [MISSING] {description} - only {change_ratio:.1%} change")
            return LayerResult(
                "L2", "opt_missing", Severity.WARNING,
                f"Objective term '{description}' is likely MISSING: "
                f"extreme perturbation ({perturbation_desc}) caused only {change_ratio:.1%} change",
                0.75,
                {
                    "term_name": description,
                    "term_role": role,
                    "related_param": param,
                    "change_ratio": change_ratio,
                    "trigger_repair": True,
                    "is_likely_normal": False,
                    "repair_hint": f"Add objective term: {description}"
                }
            )

        elif change_ratio < 0.30:
            # 5%-30%: INFO (uncertain)
            if verbose:
                print(f"    [UNCERTAIN] {description} - {change_ratio:.1%} change")
            return LayerResult(
                "L2", "opt_uncertain", Severity.INFO,
                f"Objective term '{description}': {change_ratio:.1%} change (uncertain, may be normal)",
                0.5,
                {
                    "term_name": description,
                    "change_ratio": change_ratio,
                    "trigger_repair": False,
                    "is_likely_normal": True,
                    "note": "Moderate change - term may have small contribution"
                }
            )

        else:
            # > 30%: PASS (objective term present)
            if verbose:
                print(f"    [PRESENT] {description} - {change_ratio:.1%} change")
            return LayerResult(
                "L2", "opt_present", Severity.PASS,
                f"Objective term '{description}': {change_ratio:.1%} change - term is active",
                0.85,
                {"trigger_repair": False}
            )

    def _opt_extract_candidates(self, problem_description: str, data: Dict) -> List[Dict]:
        """LLM-based objective function term extraction."""
        if not self.llm_client:
            return []

        prompt = f"""Analyze this optimization problem and extract the KEY OBJECTIVE FUNCTION TERMS (cost and revenue components) that should be present in the model's objective function.

## Problem Description
{problem_description}

## Available Data Parameters
{list(data.keys())}

## Task
Identify cost and revenue terms that MUST appear in the objective function. Focus on:
1. **Cost terms**: purchasing/procurement cost, holding/storage cost, transportation cost, shortage/backorder cost, setup/fixed cost, penalty cost
2. **Revenue terms**: sales revenue, demand revenue, return/salvage value

For each term, identify which data parameter(s) provide its coefficient.

## Output Format
Return ONLY a JSON array with this exact format:
```json
[
  {{"description": "unit purchasing cost", "role": "cost", "parameters": ["unit_cost"]}},
  {{"description": "sales revenue per unit", "role": "revenue", "parameters": ["selling_price"]}}
]
```

Return ONLY the JSON array, no explanation."""

        try:
            response = self.llm_client.generate(prompt)

            # Try to find JSON array
            match = re.search(r'\[[\s\S]*\]', response)
            if match:
                json_str = match.group()
                candidates = json.loads(json_str)
                valid = [c for c in candidates
                        if isinstance(c, dict) and "description" in c and "parameters" in c]
                return valid

            try:
                candidates = json.loads(response.strip())
                if isinstance(candidates, list):
                    return [c for c in candidates
                            if isinstance(c, dict) and "description" in c]
            except json.JSONDecodeError:
                pass

        except json.JSONDecodeError:
            pass
        except Exception:
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
# Diagnostic Conversion
# =============================================================================

def layer_results_to_diagnostics(
    layer_results: List[LayerResult],
    baseline_obj: Optional[float] = None,
    delta: float = 0.1,
) -> List[Diagnostic]:
    """
    Convert LayerResult list to unified Diagnostic list.

    Handles L1 (execution + duality) and L2 (CPT + OPT) results.
    """
    diagnostics: List[Diagnostic] = []

    for r in layer_results:
        if r.severity == Severity.PASS:
            continue

        d = r.details or {}

        # --- L1 ---
        if r.layer == "L1":
            if r.check == "syntax":
                diagnostics.append(Diagnostic(
                    layer="L1",
                    issue_type="SYNTAX_ERROR",
                    severity="ERROR",
                    target_name="code",
                    evidence=r.message,
                    triggers_repair=True,
                ))
            elif r.check == "runtime":
                diagnostics.append(Diagnostic(
                    layer="L1",
                    issue_type="RUNTIME_ERROR",
                    severity="ERROR",
                    target_name="code",
                    evidence=r.message,
                    triggers_repair=True,
                ))
            elif r.check == "output":
                diagnostics.append(Diagnostic(
                    layer="L1",
                    issue_type="NO_OUTPUT",
                    severity="ERROR",
                    target_name="code",
                    evidence=r.message,
                    triggers_repair=True,
                ))
            elif r.check == "feasibility":
                iis_constrs = d.get("iis_constraints") or []
                target = ", ".join(
                    c.split(" (")[0] for c in iis_constrs[:5]
                ) if iis_constrs else "model"
                diagnostics.append(Diagnostic(
                    layer="L1",
                    issue_type="INFEASIBLE",
                    severity="ERROR",
                    target_name=target,
                    evidence=r.message,
                    triggers_repair=True,
                ))
            elif r.check == "boundedness":
                ub_vars = d.get("unbounded_vars") or []
                target = ", ".join(
                    v.split(" (")[0] for v in ub_vars[:5]
                ) if ub_vars else "objective"
                diagnostics.append(Diagnostic(
                    layer="L1",
                    issue_type="UNBOUNDED",
                    severity="ERROR",
                    target_name=target,
                    evidence=r.message,
                    triggers_repair=True,
                ))
            elif r.check == "timeout":
                diagnostics.append(Diagnostic(
                    layer="L1",
                    issue_type="TIMEOUT",
                    severity="ERROR",
                    target_name="solver",
                    evidence=r.message,
                    triggers_repair=True,
                ))
            elif r.check == "duality_gap" and r.severity == Severity.INFO:
                primal = d.get("primal", baseline_obj or 0)
                dual = d.get("dual", 0)
                gap = d.get("gap", 0)
                diagnostics.append(Diagnostic(
                    layer="L1",
                    issue_type="DUALITY_GAP",
                    severity="INFO",
                    target_name="primal_dual",
                    evidence=(
                        f"Primal-dual gap detected: "
                        f"primal={primal:.6f}, dual={dual:.6f}, "
                        f"relative gap={gap:.2%} (threshold=1%). "
                        f"This is typically a numerical artifact rather than a modeling error."
                    ),
                    triggers_repair=False,
                ))
            elif r.severity == Severity.INFO:
                diagnostics.append(Diagnostic(
                    layer="L1",
                    issue_type="DUALITY_GAP",
                    severity="INFO",
                    target_name="primal_dual",
                    evidence=r.message,
                    triggers_repair=False,
                ))

        # --- L2 (CPT + OPT) ---
        elif r.layer == "L2":
            # CPT results
            if r.check == "cpt_missing" and r.severity == Severity.WARNING:
                desc = d.get("constraint_name", "unknown")
                param = d.get("related_param", "unknown")
                ratio = d.get("change_ratio", 0)
                diagnostics.append(Diagnostic(
                    layer="L2",
                    issue_type="MISSING_CONSTRAINT",
                    severity="WARNING",
                    target_name=desc,
                    evidence=(
                        f"Constraint related to '{desc}' is likely MISSING. "
                        f"Extreme perturbation ({param} -> extreme value) caused only "
                        f"{ratio:.1%} objective change (threshold: <5% = missing). "
                        f"A correctly constrained model should show significant objective "
                        f"change under this perturbation."
                    ),
                    triggers_repair=True,
                ))
            elif r.check == "cpt_uncertain" and r.severity == Severity.INFO:
                desc = d.get("constraint_name", "unknown")
                ratio = d.get("change_ratio", 0)
                diagnostics.append(Diagnostic(
                    layer="L2",
                    issue_type="MISSING_CONSTRAINT",
                    severity="INFO",
                    target_name=desc,
                    evidence=(
                        f"Constraint related to '{desc}' shows UNCERTAIN response. "
                        f"Extreme perturbation caused {ratio:.1%} objective change "
                        f"(threshold: 5-30% = uncertain)."
                    ),
                    triggers_repair=False,
                ))

            # OPT results
            elif r.check == "opt_missing" and r.severity == Severity.WARNING:
                desc = d.get("term_name", "unknown")
                param = d.get("related_param", "unknown")
                ratio = d.get("change_ratio", 0)
                diagnostics.append(Diagnostic(
                    layer="L2",
                    issue_type="MISSING_OBJECTIVE_TERM",
                    severity="WARNING",
                    target_name=desc,
                    evidence=(
                        f"Objective term '{desc}' is likely MISSING. "
                        f"Extreme perturbation ({param} -> extreme value) caused only "
                        f"{ratio:.1%} objective change (threshold: <5% = missing). "
                        f"If this cost/revenue term is in the problem description, "
                        f"it should affect the objective when perturbed."
                    ),
                    triggers_repair=True,
                ))
            elif r.check == "opt_uncertain" and r.severity == Severity.INFO:
                desc = d.get("term_name", "unknown")
                ratio = d.get("change_ratio", 0)
                diagnostics.append(Diagnostic(
                    layer="L2",
                    issue_type="MISSING_OBJECTIVE_TERM",
                    severity="INFO",
                    target_name=desc,
                    evidence=(
                        f"Objective term '{desc}' shows UNCERTAIN response. "
                        f"Extreme perturbation caused {ratio:.1%} objective change "
                        f"(threshold: 5-30% = uncertain)."
                    ),
                    triggers_repair=False,
                ))

            # Generic L2 INFO (skipped, error, etc.)
            elif r.severity == Severity.INFO:
                diagnostics.append(Diagnostic(
                    layer="L2",
                    issue_type=r.check.upper(),
                    severity="INFO",
                    target_name="behavioral_test",
                    evidence=r.message,
                    triggers_repair=False,
                ))

    return diagnostics


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
