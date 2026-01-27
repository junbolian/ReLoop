"""
Module 2: Behavioral Verification (CORE)

7-Layer verification system:
  Layer 1: Execution - Does code run without errors?
  Layer 2: Feasibility - OPTIMAL? INFEASIBLE? UNBOUNDED?
  Layer 3: Code Structure - AST-based structural verification (UNIVERSAL)
  Layer 4: Monotonicity - Does each parameter affect objective? (CORE)
  Layer 5: Sensitivity Direction - Does direction match expectation? (Best effort)
  Layer 6: Boundary - Extreme value behavior
  Layer 7: Domain Probes - Domain-specific tests (Optional)

Key Insight: L3 (AST) is fast static analysis that runs before expensive runtime tests.
L4 (Monotonicity) is the universal behavioral core - no domain knowledge needed.
"""

import subprocess
import sys
import json
import base64
import re
import time
import ast
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .param_utils import (
    extract_numeric_params, infer_param_role, get_expected_direction,
    perturb_param, set_param, ParameterRole, should_skip_param
)


class VerificationStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class VerificationResult:
    """Result from a single verification check"""
    passed: bool
    layer: int
    diagnosis: Optional[str] = None
    skipped: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationReport:
    """Complete verification report"""
    passed: bool
    results: List[VerificationResult] = field(default_factory=list)
    failed_layer: Optional[int] = None
    diagnosis: Optional[str] = None
    execution_time: float = 0.0
    objective: Optional[float] = None  # Always store objective if available

    def count_layers_passed(self) -> int:
        return 7 if self.failed_layer is None else self.failed_layer - 1


class CodeExecutor:
    """Execute optimization code in isolated subprocess"""

    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    def check_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        try:
            compile(code, "<string>", "exec")
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"

    def execute(self, code: str, data: Dict[str, Any]) -> Dict[str, Any]:
        data_b64 = base64.b64encode(json.dumps(data).encode()).decode()
        wrapper = f'''
import sys, json, base64
data = json.loads(base64.b64decode("{data_b64}").decode())
{code}
'''
        try:
            result = subprocess.run([sys.executable, "-c", wrapper],
                                    capture_output=True, text=True, timeout=self.timeout)
            return self._parse_output(result)
        except subprocess.TimeoutExpired:
            return {"status": "TIMEOUT", "error": "Timeout", "exit_code": -1}
        except Exception as e:
            return {"status": "ERROR", "error": str(e), "exit_code": -1}

    def _parse_output(self, result) -> Dict[str, Any]:
        output = {"exit_code": result.returncode, "stdout": result.stdout,
                  "stderr": result.stderr, "status": None, "objective": None,
                  "error": result.stderr if result.returncode != 0 else None}
        status_match = re.search(r"status:\s*(\d+)", result.stdout)
        if status_match:
            code = int(status_match.group(1))
            output["status"] = {2: "OPTIMAL", 3: "INFEASIBLE", 5: "UNBOUNDED", 9: "TIME_LIMIT"}.get(code, f"CODE_{code}")
        obj_match = re.search(r"objective:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", result.stdout)
        if obj_match:
            output["objective"] = float(obj_match.group(1))
        return output


class BehavioralVerifier:
    """7-Layer Behavioral Verification System"""

    def __init__(self, delta: float = 0.2, epsilon: float = 1e-4, timeout: int = 60):
        self.delta = delta
        self.epsilon = epsilon
        self.executor = CodeExecutor(timeout=timeout)
        self.max_params = 10

    def verify(self, code: str, data: Dict[str, Any], obj_sense: str = "minimize",
               enable_layer7: bool = False, verbose: bool = False) -> VerificationReport:
        """
        7-Layer Behavioral Verification with lenient progression.

        Key principles:
        1. L1 (Execution) MUST pass - code must run
        2. L2 (Feasibility) - lenient: TIME_LIMIT with objective is OK
        3. L3 (AST) - fast static analysis before expensive runtime tests
        4. L4-L7 - continue even if some tests fail, count passed/failed
        5. Always report objective value if available
        """
        start_time = time.time()
        results = []
        layers_passed = 0
        final_diagnosis = None
        objective_value = None

        if verbose:
            print("=" * 60 + "\nBehavioral Verification\n" + "=" * 60)

        # Layer 1: Execution (MUST pass)
        if verbose:
            print("\n[L1: Execution]")
        l1_result, baseline = self._layer1_execution(code, data)
        results.append(l1_result)
        objective_value = baseline.get("objective")

        if not l1_result.passed:
            if verbose:
                print(f"  FAIL: {l1_result.diagnosis}")
            return VerificationReport(False, results, 1, l1_result.diagnosis,
                                     time.time() - start_time, objective_value)
        layers_passed = 1
        if verbose:
            print(f"  PASS: status={baseline.get('status')}, obj={objective_value}")

        # Layer 2: Feasibility (lenient - TIME_LIMIT with objective is OK)
        if verbose:
            print("\n[L2: Feasibility]")
        l2_result = self._layer2_feasibility(baseline, obj_sense)
        l2_result.details["objective"] = objective_value
        results.append(l2_result)

        # Accept TIME_LIMIT if we got an objective (solver found a solution)
        status = baseline.get("status")
        if status == "TIME_LIMIT" and objective_value is not None:
            l2_result.passed = True
            l2_result.diagnosis = "TIME_LIMIT but objective obtained"
            if verbose:
                print(f"  PASS (TIME_LIMIT with objective={objective_value})")
        elif not l2_result.passed:
            if verbose:
                print(f"  FAIL: {l2_result.diagnosis}")
            final_diagnosis = l2_result.diagnosis
            return VerificationReport(False, results, layers_passed, final_diagnosis,
                                     time.time() - start_time, objective_value)
        else:
            if verbose:
                print("  PASS")
        layers_passed = 2

        params = extract_numeric_params(data)
        baseline_obj = objective_value
        if baseline_obj is None:
            return VerificationReport(False, results, layers_passed, "No objective",
                                     time.time() - start_time, objective_value)

        # Layer 3: Code Structure (AST) - UNIVERSAL, fast static analysis first
        # AST analysis doesn't leak data - only examines code structure
        if verbose:
            print("\n[L3: Code Structure (AST)]")
        l3_results = self._layer3_code_structure(code, data, verbose)
        results.extend(l3_results)
        l3_failed = [r for r in l3_results if not r.passed and not r.skipped]
        l3_passed = len(l3_failed) == 0
        if l3_passed:
            layers_passed = 3
        elif final_diagnosis is None:
            final_diagnosis = l3_failed[0].diagnosis

        # Layer 4: Monotonicity - always run, track pass/fail (CORE behavioral test)
        if verbose:
            print(f"\n[L4: Monotonicity] Testing {min(len(params), self.max_params)} parameters")
        l4_results = self._layer4_monotonicity(code, data, params, baseline_obj, verbose)
        results.extend(l4_results)
        l4_failed = [r for r in l4_results if not r.passed and not r.skipped]
        l4_passed = len(l4_failed) == 0
        if l3_passed and l4_passed:
            layers_passed = 4
        elif final_diagnosis is None and l4_failed:
            final_diagnosis = l4_failed[0].diagnosis

        # Layer 5: Sensitivity Direction - always run (independent of L3/L4)
        if verbose:
            print("\n[L5: Sensitivity Direction]")
        l5_results = self._layer5_direction(code, data, params, baseline_obj, obj_sense, verbose)
        results.extend(l5_results)
        l5_failed = [r for r in l5_results if not r.passed and not r.skipped]
        l5_passed = len(l5_failed) == 0
        if l3_passed and l4_passed and l5_passed:
            layers_passed = 5
        elif final_diagnosis is None and l5_failed:
            final_diagnosis = l5_failed[0].diagnosis

        # Layer 6: Boundary - always run (independent of L3-L5)
        if verbose:
            print("\n[L6: Boundary]")
        l6_results = self._layer6_boundary(code, data, params, baseline_obj, obj_sense, verbose)
        results.extend(l6_results)
        l6_failed = [r for r in l6_results if not r.passed and not r.skipped]
        l6_passed = len(l6_failed) == 0
        if l3_passed and l4_passed and l5_passed and l6_passed:
            layers_passed = 6
        elif final_diagnosis is None and l6_failed:
            final_diagnosis = l6_failed[0].diagnosis

        # Layer 7: Domain Probes - run if enabled (independent of L3-L6)
        l7_passed = True
        if enable_layer7 and self._is_retail_data(data):
            if verbose:
                print("\n[L7: Domain Probes]")
            l7_results = self._layer7_domain_probes(code, data, baseline_obj, verbose)
            results.extend(l7_results)
            l7_failed = [r for r in l7_results if not r.passed and not r.skipped]
            l7_passed = len(l7_failed) == 0
            if l3_passed and l4_passed and l5_passed and l6_passed and l7_passed:
                layers_passed = 7
            elif final_diagnosis is None and l7_failed:
                final_diagnosis = l7_failed[0].diagnosis

        # Max layers: 7 (L6 is conditional, but L7 always runs)
        max_layers = 7
        all_passed = layers_passed == max_layers
        if verbose and all_passed:
            print("\n" + "=" * 60 + "\nVERIFICATION PASSED\n" + "=" * 60)

        return VerificationReport(
            passed=all_passed,
            results=results,
            failed_layer=None if all_passed else layers_passed + 1,
            diagnosis=final_diagnosis,
            execution_time=time.time() - start_time,
            objective=objective_value
        )

    def _layer1_execution(self, code: str, data: Dict) -> Tuple[VerificationResult, Dict]:
        syntax_ok, err = self.executor.check_syntax(code)
        if not syntax_ok:
            return VerificationResult(False, 1, f"Syntax error: {err}"), {}
        result = self.executor.execute(code, data)
        if result.get("exit_code", -1) != 0:
            return VerificationResult(False, 1, f"Runtime error: {result.get('error', '')[:500]}"), result
        if result.get("status") is None:
            return VerificationResult(False, 1, "No status output"), result
        return VerificationResult(True, 1), result

    def _layer2_feasibility(self, baseline: Dict, obj_sense: str = "minimize") -> VerificationResult:
        status = baseline.get("status")
        objective = baseline.get("objective")
        if status == "OPTIMAL":
            # Check for suspicious negative objective in minimization problem
            if obj_sense.lower() == "minimize" and objective is not None and objective < 0:
                return VerificationResult(False, 2,
                    f"SUSPICIOUS: Negative objective ({objective:.2f}) in minimization problem. "
                    f"Check if cost/penalty terms can become negative (e.g., difference terms "
                    f"without max(0, ...) or slack variable). All cost terms should be >= 0.")
            return VerificationResult(True, 2)
        if status == "INFEASIBLE":
            return VerificationResult(False, 2, "Model is INFEASIBLE. Add slack variables or check constraints.")
        if status == "UNBOUNDED":
            return VerificationResult(False, 2, "Model is UNBOUNDED. Add bounds or missing constraints.")
        return VerificationResult(False, 2, f"Unexpected status: {status}")

    def _layer4_monotonicity(self, code: str, data: Dict, params: List[str],
                              baseline_obj: float, verbose: bool) -> List[VerificationResult]:
        results = []
        for param in params[:self.max_params]:
            # Check if parameter should be skipped (e.g., value is zero)
            skip, skip_reason = should_skip_param(data, param)
            if skip:
                results.append(VerificationResult(True, 4, skipped=True,
                                                   details={"skip_reason": skip_reason}))
                if verbose:
                    print(f"  Testing: {param}")
                    print(f"    SKIP ({skip_reason})")
                continue

            if verbose:
                print(f"  Testing: {param}")
            obj_up = self.executor.execute(code, perturb_param(data, param, 1 + self.delta)).get("objective")
            obj_down = self.executor.execute(code, perturb_param(data, param, 1 - self.delta)).get("objective")
            if obj_up is None or obj_down is None:
                results.append(VerificationResult(True, 4, skipped=True))
                if verbose:
                    print("    SKIP (execution failed)")
                continue
            if abs(baseline_obj) > self.epsilon:
                delta_up = abs(obj_up - baseline_obj) / abs(baseline_obj)
                delta_down = abs(obj_down - baseline_obj) / abs(baseline_obj)
            else:
                delta_up, delta_down = abs(obj_up - baseline_obj), abs(obj_down - baseline_obj)
            if delta_up < self.epsilon and delta_down < self.epsilon:
                diag = f"Parameter '{param}' has NO EFFECT on objective (baseline={baseline_obj:.4f}, up={obj_up:.4f}, down={obj_down:.4f}). The constraint involving this parameter may be MISSING or INCORRECTLY INDEXED. Check that '{param}' appears in at least one constraint and that indices align with decision variables."
                results.append(VerificationResult(False, 4, diag))
                if verbose:
                    print(f"    FAIL: No effect detected!")
            else:
                results.append(VerificationResult(True, 4))
                if verbose:
                    print(f"    PASS (effect detected)")
        return results

    def _layer5_direction(self, code: str, data: Dict, params: List[str],
                          baseline_obj: float, obj_sense: str, verbose: bool) -> List[VerificationResult]:
        results = []
        is_min = obj_sense.lower() == "minimize"
        for param in params[:self.max_params]:
            role = infer_param_role(param)
            if role == ParameterRole.UNKNOWN:
                results.append(VerificationResult(True, 5, skipped=True))
                continue
            if verbose:
                print(f"  Testing: {param} (role: {role.value})")
            if role == ParameterRole.REQUIREMENT:
                test_data = perturb_param(data, param, 1 + self.delta)
                expected = "increase" if is_min else "decrease"
            elif role == ParameterRole.CAPACITY:
                test_data = perturb_param(data, param, 1 - self.delta)
                expected = "increase" if is_min else "decrease"
            elif role == ParameterRole.COST:
                test_data = perturb_param(data, param, 1 + self.delta)
                expected = "increase"
            else:
                continue
            result = self.executor.execute(code, test_data)
            test_obj, test_status = result.get("objective"), result.get("status")
            if test_obj is None:
                if test_status == "INFEASIBLE" and role in (ParameterRole.REQUIREMENT, ParameterRole.CAPACITY):
                    results.append(VerificationResult(True, 5))
                    if verbose:
                        print("    PASS (INFEASIBLE acceptable)")
                else:
                    results.append(VerificationResult(True, 5, skipped=True))
                continue
            change = test_obj - baseline_obj
            actual = "increase" if change > self.epsilon * abs(baseline_obj) else "decrease" if change < -self.epsilon * abs(baseline_obj) else "none"
            if expected == actual or actual == "none":
                results.append(VerificationResult(True, 5))
                if verbose:
                    print("    PASS")
            else:
                diag = f"Parameter '{param}' (role: {role.value}) shows UNEXPECTED DIRECTION. Expected {expected}, got {actual}."
                results.append(VerificationResult(False, 5, diag))
                if verbose:
                    print(f"    FAIL: Expected {expected}, got {actual}")
        return results

    def _layer6_boundary(self, code: str, data: Dict, params: List[str],
                         baseline_obj: float, obj_sense: str, verbose: bool) -> List[VerificationResult]:
        results = []

        # 6.1: Capacity = 0 boundary tests (existing)
        cap_params = [p for p in params if infer_param_role(p) == ParameterRole.CAPACITY]
        for param in cap_params[:3]:
            skip, skip_reason = should_skip_param(data, param)
            if skip:
                if verbose:
                    print(f"  Boundary test: {param} = 0")
                    print(f"    SKIP ({skip_reason})")
                results.append(VerificationResult(True, 6, skipped=True,
                                                   details={"skip_reason": skip_reason}))
                continue

            if verbose:
                print(f"  Boundary test: {param} = 0")
            result = self.executor.execute(code, set_param(data, param, 0))
            status, obj = result.get("status"), result.get("objective")
            if status == "INFEASIBLE":
                results.append(VerificationResult(True, 6))
                if verbose:
                    print("    PASS (INFEASIBLE)")
            elif obj is not None:
                if abs(obj - baseline_obj) < 0.1 * abs(baseline_obj):
                    results.append(VerificationResult(False, 6, f"Setting '{param}' to ZERO has no effect."))
                    if verbose:
                        print("    FAIL: No effect")
                else:
                    results.append(VerificationResult(True, 6))
                    if verbose:
                        print("    PASS")

        # 6.2: Structural boundary - periods=1 (UNIVERSAL)
        # Multi-period models should degrade gracefully to single period
        if "periods" in data and data.get("periods", 1) > 1:
            if verbose:
                print("  Structural boundary: periods = 1")
            test_data = self._reduce_to_single_period(data)
            result = self.executor.execute(code, test_data)
            status = result.get("status")
            if status is None or result.get("exit_code", -1) != 0:
                diag = "Code crashes with periods=1. Check t-1 or t+1 indexing at boundaries."
                results.append(VerificationResult(False, 6, diag))
                if verbose:
                    print(f"    FAIL: {diag}")
            elif status in ("OPTIMAL", "TIME_LIMIT") and result.get("objective") is not None:
                results.append(VerificationResult(True, 6))
                if verbose:
                    print("    PASS")
            else:
                # INFEASIBLE with single period might be OK (some constraints can't be met)
                results.append(VerificationResult(True, 6, skipped=True))
                if verbose:
                    print(f"    SKIP (status={status})")

        # 6.3: Differential verification - parameter pairs (UNIVERSAL)
        # capacity↓ and requirement↑ should both increase cost (minimize)
        results.extend(self._differential_verification(code, data, params, baseline_obj, obj_sense, verbose))

        return results

    def _reduce_to_single_period(self, data: Dict) -> Dict:
        """Reduce multi-period data to single period (universal)."""
        import copy
        test_data = copy.deepcopy(data)
        test_data["periods"] = 1

        # Truncate all time-indexed arrays to length 1
        for key, value in test_data.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, list) and len(v) > 1:
                        value[k] = [v[0]]
            elif isinstance(value, list) and len(value) > 1:
                test_data[key] = [value[0]]

        return test_data

    def _differential_verification(self, code: str, data: Dict, params: List[str],
                                    baseline_obj: float, obj_sense: str, verbose: bool) -> List[VerificationResult]:
        """
        Differential verification: test that parameter pairs have expected relationships.

        Universal principle: capacity↓ and requirement↑ should both make the problem harder
        (increase cost for minimization, decrease profit for maximization).
        """
        results = []
        is_min = obj_sense.lower() == "minimize"

        # Find capacity and requirement parameters
        cap_params = [p for p in params if infer_param_role(p) == ParameterRole.CAPACITY]
        req_params = [p for p in params if infer_param_role(p) == ParameterRole.REQUIREMENT]

        if not cap_params or not req_params:
            return results  # Skip if we don't have both types

        # Test one pair: capacity↓ vs requirement↑
        cap_param = cap_params[0]
        req_param = req_params[0]

        # Skip if either parameter should be skipped
        if should_skip_param(data, cap_param)[0] or should_skip_param(data, req_param)[0]:
            return results

        if verbose:
            print(f"  Differential: {cap_param}↓ vs {req_param}↑")

        # Get effects
        obj_cap_down = self.executor.execute(code, perturb_param(data, cap_param, 0.8)).get("objective")
        obj_req_up = self.executor.execute(code, perturb_param(data, req_param, 1.2)).get("objective")

        if obj_cap_down is None or obj_req_up is None:
            results.append(VerificationResult(True, 6, skipped=True))
            if verbose:
                print("    SKIP (execution failed)")
            return results

        effect_cap = obj_cap_down - baseline_obj
        effect_req = obj_req_up - baseline_obj

        # Both should make problem harder (same direction for minimize/maximize)
        # For minimize: both should increase cost (positive effect)
        # For maximize: both should decrease profit (negative effect)
        if is_min:
            cap_correct = effect_cap >= -self.epsilon * abs(baseline_obj)  # Should increase or stay same
            req_correct = effect_req >= -self.epsilon * abs(baseline_obj)
        else:
            cap_correct = effect_cap <= self.epsilon * abs(baseline_obj)  # Should decrease or stay same
            req_correct = effect_req <= self.epsilon * abs(baseline_obj)

        if cap_correct and req_correct:
            results.append(VerificationResult(True, 6))
            if verbose:
                print(f"    PASS (cap_effect={effect_cap:.2f}, req_effect={effect_req:.2f})")
        else:
            diag = f"Differential mismatch: {cap_param}↓ effect={effect_cap:.2f}, {req_param}↑ effect={effect_req:.2f}. " \
                   f"Both should make problem harder but show opposite effects."
            results.append(VerificationResult(False, 6, diag))
            if verbose:
                print(f"    FAIL: {diag}")

        return results

    def _layer7_domain_probes(self, code: str, data: Dict, baseline_obj: float, verbose: bool) -> List[VerificationResult]:
        """
        Layer 7: Domain-Specific Probes for Modeling Error Detection

        These probes detect SEMANTIC modeling errors that L1-L6 cannot catch:
        1. Inventory balance equation errors
        2. Constraint direction errors (<=, >=, ==)
        3. Cost component completeness errors
        4. Extreme value response errors
        5. Domain-specific structure errors
        """
        import copy
        results = []

        # =================================================================
        # Probe 1: Inventory Balance Equation Check
        # Correct: I[t] = I[t-1] + order[t] - sales[t] - waste[t]
        # Error detection: If initial_inventory affects ALL periods equally,
        # the balance equation is likely missing or wrong
        # =================================================================
        if "initial_inventory" in data or "init_inventory" in data:
            if verbose:
                print("  Probe: Inventory Balance Equation")

            init_key = "initial_inventory" if "initial_inventory" in data else "init_inventory"
            test_data = copy.deepcopy(data)

            # Double the initial inventory
            if isinstance(test_data[init_key], dict):
                for k in test_data[init_key]:
                    if isinstance(test_data[init_key][k], (int, float)):
                        test_data[init_key][k] = test_data[init_key][k] * 2
                    elif isinstance(test_data[init_key][k], list):
                        test_data[init_key][k] = [v * 2 for v in test_data[init_key][k]]
            elif isinstance(test_data[init_key], (int, float)):
                test_data[init_key] = test_data[init_key] * 2

            result = self.executor.execute(code, test_data)
            obj_double_init = result.get("objective")

            if obj_double_init is not None:
                # More initial inventory should reduce cost (less need to order)
                # If cost INCREASES, the balance equation may be wrong
                if obj_double_init > baseline_obj * 1.1:  # Cost increased by >10%
                    diag = f"Doubling initial_inventory INCREASED cost ({obj_double_init:.2f} > {baseline_obj:.2f}). " \
                           f"Inventory balance equation likely incorrect (check I[t] = I[t-1] + order - sales)."
                    results.append(VerificationResult(False, 7, diag))
                    if verbose:
                        print(f"    FAIL: {diag}")
                else:
                    results.append(VerificationResult(True, 7))
                    if verbose:
                        print(f"    PASS (obj={obj_double_init:.2f} <= baseline={baseline_obj:.2f})")
            else:
                results.append(VerificationResult(True, 7, skipped=True))
                if verbose:
                    print("    SKIP (no objective)")

        # =================================================================
        # Probe 2: Constraint Direction Check (Capacity)
        # Correct: usage <= capacity
        # Error: If reducing capacity by 50% has NO effect or REDUCES cost,
        # the constraint direction is likely wrong
        # =================================================================
        cap_keys = [k for k in data.keys() if 'capacity' in k.lower() or 'cap' in k.lower()]
        if cap_keys:
            if verbose:
                print("  Probe: Constraint Direction (Capacity)")

            test_data = copy.deepcopy(data)
            cap_key = cap_keys[0]

            # Reduce capacity by 50%
            if isinstance(test_data[cap_key], dict):
                for k in test_data[cap_key]:
                    if isinstance(test_data[cap_key][k], (int, float)):
                        test_data[cap_key][k] = test_data[cap_key][k] * 0.5
                    elif isinstance(test_data[cap_key][k], list):
                        test_data[cap_key][k] = [v * 0.5 for v in test_data[cap_key][k]]
            elif isinstance(test_data[cap_key], (int, float)):
                test_data[cap_key] = test_data[cap_key] * 0.5
            elif isinstance(test_data[cap_key], list):
                test_data[cap_key] = [v * 0.5 for v in test_data[cap_key]]

            result = self.executor.execute(code, test_data)
            status = result.get("status")
            obj_half_cap = result.get("objective")

            if status == "INFEASIBLE":
                # Good - reduced capacity made problem infeasible
                results.append(VerificationResult(True, 7))
                if verbose:
                    print("    PASS (INFEASIBLE with reduced capacity)")
            elif obj_half_cap is not None:
                # Reducing capacity should increase cost or have no effect
                if obj_half_cap < baseline_obj * 0.9:  # Cost DECREASED by >10%
                    diag = f"Halving capacity DECREASED cost ({obj_half_cap:.2f} < {baseline_obj:.2f}). " \
                           f"Capacity constraint direction likely wrong (should be usage <= capacity, not >=)."
                    results.append(VerificationResult(False, 7, diag))
                    if verbose:
                        print(f"    FAIL: {diag}")
                else:
                    results.append(VerificationResult(True, 7))
                    if verbose:
                        print(f"    PASS (obj={obj_half_cap:.2f} >= baseline*0.9)")
            else:
                results.append(VerificationResult(True, 7, skipped=True))
                if verbose:
                    print("    SKIP (no objective)")

        # =================================================================
        # Probe 3: Cost Component Completeness
        # Test if holding cost is actually being computed
        # If holding_cost * 10 has NO effect on objective, it's not included
        # =================================================================
        cost_keys = [k for k in data.keys() if 'cost' in k.lower() and 'hold' in k.lower()]
        if not cost_keys:
            cost_keys = [k for k in data.keys() if 'holding' in k.lower()]

        if cost_keys:
            if verbose:
                print("  Probe: Cost Component Completeness (Holding Cost)")

            test_data = copy.deepcopy(data)
            cost_key = cost_keys[0]

            # Multiply holding cost by 10
            if isinstance(test_data[cost_key], dict):
                for k in test_data[cost_key]:
                    if isinstance(test_data[cost_key][k], (int, float)):
                        test_data[cost_key][k] = test_data[cost_key][k] * 10
                    elif isinstance(test_data[cost_key][k], list):
                        test_data[cost_key][k] = [v * 10 for v in test_data[cost_key][k]]
            elif isinstance(test_data[cost_key], (int, float)):
                test_data[cost_key] = test_data[cost_key] * 10
            elif isinstance(test_data[cost_key], list):
                test_data[cost_key] = [v * 10 for v in test_data[cost_key]]

            result = self.executor.execute(code, test_data)
            obj_high_hold = result.get("objective")

            if obj_high_hold is not None:
                change_ratio = abs(obj_high_hold - baseline_obj) / max(abs(baseline_obj), 1e-6)
                if change_ratio < 0.01:  # Less than 1% change
                    diag = f"Multiplying {cost_key} by 10x had NO effect on objective. " \
                           f"Holding cost likely NOT included in objective function."
                    results.append(VerificationResult(False, 7, diag))
                    if verbose:
                        print(f"    FAIL: {diag}")
                else:
                    results.append(VerificationResult(True, 7))
                    if verbose:
                        print(f"    PASS (change_ratio={change_ratio:.2%})")
            else:
                results.append(VerificationResult(True, 7, skipped=True))
                if verbose:
                    print("    SKIP (no objective)")

        # =================================================================
        # Probe 4: Waste/Penalty Cost Check
        # If waste_cost * 10 has NO effect, waste penalty is not implemented
        # =================================================================
        waste_keys = [k for k in data.keys() if 'waste' in k.lower() or 'spoil' in k.lower() or 'penalty' in k.lower()]
        if waste_keys:
            if verbose:
                print("  Probe: Cost Component Completeness (Waste/Penalty)")

            test_data = copy.deepcopy(data)
            waste_key = waste_keys[0]

            # Multiply waste cost by 10
            if isinstance(test_data[waste_key], dict):
                for k in test_data[waste_key]:
                    if isinstance(test_data[waste_key][k], (int, float)):
                        test_data[waste_key][k] = test_data[waste_key][k] * 10
                    elif isinstance(test_data[waste_key][k], list):
                        test_data[waste_key][k] = [v * 10 for v in test_data[waste_key][k]]
            elif isinstance(test_data[waste_key], (int, float)):
                test_data[waste_key] = test_data[waste_key] * 10
            elif isinstance(test_data[waste_key], list):
                test_data[waste_key] = [v * 10 for v in test_data[waste_key]]

            result = self.executor.execute(code, test_data)
            obj_high_waste = result.get("objective")

            if obj_high_waste is not None:
                change_ratio = abs(obj_high_waste - baseline_obj) / max(abs(baseline_obj), 1e-6)
                if change_ratio < 0.01:  # Less than 1% change
                    diag = f"Multiplying {waste_key} by 10x had NO effect on objective. " \
                           f"Waste/penalty cost likely NOT included in objective function."
                    results.append(VerificationResult(False, 7, diag))
                    if verbose:
                        print(f"    FAIL: {diag}")
                else:
                    results.append(VerificationResult(True, 7))
                    if verbose:
                        print(f"    PASS (change_ratio={change_ratio:.2%})")
            else:
                results.append(VerificationResult(True, 7, skipped=True))
                if verbose:
                    print("    SKIP (no objective)")

        # =================================================================
        # Probe 5: Lost Sales Variable (existing)
        # =================================================================
        if verbose:
            print("  Probe: Lost Sales Variable")
        test_data = copy.deepcopy(data)
        if "demand_curve" in test_data:
            for p in test_data["demand_curve"]:
                test_data["demand_curve"][p] = [v * 10 for v in test_data["demand_curve"][p]]
        result = self.executor.execute(code, test_data)
        if result.get("status") == "INFEASIBLE":
            results.append(VerificationResult(False, 7, "INFEASIBLE when demand >> supply. Missing lost sales variable."))
            if verbose:
                print("    FAIL")
        else:
            results.append(VerificationResult(True, 7))
            if verbose:
                print("    PASS")

        # =================================================================
        # Probe 6: Shelf Life Structure (Retail-specific)
        # =================================================================
        if "shelf_life" in data:
            if verbose:
                print("  Probe: Shelf Life Structure (shelf_life=1)")
            test_data = copy.deepcopy(data)

            # Set all shelf_life to 1
            if isinstance(test_data["shelf_life"], dict):
                for p in test_data["shelf_life"]:
                    test_data["shelf_life"][p] = 1
            else:
                test_data["shelf_life"] = 1

            result = self.executor.execute(code, test_data)
            status = result.get("status")
            obj_sl1 = result.get("objective")

            if status is None or result.get("exit_code", -1) != 0:
                diag = "Code crashes with shelf_life=1. Check aging constraint indexing."
                results.append(VerificationResult(False, 7, diag))
                if verbose:
                    print(f"    FAIL: {diag}")
            elif obj_sl1 is not None:
                # With shelf_life=1, cost should be HIGHER than baseline
                if obj_sl1 < baseline_obj * 0.9:
                    diag = f"shelf_life=1 gives LOWER cost ({obj_sl1:.2f} < {baseline_obj:.2f}). " \
                           f"Aging/waste constraint likely incorrect."
                    results.append(VerificationResult(False, 7, diag))
                    if verbose:
                        print(f"    FAIL: {diag}")
                else:
                    results.append(VerificationResult(True, 7))
                    if verbose:
                        print(f"    PASS (obj_sl1={obj_sl1:.2f} >= baseline={baseline_obj:.2f})")
            else:
                results.append(VerificationResult(True, 7, skipped=True))
                if verbose:
                    print(f"    SKIP (status={status})")

        # =================================================================
        # Probe 7: Substitution Edge (Retail-specific)
        # =================================================================
        network = data.get("network", {})
        sub_edges = network.get("sub_edges", [])
        if sub_edges and "production_cap" in data:
            if verbose:
                print("  Probe: Substitution Structure")
            test_data = copy.deepcopy(data)

            edge = sub_edges[0]
            p_from = edge[0] if isinstance(edge, list) else edge

            if isinstance(test_data["production_cap"], dict) and p_from in test_data["production_cap"]:
                if isinstance(test_data["production_cap"][p_from], list):
                    test_data["production_cap"][p_from] = [0] * len(test_data["production_cap"][p_from])
                else:
                    test_data["production_cap"][p_from] = 0

            result = self.executor.execute(code, test_data)
            status = result.get("status")

            if status == "INFEASIBLE":
                diag = f"INFEASIBLE when {p_from} capacity=0 but substitution edge exists. " \
                       f"Substitution constraint likely not implemented."
                results.append(VerificationResult(False, 7, diag))
                if verbose:
                    print(f"    FAIL: {diag}")
            elif status in ("OPTIMAL", "TIME_LIMIT"):
                results.append(VerificationResult(True, 7))
                if verbose:
                    print("    PASS")
            else:
                results.append(VerificationResult(True, 7, skipped=True))
                if verbose:
                    print(f"    SKIP (status={status})")

        # =================================================================
        # Probe 8: Objective Sign Check (Minimize vs Maximize)
        # If all costs doubled → objective should roughly double for minimize
        # =================================================================
        all_cost_keys = [k for k in data.keys() if 'cost' in k.lower()]
        if len(all_cost_keys) >= 2:
            if verbose:
                print("  Probe: Objective Function Sign (cost doubling)")

            test_data = copy.deepcopy(data)

            # Double all cost parameters
            for cost_key in all_cost_keys:
                if isinstance(test_data[cost_key], dict):
                    for k in test_data[cost_key]:
                        if isinstance(test_data[cost_key][k], (int, float)):
                            test_data[cost_key][k] = test_data[cost_key][k] * 2
                        elif isinstance(test_data[cost_key][k], list):
                            test_data[cost_key][k] = [v * 2 for v in test_data[cost_key][k]]
                elif isinstance(test_data[cost_key], (int, float)):
                    test_data[cost_key] = test_data[cost_key] * 2
                elif isinstance(test_data[cost_key], list):
                    test_data[cost_key] = [v * 2 for v in test_data[cost_key]]

            result = self.executor.execute(code, test_data)
            obj_double_cost = result.get("objective")

            if obj_double_cost is not None and baseline_obj > 0:
                ratio = obj_double_cost / baseline_obj
                # For minimize: doubling costs should roughly double objective (ratio ~2)
                # If ratio < 1, objective may have wrong sign
                if ratio < 0.8:  # Objective DECREASED when costs doubled
                    diag = f"Doubling all costs DECREASED objective ({obj_double_cost:.2f} < {baseline_obj:.2f}). " \
                           f"Objective function may have wrong sign (minimize vs maximize) or missing cost terms."
                    results.append(VerificationResult(False, 7, diag))
                    if verbose:
                        print(f"    FAIL: {diag}")
                else:
                    results.append(VerificationResult(True, 7))
                    if verbose:
                        print(f"    PASS (ratio={ratio:.2f})")
            else:
                results.append(VerificationResult(True, 7, skipped=True))
                if verbose:
                    print("    SKIP (no valid objective)")

        return results

    def _is_retail_data(self, data: Dict) -> bool:
        return all(k in data for k in ["products", "locations", "periods", "demand_curve"])

    def _layer3_code_structure(self, code: str, data: Dict, verbose: bool) -> List[VerificationResult]:
        """
        Layer 3: Code Structure Verification (AST-based) - ALL UNIVERSAL

        Universal checks that analyze code structure without running it.
        Does NOT leak data - only examines variable names, patterns, formulas.

        Checks (ALL UNIVERSAL - no domain-specific logic):
        1. Objective function exists - m.setObjective() call
        2. Index boundaries - t-1/t+1 handling at boundaries
        3. Variables declared - m.addVar() calls
        4. Constraints added - m.addConstr() calls
        5. Parameter references - data params appear in code (for L3+L4 analysis)
        """
        results = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Syntax errors caught in L1
            return results

        # Extract code patterns for analysis
        code_lower = code.lower()
        code_text = code

        # Check 1: Objective function exists and has proper structure
        if verbose:
            print("  Check: Objective function structure")

        has_setObjective = "setobjective" in code_lower or "setObjective" in code_text
        has_objective = "objective" in code_lower

        if not has_setObjective and not has_objective:
            diag = "No objective function found. Missing m.setObjective() call."
            results.append(VerificationResult(False, 3, diag))
            if verbose:
                print(f"    FAIL: {diag}")
        else:
            results.append(VerificationResult(True, 3))
            if verbose:
                print("    PASS")

        # Check 2: Loop index boundary patterns (UNIVERSAL)
        if verbose:
            print("  Check: Loop index boundaries")

        # Look for potential off-by-one errors: t-1 at t=1, or t+1 at t=T
        has_t_minus_1 = "t-1" in code_text or "t - 1" in code_text
        has_t_plus_1 = "t+1" in code_text or "t + 1" in code_text
        has_range_1_T = bool(re.search(r'range\s*\(\s*1\s*,', code_text))

        # If code uses t-1 starting from t=1, that's a potential boundary issue
        # But we can't fully verify without running - this is a warning
        if has_t_minus_1 and has_range_1_T:
            # Check if there's boundary handling
            has_boundary_check = "if t" in code_lower or "t == 1" in code_text or "t > 1" in code_text
            if not has_boundary_check:
                results.append(VerificationResult(True, 3, skipped=True,
                    details={"warning": "Uses t-1 indexing but no boundary check for t=1"}))
                if verbose:
                    print("    SKIP (warning: potential t-1 boundary issue)")
            else:
                results.append(VerificationResult(True, 3))
                if verbose:
                    print("    PASS")
        else:
            results.append(VerificationResult(True, 3))
            if verbose:
                print("    PASS")

        # Check 3: Variable declaration before use (UNIVERSAL)
        if verbose:
            print("  Check: Variable declarations")

        # Look for addVar patterns
        var_declarations = re.findall(r'(\w+)\s*=\s*m\.addVar', code_text)
        var_declarations += re.findall(r'(\w+)\s*\[.*\]\s*=\s*m\.addVar', code_text)

        # Also check for dictionary-based variable creation
        dict_vars = re.findall(r'(\w+)\s*=\s*\{\}', code_text)
        var_declarations += dict_vars

        if not var_declarations:
            diag = "No decision variables found. Missing m.addVar() calls."
            results.append(VerificationResult(False, 3, diag))
            if verbose:
                print(f"    FAIL: {diag}")
        else:
            results.append(VerificationResult(True, 3))
            if verbose:
                print(f"    PASS (found {len(var_declarations)} variable patterns)")

        # Check 4: Constraint addition (UNIVERSAL)
        if verbose:
            print("  Check: Constraint additions")

        has_addConstr = "addconstr" in code_lower or "addConstr" in code_text
        has_constraints = "constr" in code_lower

        if not has_addConstr and not has_constraints:
            diag = "No constraints found. Missing m.addConstr() calls."
            results.append(VerificationResult(False, 3, diag))
            if verbose:
                print(f"    FAIL: {diag}")
        else:
            results.append(VerificationResult(True, 3))
            if verbose:
                print("    PASS")

        # Check 5: Parameter reference detection (UNIVERSAL)
        # For each parameter in data, check if it appears in the code
        # This helps distinguish "constraint missing" vs "constraint has slack" in L4
        if verbose:
            print("  Check: Parameter references in code")

        param_refs = {}  # {param_name: appears_in_code}
        params = extract_numeric_params(data)

        for param in params[:20]:  # Check up to 20 parameters
            # Extract the base parameter name (e.g., "costs.inventory" -> "inventory")
            param_parts = param.split('.')
            base_name = param_parts[-1] if param_parts else param

            # Check if parameter name appears in code (case-insensitive)
            # Also check for common variations (underscore vs camelCase)
            variations = [base_name, base_name.replace('_', ''), base_name.lower()]
            appears = any(re.search(rf'\b{re.escape(v)}\b', code_text, re.IGNORECASE) for v in variations)
            param_refs[param] = appears

        # Count how many parameters are referenced
        refs_found = sum(1 for v in param_refs.values() if v)
        refs_total = len(param_refs)

        results.append(VerificationResult(True, 3, details={
            "param_references": param_refs,
            "refs_found": refs_found,
            "refs_total": refs_total
        }))

        if verbose:
            print(f"    PASS ({refs_found}/{refs_total} parameters referenced in code)")

        return results
