"""
Module 2: Behavioral Verification (CORE)

6-Layer verification system:
  Layer 1: Execution - Does code run without errors?
  Layer 2: Feasibility - OPTIMAL? INFEASIBLE? UNBOUNDED?
  Layer 3: Monotonicity - Does each parameter affect objective? (CORE)
  Layer 4: Sensitivity Direction - Does direction match expectation? (Best effort)
  Layer 5: Boundary - Extreme value behavior
  Layer 6: Domain Probes - Domain-specific tests (Optional)

Key Insight: L3 is the universal core - no domain knowledge needed.
"""

import subprocess
import sys
import json
import base64
import re
import time
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

    def count_layers_passed(self) -> int:
        return 6 if self.failed_layer is None else self.failed_layer - 1


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
            output["status"] = {2: "OPTIMAL", 3: "INFEASIBLE", 5: "UNBOUNDED"}.get(code, f"CODE_{code}")
        obj_match = re.search(r"objective:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", result.stdout)
        if obj_match:
            output["objective"] = float(obj_match.group(1))
        return output


class BehavioralVerifier:
    """6-Layer Behavioral Verification System"""

    def __init__(self, delta: float = 0.2, epsilon: float = 1e-4, timeout: int = 60):
        self.delta = delta
        self.epsilon = epsilon
        self.executor = CodeExecutor(timeout=timeout)
        self.max_params = 10

    def verify(self, code: str, data: Dict[str, Any], obj_sense: str = "minimize",
               enable_layer6: bool = False, verbose: bool = False) -> VerificationReport:
        start_time = time.time()
        results = []

        if verbose:
            print("=" * 60 + "\nBehavioral Verification\n" + "=" * 60)

        # Layer 1: Execution
        if verbose:
            print("\n[L1: Execution]")
        l1_result, baseline = self._layer1_execution(code, data)
        results.append(l1_result)
        if not l1_result.passed:
            if verbose:
                print(f"  FAIL: {l1_result.diagnosis}")
            return VerificationReport(False, results, 1, l1_result.diagnosis, time.time() - start_time)
        if verbose:
            print(f"  PASS: status={baseline.get('status')}, obj={baseline.get('objective')}")

        # Layer 2: Feasibility
        if verbose:
            print("\n[L2: Feasibility]")
        l2_result = self._layer2_feasibility(baseline)
        # Store objective in L2 details for extraction
        l2_result.details["objective"] = baseline.get("objective")
        results.append(l2_result)
        if not l2_result.passed:
            if verbose:
                print(f"  FAIL: {l2_result.diagnosis}")
            return VerificationReport(False, results, 2, l2_result.diagnosis, time.time() - start_time)
        if verbose:
            print("  PASS")

        params = extract_numeric_params(data)
        baseline_obj = baseline.get("objective")
        if baseline_obj is None:
            return VerificationReport(False, results, 2, "No objective", time.time() - start_time)

        # Layer 3: Monotonicity (CORE)
        if verbose:
            print(f"\n[L3: Monotonicity] Testing {min(len(params), self.max_params)} parameters")
        l3_results = self._layer3_monotonicity(code, data, params, baseline_obj, verbose)
        results.extend(l3_results)
        failed = [r for r in l3_results if not r.passed and not r.skipped]
        if failed:
            return VerificationReport(False, results, 3, failed[0].diagnosis, time.time() - start_time)

        # Layer 4: Direction
        if verbose:
            print("\n[L4: Sensitivity Direction]")
        l4_results = self._layer4_direction(code, data, params, baseline_obj, obj_sense, verbose)
        results.extend(l4_results)
        failed = [r for r in l4_results if not r.passed and not r.skipped]
        if failed:
            return VerificationReport(False, results, 4, failed[0].diagnosis, time.time() - start_time)

        # Layer 5: Boundary
        if verbose:
            print("\n[L5: Boundary]")
        l5_results = self._layer5_boundary(code, data, params, baseline_obj, obj_sense, verbose)
        results.extend(l5_results)
        failed = [r for r in l5_results if not r.passed and not r.skipped]
        if failed:
            return VerificationReport(False, results, 5, failed[0].diagnosis, time.time() - start_time)

        # Layer 6: Domain Probes (Optional)
        if enable_layer6 and self._is_retail_data(data):
            if verbose:
                print("\n[L6: Domain Probes]")
            l6_results = self._layer6_domain_probes(code, data, baseline_obj, verbose)
            results.extend(l6_results)
            failed = [r for r in l6_results if not r.passed and not r.skipped]
            if failed:
                return VerificationReport(False, results, 6, failed[0].diagnosis, time.time() - start_time)

        if verbose:
            print("\n" + "=" * 60 + "\nVERIFICATION PASSED\n" + "=" * 60)
        return VerificationReport(True, results, execution_time=time.time() - start_time)

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

    def _layer2_feasibility(self, baseline: Dict) -> VerificationResult:
        status = baseline.get("status")
        if status == "OPTIMAL":
            return VerificationResult(True, 2)
        if status == "INFEASIBLE":
            return VerificationResult(False, 2, "Model is INFEASIBLE. Add slack variables or check constraints.")
        if status == "UNBOUNDED":
            return VerificationResult(False, 2, "Model is UNBOUNDED. Add bounds or missing constraints.")
        return VerificationResult(False, 2, f"Unexpected status: {status}")

    def _layer3_monotonicity(self, code: str, data: Dict, params: List[str],
                              baseline_obj: float, verbose: bool) -> List[VerificationResult]:
        results = []
        for param in params[:self.max_params]:
            # Check if parameter should be skipped (e.g., value is zero)
            skip, skip_reason = should_skip_param(data, param)
            if skip:
                results.append(VerificationResult(True, 3, skipped=True,
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
                results.append(VerificationResult(True, 3, skipped=True))
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
                results.append(VerificationResult(False, 3, diag))
                if verbose:
                    print(f"    FAIL: No effect detected!")
            else:
                results.append(VerificationResult(True, 3))
                if verbose:
                    print(f"    PASS (effect detected)")
        return results

    def _layer4_direction(self, code: str, data: Dict, params: List[str],
                          baseline_obj: float, obj_sense: str, verbose: bool) -> List[VerificationResult]:
        results = []
        is_min = obj_sense.lower() == "minimize"
        for param in params[:self.max_params]:
            role = infer_param_role(param)
            if role == ParameterRole.UNKNOWN:
                results.append(VerificationResult(True, 4, skipped=True))
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
                    results.append(VerificationResult(True, 4))
                    if verbose:
                        print("    PASS (INFEASIBLE acceptable)")
                else:
                    results.append(VerificationResult(True, 4, skipped=True))
                continue
            change = test_obj - baseline_obj
            actual = "increase" if change > self.epsilon * abs(baseline_obj) else "decrease" if change < -self.epsilon * abs(baseline_obj) else "none"
            if expected == actual or actual == "none":
                results.append(VerificationResult(True, 4))
                if verbose:
                    print("    PASS")
            else:
                diag = f"Parameter '{param}' (role: {role.value}) shows UNEXPECTED DIRECTION. Expected {expected}, got {actual}."
                results.append(VerificationResult(False, 4, diag))
                if verbose:
                    print(f"    FAIL: Expected {expected}, got {actual}")
        return results

    def _layer5_boundary(self, code: str, data: Dict, params: List[str],
                         baseline_obj: float, obj_sense: str, verbose: bool) -> List[VerificationResult]:
        results = []
        cap_params = [p for p in params if infer_param_role(p) == ParameterRole.CAPACITY]
        for param in cap_params[:3]:
            # Skip Big M or zero-value parameters
            skip, skip_reason = should_skip_param(data, param)
            if skip:
                if verbose:
                    print(f"  Boundary test: {param} = 0")
                    print(f"    SKIP ({skip_reason})")
                results.append(VerificationResult(True, 5, skipped=True,
                                                   details={"skip_reason": skip_reason}))
                continue

            if verbose:
                print(f"  Boundary test: {param} = 0")
            result = self.executor.execute(code, set_param(data, param, 0))
            status, obj = result.get("status"), result.get("objective")
            if status == "INFEASIBLE":
                results.append(VerificationResult(True, 5))
                if verbose:
                    print("    PASS (INFEASIBLE)")
            elif obj is not None:
                if abs(obj - baseline_obj) < 0.1 * abs(baseline_obj):
                    results.append(VerificationResult(False, 5, f"Setting '{param}' to ZERO has no effect."))
                    if verbose:
                        print("    FAIL: No effect")
                else:
                    results.append(VerificationResult(True, 5))
                    if verbose:
                        print("    PASS")
        return results

    def _layer6_domain_probes(self, code: str, data: Dict, baseline_obj: float, verbose: bool) -> List[VerificationResult]:
        import copy
        results = []

        # Probe: Lost Sales
        if verbose:
            print("  Probe: Lost Sales Variable")
        test_data = copy.deepcopy(data)
        if "demand_curve" in test_data:
            for p in test_data["demand_curve"]:
                test_data["demand_curve"][p] = [v * 10 for v in test_data["demand_curve"][p]]
        result = self.executor.execute(code, test_data)
        if result.get("status") == "INFEASIBLE":
            results.append(VerificationResult(False, 6, "INFEASIBLE when demand >> supply. Missing lost sales variable."))
            if verbose:
                print("    FAIL")
        else:
            results.append(VerificationResult(True, 6))
            if verbose:
                print("    PASS")

        return results

    def _is_retail_data(self, data: Dict) -> bool:
        return all(k in data for k in ["products", "locations", "periods", "demand_curve"])
