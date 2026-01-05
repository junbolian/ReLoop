# ==============================================================================
# FILE: semantic_probes.py
# DESCRIPTION: Semantic Probe Verification Framework
#
# Core Idea:
#   LLM-generated code may "run but produce wrong answers" (Silent Failure).
#   By constructing boundary test cases, we verify if key constraints are correct.
#
# Design Principles:
#   1. Each probe tests ONE specific mechanism
#   2. Test data is carefully constructed so correct/incorrect behavior differs
#   3. Judgment based on observable outputs (objective, status), no code parsing
#
# ==============================================================================

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import os
import base64
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from ..schemas import ProbeResult, SemanticProbeReport


class ProbeStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    CRASH = "CRASH"
    TIMEOUT = "TIMEOUT"


# ==============================================================================
# BASE PROBE CLASS
# ==============================================================================

class SemanticProbe:
    """Base class for all semantic probes."""
    
    name: str = "base_probe"
    description: str = "Base probe"
    target_mechanism: str = "unknown"
    
    def generate_test_data(self, base_data: dict = None) -> dict:
        """Generate test data that isolates the mechanism being tested."""
        raise NotImplementedError
    
    def expected_behavior(self, test_data: dict) -> dict:
        """Return expected values for key variables/objective."""
        raise NotImplementedError
    
    def check_result(self, test_data: dict, actual_result: dict) -> ProbeResult:
        """Compare actual result against expected behavior."""
        raise NotImplementedError


# ==============================================================================
# PROBE 1: SUBSTITUTION SEMANTICS
# ==============================================================================

class SubstitutionProbe(SemanticProbe):
    """
    Test if substitution constraint is correctly implemented.
    
    Scenario:
    - 2 products: Basic, Premium
    - 1 location, 1 period
    - Basic: demand=100, production=0 (stockout)
    - Premium: demand=0, production=80
    - Edge: [Basic, Premium] means Basic's demand can be served by Premium
    
    Correct behavior:
    - S[Basic, Premium] = 80 (Premium serves Basic's demand)
    - L[Basic] = 20 (remaining lost sales)
    """
    
    name = "substitution_basic"
    description = "Test if substitution constraint is correctly implemented"
    target_mechanism = "substitution"
    
    def generate_test_data(self, base_data: dict = None) -> dict:
        return {
            "name": "probe_substitution",
            "periods": 1,
            "products": ["SKU_Basic", "SKU_Premium"],
            "locations": ["Loc_A"],
            "shelf_life": {"SKU_Basic": 2, "SKU_Premium": 2},
            "demand_curve": {"SKU_Basic": [100], "SKU_Premium": [0]},
            "demand_share": {"Loc_A": 1.0},
            "production_cap": {"SKU_Basic": [0], "SKU_Premium": [80]},
            "cold_capacity": {"Loc_A": 10000},
            "cold_usage": {"SKU_Basic": 1, "SKU_Premium": 1},
            "network": {
                "sub_edges": [["SKU_Basic", "SKU_Premium"]],
                "trans_edges": []
            },
            "costs": {
                "inventory": {"SKU_Basic": 1, "SKU_Premium": 1},
                "waste": {"SKU_Basic": 10, "SKU_Premium": 10},
                "lost_sales": {"SKU_Basic": 50, "SKU_Premium": 50},
                "purchasing": {"SKU_Basic": 5, "SKU_Premium": 8}
            },
            "constraints": {}
        }
    
    def expected_behavior(self, test_data: dict) -> dict:
        return {
            "S_Basic_Premium": 80,
            "L_Basic": 20,
            "objective_range": (1000, 2000)
        }
    
    def check_result(self, test_data: dict, actual_result: dict) -> ProbeResult:
        expected = self.expected_behavior(test_data)
        
        if actual_result.get("status") not in ("2", "OPTIMAL", "GRB.OPTIMAL"):
            return ProbeResult(
                probe_name=self.name,
                result="CRASH",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Model did not solve optimally: {actual_result.get('status')}"
            )
        
        obj = actual_result.get("objective", 0)
        if obj is None:
            obj = 0
        obj_min, obj_max = expected["objective_range"]
        
        if obj < obj_min * 0.3:
            return ProbeResult(
                probe_name=self.name,
                result="FAIL",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Objective too low ({obj} < {obj_min}). "
                          f"Missing demand_route constraint (S_out <= demand)."
            )
        
        if obj > obj_max * 3:
            return ProbeResult(
                probe_name=self.name,
                result="FAIL",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Objective too high ({obj} > {obj_max}). "
                          f"Substitution may not be implemented."
            )
        
        return ProbeResult(
            probe_name=self.name,
            result="PASS",
            expected=expected,
            actual=actual_result,
            diagnosis="Substitution appears correctly implemented."
        )


# ==============================================================================
# PROBE 2: DEMAND ROUTE CONSTRAINT
# ==============================================================================

class DemandRouteProbe(SemanticProbe):
    """
    Test if demand_route: S_out <= demand is enforced.
    
    If missing, model may become UNBOUNDED or have suspiciously low objective.
    """
    
    name = "demand_route_constraint"
    description = "Test if demand_route (S_out <= demand) is enforced"
    target_mechanism = "substitution"
    
    def generate_test_data(self, base_data: dict = None) -> dict:
        return {
            "name": "probe_demand_route",
            "periods": 1,
            "products": ["SKU_Basic", "SKU_Premium"],
            "locations": ["Loc_A"],
            "shelf_life": {"SKU_Basic": 2, "SKU_Premium": 2},
            "demand_curve": {"SKU_Basic": [10], "SKU_Premium": [0]},
            "demand_share": {"Loc_A": 1.0},
            "production_cap": {"SKU_Basic": [0], "SKU_Premium": [1000]},
            "cold_capacity": {"Loc_A": 10000},
            "cold_usage": {"SKU_Basic": 1, "SKU_Premium": 1},
            "network": {
                "sub_edges": [["SKU_Basic", "SKU_Premium"]],
                "trans_edges": []
            },
            "costs": {
                "inventory": {"SKU_Basic": 1, "SKU_Premium": 0.01},
                "waste": {"SKU_Basic": 10, "SKU_Premium": 0.01},
                "lost_sales": {"SKU_Basic": 1000, "SKU_Premium": 1000},
                "purchasing": {"SKU_Basic": 5, "SKU_Premium": 0.01}
            },
            "constraints": {}
        }
    
    def expected_behavior(self, test_data: dict) -> dict:
        return {"min_objective": 0.05, "max_reasonable": 200}
    
    def check_result(self, test_data: dict, actual_result: dict) -> ProbeResult:
        expected = self.expected_behavior(test_data)
        
        if actual_result.get("status") in ("5", "UNBOUNDED", "GRB.UNBOUNDED"):
            return ProbeResult(
                probe_name=self.name,
                result="FAIL",
                expected=expected,
                actual=actual_result,
                diagnosis="Model is UNBOUNDED. Missing demand_route constraint."
            )
        
        if actual_result.get("status") not in ("2", "OPTIMAL", "GRB.OPTIMAL"):
            return ProbeResult(
                probe_name=self.name,
                result="CRASH",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Model status: {actual_result.get('status')}"
            )
        
        obj = actual_result.get("objective", 0) or 0
        
        if obj < expected["min_objective"]:
            return ProbeResult(
                probe_name=self.name,
                result="FAIL",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Objective suspiciously low ({obj}). Check demand_route constraint."
            )
        
        return ProbeResult(
            probe_name=self.name,
            result="PASS",
            expected=expected,
            actual=actual_result,
            diagnosis="demand_route constraint appears to be working."
        )


# ==============================================================================
# PROBE 3: NO SUBSTITUTION WHEN EMPTY
# ==============================================================================

class NoSubstitutionProbe(SemanticProbe):
    """
    Test that substitution does not occur when sub_edges is empty.
    """
    
    name = "no_substitution"
    description = "Test that substitution does not occur when sub_edges is empty"
    target_mechanism = "substitution"
    
    def generate_test_data(self, base_data: dict = None) -> dict:
        return {
            "name": "probe_no_sub",
            "periods": 1,
            "products": ["SKU_Basic", "SKU_Premium"],
            "locations": ["Loc_A"],
            "shelf_life": {"SKU_Basic": 2, "SKU_Premium": 2},
            "demand_curve": {"SKU_Basic": [100], "SKU_Premium": [0]},
            "demand_share": {"Loc_A": 1.0},
            "production_cap": {"SKU_Basic": [0], "SKU_Premium": [100]},
            "cold_capacity": {"Loc_A": 10000},
            "cold_usage": {"SKU_Basic": 1, "SKU_Premium": 1},
            "network": {"sub_edges": [], "trans_edges": []},  # EMPTY!
            "costs": {
                "inventory": {"SKU_Basic": 1, "SKU_Premium": 1},
                "waste": {"SKU_Basic": 10, "SKU_Premium": 10},
                "lost_sales": {"SKU_Basic": 50, "SKU_Premium": 50},
                "purchasing": {"SKU_Basic": 5, "SKU_Premium": 5}
            },
            "constraints": {}
        }
    
    def expected_behavior(self, test_data: dict) -> dict:
        # Basic 100 units all lost sales = 100 * 50 = 5000
        return {"lost_sales_basic": 100, "objective_range": (4500, 5500)}
    
    def check_result(self, test_data: dict, actual_result: dict) -> ProbeResult:
        expected = self.expected_behavior(test_data)
        
        if actual_result.get("status") not in ("2", "OPTIMAL", "GRB.OPTIMAL"):
            return ProbeResult(
                probe_name=self.name,
                result="CRASH",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Model status: {actual_result.get('status')}"
            )
        
        obj = actual_result.get("objective", 0) or 0
        obj_min, obj_max = expected["objective_range"]
        
        if obj < obj_min * 0.5:
            return ProbeResult(
                probe_name=self.name,
                result="FAIL",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Objective too low ({obj}). Substitution happening despite empty sub_edges."
            )
        
        return ProbeResult(
            probe_name=self.name,
            result="PASS",
            expected=expected,
            actual=actual_result,
            diagnosis="No spurious substitution when sub_edges is empty."
        )


# ==============================================================================
# PROBE 4: PRODUCTION CAPACITY
# ==============================================================================

class ProductionCapacityProbe(SemanticProbe):
    """
    Test if production capacity is correctly enforced.
    """
    
    name = "production_capacity"
    description = "Test if production capacity is correctly enforced"
    target_mechanism = "capacity"
    
    def generate_test_data(self, base_data: dict = None) -> dict:
        return {
            "name": "probe_prod_cap",
            "periods": 1,
            "products": ["SKU_A"],
            "locations": ["Loc_A"],
            "shelf_life": {"SKU_A": 2},
            "demand_curve": {"SKU_A": [200]},  # demand = 200
            "demand_share": {"Loc_A": 1.0},
            "production_cap": {"SKU_A": [50]},  # can only produce 50
            "cold_capacity": {"Loc_A": 10000},
            "cold_usage": {"SKU_A": 1},
            "network": {"sub_edges": [], "trans_edges": []},
            "costs": {
                "inventory": {"SKU_A": 1},
                "waste": {"SKU_A": 10},
                "lost_sales": {"SKU_A": 100},
                "purchasing": {"SKU_A": 5}
            },
            "constraints": {}
        }
    
    def expected_behavior(self, test_data: dict) -> dict:
        # Produce 50, lost sales 150: 50*5 + 150*100 = 250 + 15000 = 15250
        return {"objective_range": (15000, 16000)}
    
    def check_result(self, test_data: dict, actual_result: dict) -> ProbeResult:
        expected = self.expected_behavior(test_data)
        
        if actual_result.get("status") not in ("2", "OPTIMAL", "GRB.OPTIMAL"):
            return ProbeResult(
                probe_name=self.name,
                result="CRASH",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Model status: {actual_result.get('status')}"
            )
        
        obj = actual_result.get("objective", 0) or 0
        obj_min, obj_max = expected["objective_range"]
        
        if obj < obj_min * 0.5:
            return ProbeResult(
                probe_name=self.name,
                result="FAIL",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Objective too low ({obj}). Production capacity not enforced."
            )
        
        return ProbeResult(
            probe_name=self.name,
            result="PASS",
            expected=expected,
            actual=actual_result,
            diagnosis="Production capacity constraint appears correct."
        )


# ==============================================================================
# PROBE 5: STORAGE CAPACITY
# ==============================================================================

class StorageCapacityProbe(SemanticProbe):
    """
    Test if storage capacity is correctly enforced.
    """
    
    name = "storage_capacity"
    description = "Test if storage capacity is correctly enforced"
    target_mechanism = "capacity"
    
    def generate_test_data(self, base_data: dict = None) -> dict:
        return {
            "name": "probe_storage_cap",
            "periods": 2,
            "products": ["SKU_A"],
            "locations": ["Loc_A"],
            "shelf_life": {"SKU_A": 3},
            "demand_curve": {"SKU_A": [20, 20]},
            "demand_share": {"Loc_A": 1.0},
            "production_cap": {"SKU_A": [100, 100]},
            "cold_capacity": {"Loc_A": 30},  # tight storage
            "cold_usage": {"SKU_A": 1},
            "network": {"sub_edges": [], "trans_edges": []},
            "costs": {
                "inventory": {"SKU_A": 1},
                "waste": {"SKU_A": 50},
                "lost_sales": {"SKU_A": 200},
                "purchasing": {"SKU_A": 5}
            },
            "constraints": {}
        }
    
    def expected_behavior(self, test_data: dict) -> dict:
        return {"max_inventory": 30, "objective_range": (200, 600)}
    
    def check_result(self, test_data: dict, actual_result: dict) -> ProbeResult:
        expected = self.expected_behavior(test_data)
        
        if actual_result.get("status") in ("3", "INFEASIBLE", "GRB.INFEASIBLE"):
            return ProbeResult(
                probe_name=self.name,
                result="FAIL",
                expected=expected,
                actual=actual_result,
                diagnosis="Model INFEASIBLE. Storage constraint may be incorrectly formulated."
            )
        
        if actual_result.get("status") not in ("2", "OPTIMAL", "GRB.OPTIMAL"):
            return ProbeResult(
                probe_name=self.name,
                result="CRASH",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Model status: {actual_result.get('status')}"
            )
        
        return ProbeResult(
            probe_name=self.name,
            result="PASS",
            expected=expected,
            actual=actual_result,
            diagnosis="Storage capacity appears to be enforced."
        )


# ==============================================================================
# PROBE 6: AGING DYNAMICS
# ==============================================================================

class AgingProbe(SemanticProbe):
    """
    Test if inventory aging is correctly implemented.
    """
    
    name = "aging_dynamics"
    description = "Test if inventory aging is correctly implemented"
    target_mechanism = "shelf_life"
    
    def generate_test_data(self, base_data: dict = None) -> dict:
        return {
            "name": "probe_aging",
            "periods": 3,
            "products": ["SKU_A"],
            "locations": ["Loc_A"],
            "shelf_life": {"SKU_A": 2},  # 2-period shelf life
            "demand_curve": {"SKU_A": [0, 50, 30]},
            "demand_share": {"Loc_A": 1.0},
            "production_cap": {"SKU_A": [100, 0, 100]},
            "cold_capacity": {"Loc_A": 10000},
            "cold_usage": {"SKU_A": 1},
            "network": {"sub_edges": [], "trans_edges": []},
            "costs": {
                "inventory": {"SKU_A": 1},
                "waste": {"SKU_A": 20},
                "lost_sales": {"SKU_A": 100},
                "purchasing": {"SKU_A": 5}
            },
            "constraints": {}
        }
    
    def expected_behavior(self, test_data: dict) -> dict:
        # T1: produce 100, no demand
        # T2: demand 50, sell 50, 50 remains (ages to a=1)
        # T3: 50 expires -> waste=50
        return {"total_waste": 50, "objective_range": (1000, 2500)}
    
    def check_result(self, test_data: dict, actual_result: dict) -> ProbeResult:
        expected = self.expected_behavior(test_data)
        
        if actual_result.get("status") not in ("2", "OPTIMAL", "GRB.OPTIMAL"):
            return ProbeResult(
                probe_name=self.name,
                result="CRASH",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Model status: {actual_result.get('status')}"
            )
        
        obj = actual_result.get("objective", 0) or 0
        obj_min, obj_max = expected["objective_range"]
        
        if obj < obj_min * 0.3:
            return ProbeResult(
                probe_name=self.name,
                result="FAIL",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Objective too low ({obj}). Aging/waste may not be implemented."
            )
        
        return ProbeResult(
            probe_name=self.name,
            result="PASS",
            expected=expected,
            actual=actual_result,
            diagnosis="Aging dynamics appear correct."
        )


# ==============================================================================
# PROBE 7: LOST SALES SLACK
# ==============================================================================

class LostSalesProbe(SemanticProbe):
    """
    Test if lost sales variable L is correctly implemented.
    Without L, model may be INFEASIBLE when demand > supply.
    """
    
    name = "lost_sales_slack"
    description = "Test if lost sales slack variable is implemented"
    target_mechanism = "lost_sales"
    
    def generate_test_data(self, base_data: dict = None) -> dict:
        return {
            "name": "probe_lost_sales",
            "periods": 1,
            "products": ["SKU_A"],
            "locations": ["Loc_A"],
            "shelf_life": {"SKU_A": 2},
            "demand_curve": {"SKU_A": [1000]},  # high demand
            "demand_share": {"Loc_A": 1.0},
            "production_cap": {"SKU_A": [10]},  # low capacity
            "cold_capacity": {"Loc_A": 10000},
            "cold_usage": {"SKU_A": 1},
            "network": {"sub_edges": [], "trans_edges": []},
            "costs": {
                "inventory": {"SKU_A": 1},
                "waste": {"SKU_A": 10},
                "lost_sales": {"SKU_A": 50},
                "purchasing": {"SKU_A": 5}
            },
            "constraints": {}
        }
    
    def expected_behavior(self, test_data: dict) -> dict:
        # Produce 10, lost sales 990: 10*5 + 990*50 = 50 + 49500 = 49550
        return {"objective_range": (49000, 50500)}
    
    def check_result(self, test_data: dict, actual_result: dict) -> ProbeResult:
        expected = self.expected_behavior(test_data)
        
        if actual_result.get("status") in ("3", "INFEASIBLE", "GRB.INFEASIBLE"):
            return ProbeResult(
                probe_name=self.name,
                result="FAIL",
                expected=expected,
                actual=actual_result,
                diagnosis="Model INFEASIBLE. Missing lost sales variable L as slack."
            )
        
        if actual_result.get("status") not in ("2", "OPTIMAL", "GRB.OPTIMAL"):
            return ProbeResult(
                probe_name=self.name,
                result="CRASH",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Model status: {actual_result.get('status')}"
            )
        
        return ProbeResult(
            probe_name=self.name,
            result="PASS",
            expected=expected,
            actual=actual_result,
            diagnosis="Lost sales slack correctly implemented."
        )


# ==============================================================================
# PROBE 8: INVENTORY NON-NEGATIVITY
# ==============================================================================

class NonNegativityProbe(SemanticProbe):
    """
    Test if inventory non-negativity is enforced.
    """
    
    name = "inventory_nonnegativity"
    description = "Test if inventory non-negativity is enforced"
    target_mechanism = "bounds"
    
    def generate_test_data(self, base_data: dict = None) -> dict:
        return {
            "name": "probe_nonneg",
            "periods": 2,
            "products": ["SKU_A"],
            "locations": ["Loc_A"],
            "shelf_life": {"SKU_A": 3},
            "demand_curve": {"SKU_A": [50, 50]},
            "demand_share": {"Loc_A": 1.0},
            "production_cap": {"SKU_A": [40, 40]},  # < demand each period
            "cold_capacity": {"Loc_A": 10000},
            "cold_usage": {"SKU_A": 1},
            "network": {"sub_edges": [], "trans_edges": []},
            "costs": {
                "inventory": {"SKU_A": 1},
                "waste": {"SKU_A": 10},
                "lost_sales": {"SKU_A": 100},
                "purchasing": {"SKU_A": 5}
            },
            "constraints": {}
        }
    
    def expected_behavior(self, test_data: dict) -> dict:
        # Each period: produce 40, demand 50, lost sales 10
        # Total: 2*40*5 + 2*10*100 = 400 + 2000 = 2400
        return {"objective_range": (2200, 2600)}
    
    def check_result(self, test_data: dict, actual_result: dict) -> ProbeResult:
        expected = self.expected_behavior(test_data)
        
        if actual_result.get("status") not in ("2", "OPTIMAL", "GRB.OPTIMAL"):
            return ProbeResult(
                probe_name=self.name,
                result="CRASH",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Model status: {actual_result.get('status')}"
            )
        
        obj = actual_result.get("objective", 0) or 0
        obj_min, obj_max = expected["objective_range"]
        
        # If objective is way too low, might be allowing negative inventory
        if obj < obj_min * 0.3:
            return ProbeResult(
                probe_name=self.name,
                result="FAIL",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Objective too low ({obj}). May be allowing negative inventory."
            )
        
        return ProbeResult(
            probe_name=self.name,
            result="PASS",
            expected=expected,
            actual=actual_result,
            diagnosis="Inventory non-negativity appears correct."
        )


# ==============================================================================
# PROBE RUNNER
# ==============================================================================

class ProbeRunner:
    """Run all probes against a generated model."""
    
    def __init__(self):
        self.probes = [
            SubstitutionProbe(),
            DemandRouteProbe(),
            NoSubstitutionProbe(),
            ProductionCapacityProbe(),
            StorageCapacityProbe(),
            AgingProbe(),
            LostSalesProbe(),
            NonNegativityProbe(),
        ]
    
    def run_model_code(self, code: str, data: dict, timeout: int = 60) -> dict:
        """Execute generated code with test data and return results."""
        
        import os
        env = os.environ.copy()
        env["RELOOP_CODE_B64"] = base64.b64encode(code.encode("utf-8")).decode("utf-8")
        env["RELOOP_DATA_B64"] = base64.b64encode(
            json.dumps(data).encode("utf-8")
        ).decode("utf-8")

        runner = '''
import base64, json, os, sys, time
try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None

code = base64.b64decode(os.environ["RELOOP_CODE_B64"]).decode("utf-8")
data = json.loads(base64.b64decode(os.environ["RELOOP_DATA_B64"]).decode("utf-8"))
globals_ns = {"data": data, "__name__": "__main__"}

solve_meta = {"status": None, "objective": None}

try:
    exec(compile(code, "<generated>", "exec"), globals_ns)
except Exception as e:
    solve_meta["error"] = str(e)
finally:
    model = globals_ns.get("m") or globals_ns.get("model")
    if model is not None:
        try:
            solve_meta["status"] = str(getattr(model, "Status", None))
            solve_meta["objective"] = getattr(model, "objVal", None)
        except:
            pass
    print("PROBE_RESULT::" + json.dumps(solve_meta))
'''
        
        try:
            proc = subprocess.run(
                [sys.executable, "-c", runner],
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout,
            )
            
            for line in proc.stdout.splitlines():
                if line.startswith("PROBE_RESULT::"):
                    payload = line.split("PROBE_RESULT::", 1)[1]
                    try:
                        return json.loads(payload)
                    except:
                        pass
            
            return {"status": "UNKNOWN", "objective": None, "stderr": proc.stderr}
            
        except subprocess.TimeoutExpired:
            return {"status": "TIMEOUT", "objective": None}
        except Exception as e:
            return {"status": "ERROR", "objective": None, "error": str(e)}
    
    def run_all_probes(self, model_code: str) -> SemanticProbeReport:
        """Run all probes and return aggregate report."""
        probe_results: List[ProbeResult] = []
        
        for probe in self.probes:
            test_data = probe.generate_test_data()
            result = self.run_model_code(model_code, test_data)
            report = probe.check_result(test_data, result)
            probe_results.append(report)
        
        passed = sum(1 for r in probe_results if r.result == "PASS")
        failed = sum(1 for r in probe_results if r.result == "FAIL")
        crashed = sum(1 for r in probe_results if r.result in ("CRASH", "TIMEOUT"))
        
        return SemanticProbeReport(
            total=len(probe_results),
            passed=passed,
            failed=failed,
            crashed=crashed,
            pass_rate=passed / len(probe_results) if probe_results else 0,
            failed_probes=[r.probe_name for r in probe_results if r.result == "FAIL"],
            probe_results=probe_results,
            diagnoses={r.probe_name: r.diagnosis for r in probe_results if r.result != "PASS"}
        )
    
    def run_selected_probes(self, model_code: str, probe_names: List[str]) -> SemanticProbeReport:
        """Run selected probes only."""
        probe_results: List[ProbeResult] = []
        
        for probe in self.probes:
            if probe.name in probe_names:
                test_data = probe.generate_test_data()
                result = self.run_model_code(model_code, test_data)
                report = probe.check_result(test_data, result)
                probe_results.append(report)
        
        passed = sum(1 for r in probe_results if r.result == "PASS")
        failed = sum(1 for r in probe_results if r.result == "FAIL")
        crashed = sum(1 for r in probe_results if r.result in ("CRASH", "TIMEOUT"))
        
        return SemanticProbeReport(
            total=len(probe_results),
            passed=passed,
            failed=failed,
            crashed=crashed,
            pass_rate=passed / len(probe_results) if probe_results else 0,
            failed_probes=[r.probe_name for r in probe_results if r.result == "FAIL"],
            probe_results=probe_results,
            diagnoses={r.probe_name: r.diagnosis for r in probe_results if r.result != "PASS"}
        )


def get_probe_diagnosis(model_code: str) -> str:
    """
    Get probe diagnosis string for repair prompts.
    """
    runner = ProbeRunner()
    report = runner.run_all_probes(model_code)
    
    if report.failed == 0 and report.crashed == 0:
        return "All semantic probes passed."
    
    lines = [f"Semantic probe failures ({report.failed} failed, {report.crashed} crashed):"]
    for probe_result in report.probe_results:
        if probe_result.result != "PASS":
            lines.append(f"- {probe_result.probe_name}: {probe_result.diagnosis}")
    
    return "\n".join(lines)
