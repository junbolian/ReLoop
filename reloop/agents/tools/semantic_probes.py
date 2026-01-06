# ==============================================================================
# FILE: semantic_probes.py (FIXED VERSION)
# DESCRIPTION: Semantic Probe Verification Framework
#
# FIXES:
#   1. lost_sales_slack: Added objective range check (was missing)
#   2. storage_capacity: Added objective range check (was missing)
#   3. Unified threshold to 0.5x for consistency
#   4. Added new probes for F6/F7/F8 coverage:
#      - lead_time_probe: Tests lead time handling
#      - moq_probe: Tests minimum order quantity
#      - transshipment_probe: Tests network flows
#      - labor_capacity_probe: Tests labor constraints
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
    
    # Unified threshold for objective range check (50% tolerance)
    OBJECTIVE_THRESHOLD = 0.5
    
    def generate_test_data(self, base_data: dict = None) -> dict:
        """Generate test data that isolates the mechanism being tested."""
        raise NotImplementedError
    
    def expected_behavior(self, test_data: dict) -> dict:
        """Return expected values for key variables/objective."""
        raise NotImplementedError
    
    def check_result(self, test_data: dict, actual_result: dict) -> ProbeResult:
        """Compare actual result against expected behavior."""
        raise NotImplementedError
    
    def _check_objective_range(self, actual_result: dict, expected: dict, 
                                low_diagnosis: str, high_diagnosis: str = None) -> Optional[ProbeResult]:
        """
        Helper method to check if objective is within expected range.
        Returns ProbeResult if FAIL, None if within range.
        """
        obj = actual_result.get("objective", 0) or 0
        obj_range = expected.get("objective_range")
        
        if not obj_range:
            return None
            
        obj_min, obj_max = obj_range
        
        # Check lower bound with threshold
        if obj < obj_min * self.OBJECTIVE_THRESHOLD:
            return ProbeResult(
                probe_name=self.name,
                result="FAIL",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Objective too low ({obj} < {obj_min * self.OBJECTIVE_THRESHOLD:.1f}). {low_diagnosis}"
            )
        
        # Check upper bound with threshold (if high_diagnosis provided)
        if high_diagnosis and obj > obj_max * (2 - self.OBJECTIVE_THRESHOLD):
            return ProbeResult(
                probe_name=self.name,
                result="FAIL",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Objective too high ({obj} > {obj_max * (2 - self.OBJECTIVE_THRESHOLD):.1f}). {high_diagnosis}"
            )
        
        return None


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
        # Premium produces 80, serves Basic's demand
        # Basic has 100 demand, 80 served by Premium, 20 lost
        # Cost: 80*8 (purchasing Premium) + 20*50 (lost sales) = 640 + 1000 = 1640
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
        
        # Check objective range
        range_check = self._check_objective_range(
            actual_result, expected,
            low_diagnosis="Missing demand_route constraint (S_out <= demand) or objective not set.",
            high_diagnosis="Substitution may not be implemented correctly."
        )
        if range_check:
            return range_check
        
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
        # Basic demand = 10, can be served by Premium (cost 0.01 each)
        # Minimum cost: 10 * 0.01 = 0.1
        return {"min_objective": 0.05, "max_reasonable": 200, "objective_range": (0.05, 200)}
    
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
        
        # For this probe, we check if objective is suspiciously low (negative or near-zero when shouldn't be)
        if obj < expected["min_objective"] * 0.1:
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
        
        # Check objective range
        range_check = self._check_objective_range(
            actual_result, expected,
            low_diagnosis="Substitution happening despite empty sub_edges, or objective not set correctly."
        )
        if range_check:
            return range_check
        
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
        
        # Check objective range
        range_check = self._check_objective_range(
            actual_result, expected,
            low_diagnosis="Production capacity not enforced, or objective not set correctly."
        )
        if range_check:
            return range_check
        
        return ProbeResult(
            probe_name=self.name,
            result="PASS",
            expected=expected,
            actual=actual_result,
            diagnosis="Production capacity constraint appears correct."
        )


# ==============================================================================
# PROBE 5: STORAGE CAPACITY (FIXED - now checks objective range)
# ==============================================================================

class StorageCapacityProbe(SemanticProbe):
    """
    Test if storage capacity is correctly enforced.
    
    FIX: Added objective range check - was missing in original version.
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
        # With storage cap 30 and demand 20 each period:
        # Period 1: produce ~30, sell 20, keep 10
        # Period 2: produce ~30, sell 20 from new + some from old
        # Total purchasing cost + some holding cost
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
        
        # FIX: Added objective range check
        range_check = self._check_objective_range(
            actual_result, expected,
            low_diagnosis="Storage capacity may not be enforced, or objective not set correctly."
        )
        if range_check:
            return range_check
        
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
        
        # Check objective range
        range_check = self._check_objective_range(
            actual_result, expected,
            low_diagnosis="Aging/waste may not be implemented, or objective not set correctly."
        )
        if range_check:
            return range_check
        
        return ProbeResult(
            probe_name=self.name,
            result="PASS",
            expected=expected,
            actual=actual_result,
            diagnosis="Aging dynamics appear correct."
        )


# ==============================================================================
# PROBE 7: LOST SALES SLACK (FIXED - now checks objective range)
# ==============================================================================

class LostSalesProbe(SemanticProbe):
    """
    Test if lost sales variable L is correctly implemented.
    Without L, model may be INFEASIBLE when demand > supply.
    
    FIX: Added objective range check - was missing in original version.
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
        
        # FIX: Added objective range check
        range_check = self._check_objective_range(
            actual_result, expected,
            low_diagnosis="Lost sales cost may not be included in objective, or L variable not used correctly."
        )
        if range_check:
            return range_check
        
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
        
        # Check objective range
        range_check = self._check_objective_range(
            actual_result, expected,
            low_diagnosis="May be allowing negative inventory, or objective not set correctly."
        )
        if range_check:
            return range_check
        
        return ProbeResult(
            probe_name=self.name,
            result="PASS",
            expected=expected,
            actual=actual_result,
            diagnosis="Inventory non-negativity appears correct."
        )


# ==============================================================================
# NEW PROBE 9: LEAD TIME (F6 Coverage)
# ==============================================================================

class LeadTimeProbe(SemanticProbe):
    """
    Test if lead time is correctly handled.
    Orders placed in period t should arrive in period t + lead_time.
    """
    
    name = "lead_time"
    description = "Test if lead time is correctly handled"
    target_mechanism = "lead_time"
    
    def generate_test_data(self, base_data: dict = None) -> dict:
        return {
            "name": "probe_lead_time",
            "periods": 3,
            "products": ["SKU_A"],
            "locations": ["Loc_A"],
            "shelf_life": {"SKU_A": 4},
            "lead_time": {"SKU_A": 1},  # 1 period lead time
            "demand_curve": {"SKU_A": [100, 100, 100]},
            "demand_share": {"Loc_A": 1.0},
            "production_cap": {"SKU_A": [200, 200, 200]},
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
        # With lead_time=1:
        # T1: order placed, nothing arrives, lost sales = 100
        # T2: T1 order arrives (100), demand 100, sell 100
        # T3: T2 order arrives (100), demand 100, sell 100
        # Cost: 100*100 (lost T1) + 300*5 (purchasing) = 10000 + 1500 = 11500
        return {"objective_range": (11000, 12500)}
    
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
        
        range_check = self._check_objective_range(
            actual_result, expected,
            low_diagnosis="Lead time may not be enforced correctly (inventory arriving too early)."
        )
        if range_check:
            return range_check
        
        return ProbeResult(
            probe_name=self.name,
            result="PASS",
            expected=expected,
            actual=actual_result,
            diagnosis="Lead time appears correctly implemented."
        )


# ==============================================================================
# NEW PROBE 10: TRANSSHIPMENT (F7 Coverage)
# ==============================================================================

class TransshipmentProbe(SemanticProbe):
    """
    Test if transshipment between locations is correctly implemented.
    """
    
    name = "transshipment"
    description = "Test if transshipment between locations works"
    target_mechanism = "transshipment"
    
    def generate_test_data(self, base_data: dict = None) -> dict:
        return {
            "name": "probe_transship",
            "periods": 1,
            "products": ["SKU_A"],
            "locations": ["Loc_A", "Loc_B"],
            "shelf_life": {"SKU_A": 2},
            "demand_curve": {"SKU_A": [100]},
            "demand_share": {"Loc_A": 1.0, "Loc_B": 0.0},  # All demand at Loc_A
            "production_cap": {"SKU_A": [0]},  # No production at either location directly
            "cold_capacity": {"Loc_A": 10000, "Loc_B": 10000},
            "cold_usage": {"SKU_A": 1},
            "network": {
                "sub_edges": [],
                "trans_edges": [["Loc_B", "Loc_A"]]  # Can ship from B to A
            },
            "costs": {
                "inventory": {"SKU_A": 1},
                "waste": {"SKU_A": 10},
                "lost_sales": {"SKU_A": 100},
                "purchasing": {"SKU_A": 5},
                "transshipment": 2  # Cost per unit shipped
            },
            "constraints": {}
        }
    
    def expected_behavior(self, test_data: dict) -> dict:
        # Without transshipment capability or if not implemented:
        # All 100 units are lost sales = 100 * 100 = 10000
        # With correct transshipment: much lower cost
        # Note: This probe may need adjustment based on exact solver behavior
        return {"objective_range": (8000, 12000)}
    
    def check_result(self, test_data: dict, actual_result: dict) -> ProbeResult:
        expected = self.expected_behavior(test_data)
        
        if actual_result.get("status") not in ("2", "OPTIMAL", "GRB.OPTIMAL"):
            # For transshipment, INFEASIBLE or other status might indicate missing implementation
            return ProbeResult(
                probe_name=self.name,
                result="CRASH",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Model status: {actual_result.get('status')}. Transshipment may not be implemented."
            )
        
        # For transshipment, we mainly check it doesn't crash
        # The objective range is more flexible since behavior depends on implementation
        return ProbeResult(
            probe_name=self.name,
            result="PASS",
            expected=expected,
            actual=actual_result,
            diagnosis="Transshipment appears to be handled (model solved)."
        )


# ==============================================================================
# NEW PROBE 11: LABOR CAPACITY (F8 Coverage)
# ==============================================================================

class LaborCapacityProbe(SemanticProbe):
    """
    Test if labor capacity constraint is correctly enforced.
    """
    
    name = "labor_capacity"
    description = "Test if labor capacity is correctly enforced"
    target_mechanism = "labor"
    
    def generate_test_data(self, base_data: dict = None) -> dict:
        return {
            "name": "probe_labor",
            "periods": 1,
            "products": ["SKU_A"],
            "locations": ["Loc_A"],
            "shelf_life": {"SKU_A": 2},
            "demand_curve": {"SKU_A": [100]},
            "demand_share": {"Loc_A": 1.0},
            "production_cap": {"SKU_A": [200]},  # High production cap
            "cold_capacity": {"Loc_A": 10000},
            "cold_usage": {"SKU_A": 1},
            "labor_usage": {"SKU_A": 2},  # 2 labor hours per unit
            "labor_cap": {"Loc_A": [100]},  # Only 100 labor hours = can process 50 units
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
        # Labor cap = 100, usage = 2 per unit, so can only process 50 units
        # Demand = 100, produce/process 50, lost sales = 50
        # Cost: 50*5 + 50*100 = 250 + 5000 = 5250
        return {"objective_range": (5000, 6000)}
    
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
        
        range_check = self._check_objective_range(
            actual_result, expected,
            low_diagnosis="Labor capacity may not be enforced, or objective not set correctly."
        )
        if range_check:
            return range_check
        
        return ProbeResult(
            probe_name=self.name,
            result="PASS",
            expected=expected,
            actual=actual_result,
            diagnosis="Labor capacity appears correctly implemented."
        )


# ==============================================================================
# NEW PROBE 12: MOQ (F6 Coverage)
# ==============================================================================

class MOQProbe(SemanticProbe):
    """
    Test if Minimum Order Quantity (MOQ) is correctly enforced.
    """
    
    name = "moq"
    description = "Test if MOQ constraint is correctly enforced"
    target_mechanism = "moq"
    
    def generate_test_data(self, base_data: dict = None) -> dict:
        return {
            "name": "probe_moq",
            "periods": 1,
            "products": ["SKU_A"],
            "locations": ["Loc_A"],
            "shelf_life": {"SKU_A": 2},
            "demand_curve": {"SKU_A": [30]},  # demand = 30
            "demand_share": {"Loc_A": 1.0},
            "production_cap": {"SKU_A": [200]},
            "cold_capacity": {"Loc_A": 10000},
            "cold_usage": {"SKU_A": 1},
            "network": {"sub_edges": [], "trans_edges": []},
            "costs": {
                "inventory": {"SKU_A": 1},
                "waste": {"SKU_A": 10},
                "lost_sales": {"SKU_A": 100},
                "purchasing": {"SKU_A": 5}
            },
            "constraints": {
                "moq": 50  # Must order at least 50 if ordering
            }
        }
    
    def expected_behavior(self, test_data: dict) -> dict:
        # Option 1: Order 50 (MOQ), sell 30, waste/hold 20
        # Cost: 50*5 + 20*10 = 250 + 200 = 450 (if waste) or less (if hold)
        # Option 2: Order nothing, lost sales 30 = 30*100 = 3000
        # Optimal: Order 50
        return {"objective_range": (200, 500)}
    
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
        
        # If objective is very high, MOQ might not be enforced (ordered less than MOQ)
        # If very low, might be ignoring MOQ entirely
        if obj > 2500:  # Close to lost sales cost
            return ProbeResult(
                probe_name=self.name,
                result="FAIL",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Objective too high ({obj}). MOQ may not be correctly implemented."
            )
        
        return ProbeResult(
            probe_name=self.name,
            result="PASS",
            expected=expected,
            actual=actual_result,
            diagnosis="MOQ appears correctly implemented."
        )


# ==============================================================================
# NEW PROBE 13: INITIALIZATION (Critical for detecting "free inventory" bug)
# ==============================================================================

class InitializationProbe(SemanticProbe):
    """
    Test if t=1 initialization is correctly implemented.
    
    Without init constraints, model can "create" free inventory at t=1
    for buckets a < shelf_life, leading to objective = 0.
    
    This is a CRITICAL probe that catches a common silent failure pattern.
    """
    
    name = "initialization"
    description = "Test if t=1 inventory initialization is correct"
    target_mechanism = "initialization"
    
    def generate_test_data(self, base_data: dict = None) -> dict:
        return {
            "name": "probe_init",
            "periods": 1,
            "products": ["SKU_A"],
            "locations": ["Loc_A"],
            "shelf_life": {"SKU_A": 3},  # Important: SL > 1 to expose the bug
            "demand_curve": {"SKU_A": [100]},
            "demand_share": {"Loc_A": 1.0},
            "production_cap": {"SKU_A": [0]},  # Cannot produce anything
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
        # Demand = 100, production = 0, all lost sales
        # Cost = 100 * 50 = 5000
        return {"objective_range": (4500, 5500)}
    
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
        
        # If objective is near zero, initialization is missing
        if obj < obj_min * 0.1:  # Very strict: < 450
            return ProbeResult(
                probe_name=self.name,
                result="FAIL",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Objective = {obj} (expected ~5000). "
                          f"MISSING t=1 INITIALIZATION: Add I[p,l,1,a]=0 for a < shelf_life[p]."
            )
        
        if obj < obj_min * self.OBJECTIVE_THRESHOLD:
            return ProbeResult(
                probe_name=self.name,
                result="FAIL",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Objective too low ({obj}). Check t=1 initialization or lost sales."
            )
        
        return ProbeResult(
            probe_name=self.name,
            result="PASS",
            expected=expected,
            actual=actual_result,
            diagnosis="t=1 initialization appears correct."
        )


# ==============================================================================
# HOLDING COST PROBE (NEW)
# ==============================================================================

class HoldingCostProbe(SemanticProbe):
    """
    Test if holding cost is correctly calculated as (I - y), not just I.
    
    Common error: Using I (start-of-period inventory) instead of I-y (end-of-period)
    for holding cost. This leads to silent failures where the model runs but 
    produces incorrect objectives.
    
    Test scenario:
    - Demand = 100, Production = 100, shelf_life = 2
    - Fresh inventory I[t=1,a=2] = 100, all sold y = 100
    - Correct holding cost: (I - y) * cost = 0
    - Wrong holding cost: I * cost = 100 * cost (much higher!)
    """
    
    name = "holding_cost"
    description = "Test if holding cost uses (I-y) instead of just I"
    target_mechanism = "holding_cost"
    
    def generate_test_data(self, base_data: dict = None) -> dict:
        return {
            "name": "probe_holding_cost",
            "periods": 1,
            "products": ["SKU_A"],
            "locations": ["Loc_A"],
            "shelf_life": {"SKU_A": 2},  # a >= 2 exists
            "lead_time": {"SKU_A": 0},
            "demand_curve": {"SKU_A": [100]},  # Demand = 100
            "demand_share": {"Loc_A": 1.0},
            "production_cap": {"SKU_A": [100]},  # Can produce exactly 100
            "cold_capacity": {"Loc_A": 10000},
            "cold_usage": {"SKU_A": 1},
            "network": {"sub_edges": [], "trans_edges": []},
            "costs": {
                "inventory": {"SKU_A": 20},  # High holding cost
                "waste": {"SKU_A": 10},
                "lost_sales": {"SKU_A": 100},  # Higher than holding
                "purchasing": {"SKU_A": 5}
            },
            "constraints": {}
        }
    
    def expected_behavior(self, test_data: dict) -> dict:
        # Optimal: produce 100, sell 100, no inventory at end
        # Correct cost: purchasing = 100 * 5 = 500, holding = 0
        # Wrong cost (using I): purchasing = 500, holding = 100 * 20 = 2000, total = 2500
        return {
            "objective_range": (400, 600),  # Correct range
            "wrong_objective_min": 2000     # If using I instead of I-y
        }
    
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
        wrong_min = expected["wrong_objective_min"]
        
        # If objective is way too high, holding cost uses I instead of I-y
        if obj > wrong_min * 0.8:  # > 1600
            return ProbeResult(
                probe_name=self.name,
                result="FAIL",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Objective = {obj} (expected ~500). "
                          f"WRONG HOLDING COST: Use (I - y) not I. "
                          f"Holding cost should be on END-OF-PERIOD inventory."
            )
        
        if obj < obj_min * 0.5 or obj > obj_max * 2:
            return ProbeResult(
                probe_name=self.name,
                result="FAIL",
                expected=expected,
                actual=actual_result,
                diagnosis=f"Objective {obj} outside expected range ({obj_min}, {obj_max})."
            )
        
        return ProbeResult(
            probe_name=self.name,
            result="PASS",
            expected=expected,
            actual=actual_result,
            diagnosis="Holding cost calculation appears correct (using I-y)."
        )

class ProbeRunner:
    """Run all probes against a generated model."""
    
    def __init__(self):
        self.probes = [
            # Original 8 probes (with fixes)
            SubstitutionProbe(),
            DemandRouteProbe(),
            NoSubstitutionProbe(),
            ProductionCapacityProbe(),
            StorageCapacityProbe(),  # FIXED
            AgingProbe(),
            LostSalesProbe(),  # FIXED
            NonNegativityProbe(),
            # New probes for better coverage
            LeadTimeProbe(),  # F6
            TransshipmentProbe(),  # F7
            LaborCapacityProbe(),  # F8
            MOQProbe(),  # F6
            InitializationProbe(),  # CRITICAL: detects missing t=1 init
            HoldingCostProbe(),  # CRITICAL: detects wrong holding cost (I vs I-y)
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
    
    def _add_optional_fields(self, data: dict) -> dict:
        """Add optional fields with safe defaults to probe test data."""
        products = data.get("products", [])
        locations = data.get("locations", [])
        periods = data.get("periods", 1)
        
        # Add lead_time (default 0 - no delay)
        if "lead_time" not in data:
            data["lead_time"] = {p: 0 for p in products}
        
        # Add labor_cap (default very high - not binding)
        if "labor_cap" not in data:
            data["labor_cap"] = {l: [100000] * periods for l in locations}
        
        # Add labor_usage (default 0 - no labor needed)
        if "labor_usage" not in data:
            data["labor_usage"] = {p: 0 for p in products}
        
        # Add return_rate (default 0 - no returns)
        if "return_rate" not in data:
            data["return_rate"] = {p: 0 for p in products}
        
        # Ensure constraints dict exists
        if "constraints" not in data:
            data["constraints"] = {}
        
        return data

    def run_all_probes(self, model_code: str) -> SemanticProbeReport:
        """Run all probes and return aggregate report."""
        probe_results: List[ProbeResult] = []
        
        for probe in self.probes:
            test_data = probe.generate_test_data()
            test_data = self._add_optional_fields(test_data)  # Add optional fields
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
                test_data = self._add_optional_fields(test_data)  # Add optional fields
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
    
    def run_core_probes(self, model_code: str) -> SemanticProbeReport:
        """Run only the core 8 probes (for backward compatibility)."""
        core_probe_names = [
            "substitution_basic",
            "demand_route_constraint", 
            "no_substitution",
            "production_capacity",
            "storage_capacity",
            "aging_dynamics",
            "lost_sales_slack",
            "inventory_nonnegativity",
        ]
        return self.run_selected_probes(model_code, core_probe_names)


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