"""
End-to-end tests for perturbation integration with verification.

Tests two scenarios:
1. Hardcode-style code (source_code mode): perturbation mode detection
   and L1 verification without LLM.
2. Data-dict style code (data_dict mode): perturbation mode detection
   and L1 verification without LLM.

Note: L2 behavioral testing (CPT + OPT) requires an LLM client for
candidate extraction, so it is not tested here. The underlying
perturbation utilities are tested in test_perturbation.py.
"""

import pytest
from reloop.verification import ReLoopVerifier, Severity
from reloop.perturbation import detect_perturbation_mode


# ============================================================================
# Hardcode-style LP code (IndustryOR / MAMO style)
# Uses hardcoded numeric values, NOT from data dict.
# Prints output in executor-expected format: status: N, objective: X
# ============================================================================

HARDCODE_LP_CODE = """\
import gurobipy as gp
from gurobipy import GRB

# Hardcoded parameters (not from data dict)
capacity = 500
demand = 300
unit_cost = 10
unit_price = 25
holding_cost = 3
shortage_penalty = 50
labor_cost = 8
transport_cost = 5

m = gp.Model("hardcode_lp")
m.Params.OutputFlag = 0

# Decision variables
x = m.addVar(name="production", lb=0, ub=capacity)
inv = m.addVar(name="inventory", lb=0)
short = m.addVar(name="shortage", lb=0)

# Constraints
m.addConstr(x + inv - short == demand, "balance")
m.addConstr(x <= capacity, "cap")

# Objective: minimize total cost
m.setObjective(
    unit_cost * x + holding_cost * inv + shortage_penalty * short
    + labor_cost * x + transport_cost * x,
    GRB.MINIMIZE
)

m.optimize()

print(f"status: {m.Status}")
if m.Status == GRB.OPTIMAL:
    print(f"objective: {m.ObjVal}")
    sol = {v.VarName: v.X for v in m.getVars()}
    print(f"solution: {sol}")
"""


# ============================================================================
# Data-dict style LP code (RetailOpt-190 style)
# Reads parameters from data dict injected by executor.
# ============================================================================

DATA_DICT_LP_CODE = """\
import gurobipy as gp
from gurobipy import GRB

capacity = data["capacity"]
demand = data["demand"]
unit_cost = data["unit_cost"]
unit_price = data["unit_price"]
holding_cost = data["holding_cost"]
shortage_penalty = data["shortage_penalty"]

m = gp.Model("data_dict_lp")
m.Params.OutputFlag = 0

x = m.addVar(name="production", lb=0, ub=capacity)
inv = m.addVar(name="inventory", lb=0)
short = m.addVar(name="shortage", lb=0)

m.addConstr(x + inv - short == demand, "balance")
m.addConstr(x <= capacity, "cap")

m.setObjective(
    unit_cost * x + holding_cost * inv + shortage_penalty * short,
    GRB.MINIMIZE
)

m.optimize()

print(f"status: {m.Status}")
if m.Status == GRB.OPTIMAL:
    print(f"objective: {m.ObjVal}")
    sol = {v.VarName: v.X for v in m.getVars()}
    print(f"solution: {sol}")
"""

DATA_DICT = {
    "capacity": 500,
    "demand": 300,
    "unit_cost": 10,
    "unit_price": 25,
    "holding_cost": 3,
    "shortage_penalty": 50,
}


# ============================================================================
# Tests
# ============================================================================

class TestHardcodeStyleE2E:
    """Test hardcoded-style code with verification."""

    def test_mode_detection(self):
        """Hardcoded code should be detected as source_code mode."""
        mode = detect_perturbation_mode(HARDCODE_LP_CODE, None)
        assert mode == "source_code"

    def test_mode_detection_with_empty_data(self):
        """Even with empty data dict, should detect source_code mode."""
        mode = detect_perturbation_mode(HARDCODE_LP_CODE, {})
        assert mode == "source_code"

    def test_l1_passes(self):
        """L1 should pass for valid hardcoded code (no LLM needed)."""
        verifier = ReLoopVerifier(delta=0.2, timeout=30)
        report = verifier.verify(HARDCODE_LP_CODE, {}, verbose=True)

        # Should not be FAILED (code is valid)
        assert report.status != 'FAILED', f"L1 failed: {report.recommendations}"

        # L1 results should exist
        l1_results = [r for r in report.layer_results if r.layer == "L1"]
        assert len(l1_results) > 0, "No L1 results"

        # Without LLM, no L2 behavioral testing results
        l2_results = [r for r in report.layer_results if r.layer == "L2"]
        assert len(l2_results) == 0, "L2 results without LLM should be empty"


class TestDataDictStyleE2E:
    """Test that data-dict style code works with verification."""

    def test_mode_detection(self):
        """Data-dict code should be detected as data_dict mode."""
        mode = detect_perturbation_mode(DATA_DICT_LP_CODE, DATA_DICT)
        assert mode == "data_dict"

    def test_l1_passes(self):
        """L1 should pass for valid data-dict style code."""
        verifier = ReLoopVerifier(delta=0.2, timeout=30)
        report = verifier.verify(DATA_DICT_LP_CODE, DATA_DICT, verbose=True)

        assert report.status != 'FAILED', f"L1 failed: {report.recommendations}"

        l1_results = [r for r in report.layer_results if r.layer == "L1"]
        assert len(l1_results) > 0, "No L1 results"

        # Without LLM, no L2 behavioral testing results
        l2_results = [r for r in report.layer_results if r.layer == "L2"]
        assert len(l2_results) == 0, "L2 results without LLM should be empty"


class TestModeDetectionEdgeCases:
    """Test mode detection on various code patterns."""

    def test_no_data_returns_source_code(self):
        assert detect_perturbation_mode("x = 100\n", None) == "source_code"

    def test_data_access_with_few_hardcoded(self):
        code = 'x = data["x"]\ny = 10\n'
        assert detect_perturbation_mode(code, {"x": 1}) == "data_dict"

    def test_many_hardcoded_no_data_access(self):
        lines = [f"p{i} = {i * 100}" for i in range(3, 15)]
        code = "\n".join(lines) + "\n"
        assert detect_perturbation_mode(code, {"x": 1}) == "source_code"
