"""Unit tests for repair safety guardrails."""

import pytest
from reloop.repair_safety import validate_repair_code, PROTECTED_VARIABLES


class TestValidateRepairCode:
    """Tests for validate_repair_code()."""

    # ---- SHOULD BE DETECTED AS VIOLATIONS ----

    def test_redefine_data_with_dict_literal(self):
        """data = {...} should be detected as a violation."""
        code = '''from gurobipy import Model, GRB
data = {
    "food_items": [{"protein": 14, "cost": 4}],
    "protein_requirement": 120
}
m = Model()
m.Params.OutputFlag = 0
m.optimize()
print(f"status: {m.Status}")
print(f"objective: {m.ObjVal}")
'''
        is_safe, violations = validate_repair_code(code, "", None)
        assert not is_safe
        assert any("Assignment to protected variable 'data'" in v for v in violations)

    def test_redefine_data_single_line(self):
        """data = {"key": value} on a single line should be detected."""
        code = '''from gurobipy import Model, GRB
data = {"protein_requirement": 120, "calorie_requirement": 2089}
m = Model()
m.optimize()
print(f"status: {m.Status}")
print(f"objective: {m.ObjVal}")
'''
        is_safe, violations = validate_repair_code(code, "", None)
        assert not is_safe
        assert len(violations) >= 1

    def test_modify_data_subscript(self):
        """data["key"] = value should be detected as a violation."""
        code = '''from gurobipy import Model, GRB
data["protein_requirement"] = 120
m = Model()
m.Params.OutputFlag = 0
n = len(data["food_items"])
x = [m.addVar(lb=0) for i in range(n)]
m.optimize()
print(f"status: {m.Status}")
print(f"objective: {m.ObjVal}")
'''
        is_safe, violations = validate_repair_code(code, "", None)
        assert not is_safe
        assert any("Modification of protected variable" in v
                    or "Modifies data variable contents" in v
                    for v in violations)

    def test_augmented_assign_data(self):
        """data += {...} should be detected."""
        code = '''from gurobipy import Model, GRB
data += {"extra_key": 42}
m = Model()
m.optimize()
print(f"status: {m.Status}")
print(f"objective: {m.ObjVal}")
'''
        is_safe, violations = validate_repair_code(code, "", None)
        assert not is_safe
        assert any("Augmented assignment" in v for v in violations)

    def test_import_os(self):
        """import os should be detected."""
        code = '''import os
from gurobipy import Model, GRB
m = Model()
m.optimize()
print(f"status: {m.Status}")
print(f"objective: {m.ObjVal}")
'''
        is_safe, violations = validate_repair_code(code, "", None)
        assert not is_safe
        assert any("os" in v for v in violations)

    def test_import_subprocess(self):
        """import subprocess should be detected."""
        code = '''import subprocess
from gurobipy import Model, GRB
m = Model()
m.optimize()
print(f"status: {m.Status}")
print(f"objective: {m.ObjVal}")
'''
        is_safe, violations = validate_repair_code(code, "", None)
        assert not is_safe
        assert any("subprocess" in v for v in violations)

    def test_from_os_import(self):
        """from os import ... should be detected."""
        code = '''from os import path
from gurobipy import Model, GRB
m = Model()
m.optimize()
print(f"status: {m.Status}")
print(f"objective: {m.ObjVal}")
'''
        is_safe, violations = validate_repair_code(code, "", None)
        assert not is_safe
        assert any("os" in v for v in violations)

    def test_data_json_loads(self):
        """data = json.loads(...) should be detected."""
        code = '''import json
from gurobipy import Model, GRB
data = json.loads('{"key": 1}')
m = Model()
m.optimize()
print(f"status: {m.Status}")
print(f"objective: {m.ObjVal}")
'''
        is_safe, violations = validate_repair_code(code, "", None)
        assert not is_safe
        assert len(violations) >= 1

    # ---- SHOULD NOT BE DETECTED (FALSE POSITIVES) ----

    def test_read_data_subscript_not_flagged(self):
        """x = data["capacity"] is a READ — must not be flagged."""
        code = '''from gurobipy import Model, GRB
m = Model()
m.Params.OutputFlag = 0
n = len(data["food_items"])
x = [m.addVar(lb=0, name=f"x_{i}") for i in range(n)]
protein_req = data["protein_requirement"]
m.addConstr(
    sum(data["food_items"][i]["protein"] * x[i] for i in range(n)) >= protein_req,
    name="protein"
)
m.setObjective(
    sum(data["food_items"][i]["cost"] * x[i] for i in range(n)),
    GRB.MINIMIZE
)
m.optimize()
print(f"status: {m.Status}")
print(f"objective: {m.ObjVal}")
'''
        is_safe, violations = validate_repair_code(code, "", None)
        assert is_safe, f"False positive! Violations: {violations}"

    def test_nested_data_read_not_flagged(self):
        """data["food_items"][i]["protein"] is a READ — must not be flagged."""
        code = '''from gurobipy import Model, GRB
m = Model()
m.Params.OutputFlag = 0
items = data["food_items"]
for i, item in enumerate(items):
    cost = item["cost"]
    protein = data["food_items"][i]["protein"]
m.optimize()
print(f"status: {m.Status}")
print(f"objective: {m.ObjVal}")
'''
        is_safe, violations = validate_repair_code(code, "", None)
        assert is_safe, f"False positive! Violations: {violations}"

    def test_normal_model_code_passes(self):
        """Normal optimization code with no data modification passes."""
        code = '''from gurobipy import Model, GRB

m = Model()
m.Params.OutputFlag = 0

n = len(data["food_items"])
x = [m.addVar(lb=0, name=f"x_{i}") for i in range(n)]

m.addConstr(
    sum(data["food_items"][i]["protein"] * x[i] for i in range(n))
    >= data["protein_requirement"],
    name="protein"
)
m.addConstr(
    sum(data["food_items"][i]["carbohydrates"] * x[i] for i in range(n))
    >= data["carbohydrate_requirement"],
    name="carbohydrates"
)
m.addConstr(
    sum(data["food_items"][i]["calories"] * x[i] for i in range(n))
    >= data["calorie_requirement"],
    name="calories"
)

m.setObjective(
    sum(data["food_items"][i]["cost"] * x[i] for i in range(n)),
    GRB.MINIMIZE
)

m.optimize()
print(f"status: {m.Status}")
print(f"objective: {m.ObjVal}")
'''
        is_safe, violations = validate_repair_code(code, "", None)
        assert is_safe, f"False positive! Violations: {violations}"

    def test_local_variable_named_data_in_loop(self):
        """A local var assigned inside a for loop shouldn't trigger if not 'data'."""
        code = '''from gurobipy import Model, GRB
m = Model()
m.Params.OutputFlag = 0
result = {}
for key in data:
    result[key] = data[key]
m.optimize()
print(f"status: {m.Status}")
print(f"objective: {m.ObjVal}")
'''
        is_safe, violations = validate_repair_code(code, "", None)
        assert is_safe, f"False positive! Violations: {violations}"

    def test_syntax_error_code_passes(self):
        """Code with syntax errors should pass safety check (L1 catches it)."""
        code = '''from gurobipy import Model, GRB
m = Model(
x = [m.addVar( for i in range(5)]
'''
        is_safe, violations = validate_repair_code(code, "", None)
        # Regex might catch something, but AST should not crash
        # The key is that it doesn't raise an exception
        assert isinstance(is_safe, bool)

    def test_data_variable_in_comment_not_flagged(self):
        """Comments mentioning data = {...} should not be flagged."""
        code = '''from gurobipy import Model, GRB
# The data variable contains: data = {"protein_requirement": 83}
# Do not redefine data = {...}
m = Model()
m.Params.OutputFlag = 0
n = len(data["food_items"])
m.optimize()
print(f"status: {m.Status}")
print(f"objective: {m.ObjVal}")
'''
        is_safe, violations = validate_repair_code(code, "", None)
        # Comments start with # so the regex pattern ^\s*data won't match
        assert is_safe, f"False positive on comments! Violations: {violations}"

    # ---- DEDUPLICATION ----

    def test_violations_deduplicated(self):
        """Same violation from regex + AST should be deduplicated."""
        code = '''from gurobipy import Model, GRB
data = {"key": 1}
m = Model()
m.optimize()
print(f"status: {m.Status}")
print(f"objective: {m.ObjVal}")
'''
        is_safe, violations = validate_repair_code(code, "", None)
        assert not is_safe
        # Should have violations but no exact duplicates
        assert len(violations) == len(set(violations))

    # ---- EDGE CASES ----

    def test_empty_code(self):
        """Empty code should pass safety check."""
        is_safe, violations = validate_repair_code("", "", None)
        assert is_safe
        assert violations == []

    def test_protected_variables_set(self):
        """Verify expected protected variables are in the set."""
        assert 'data' in PROTECTED_VARIABLES
        assert 'input_data' in PROTECTED_VARIABLES
        assert 'problem_data' in PROTECTED_VARIABLES
