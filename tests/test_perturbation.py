"""Tests for reloop.perturbation module."""

import pytest
from reloop.perturbation import (
    extract_perturbable_params,
    perturb_code,
    detect_perturbation_mode,
    should_perturb,
    get_source_code_param_names,
    _match_param,
    _apply_data_perturbation,
    run_perturbation,
)


# ============================================================================
# extract_perturbable_params
# ============================================================================

class TestExtractPerturbableParams:

    def test_simple_assignment(self):
        code = "capacity = 500\ndemand = 300\n"
        params = extract_perturbable_params(code)
        assert any(p['name'] == 'capacity' and p['value'] == 500 for p in params)
        assert any(p['name'] == 'demand' and p['value'] == 300 for p in params)

    def test_list_assignment(self):
        code = "costs = [10, 20, 15]\n"
        params = extract_perturbable_params(code)
        assert len(params) == 3
        assert params[0]['name'] == 'costs[0]'
        assert params[0]['value'] == 10
        assert params[1]['name'] == 'costs[1]'
        assert params[1]['value'] == 20
        assert params[2]['name'] == 'costs[2]'
        assert params[2]['value'] == 15

    def test_dict_assignment(self):
        code = 'demand = {"A": 100, "B": 200}\n'
        params = extract_perturbable_params(code)
        assert any(p['name'] == 'demand["A"]' and p['value'] == 100 for p in params)
        assert any(p['name'] == 'demand["B"]' and p['value'] == 200 for p in params)

    def test_nested_dict(self):
        code = 'data = {"costs": {"A": 10, "B": 20}}\n'
        params = extract_perturbable_params(code)
        assert any(p['name'] == 'data["costs"]["A"]' and p['value'] == 10 for p in params)
        assert any(p['name'] == 'data["costs"]["B"]' and p['value'] == 20 for p in params)

    def test_dict_with_list_values(self):
        code = 'info = {"prices": [100, 200]}\n'
        params = extract_perturbable_params(code)
        assert any(p['value'] == 100 for p in params)
        assert any(p['value'] == 200 for p in params)

    def test_excluded_variables(self):
        code = (
            "capacity = 500\n"
            "tolerance = 1e-6\n"
            "max_iter = 100\n"
            "M = 1000000\n"
            "i = 0\n"
        )
        params = extract_perturbable_params(code)
        names = [p['name'] for p in params]
        assert 'capacity' in names
        assert 'tolerance' not in names
        assert 'max_iter' not in names
        assert 'M' not in names
        assert 'i' not in names

    def test_gurobi_params_excluded(self):
        code = (
            "capacity = 500\n"
            "m.Params.OutputFlag = 0\n"
            "m.Params.TimeLimit = 60\n"
        )
        params = extract_perturbable_params(code)
        assert len(params) == 1
        assert params[0]['name'] == 'capacity'

    def test_syntax_error_returns_empty(self):
        code = "def f(\n"  # incomplete syntax
        params = extract_perturbable_params(code)
        assert params == []

    def test_float_values(self):
        code = "rate = 0.05\nprice = 12.5\n"
        params = extract_perturbable_params(code)
        assert any(p['name'] == 'rate' and p['value'] == 0.05 for p in params)
        assert any(p['name'] == 'price' and p['value'] == 12.5 for p in params)

    def test_small_int_excluded(self):
        """Integers 0, 1, 2 are excluded (likely indices)."""
        code = "x = 0\ny = 1\nz = 2\nbig = 3\n"
        params = extract_perturbable_params(code)
        names = [p['name'] for p in params]
        assert 'x' not in names
        assert 'y' not in names
        assert 'z' not in names
        assert 'big' in names

    def test_negative_values(self):
        """Negative values via UnaryOp are not matched (AST Constant only)."""
        # In Python 3.10+, -500 is ast.Constant(value=-500) if written
        # as a literal, but -500 as expression is UnaryOp(USub, 500).
        # Our extractor handles ast.Constant only, which is fine.
        code = "penalty = -50\n"
        params = extract_perturbable_params(code)
        # In CPython 3.12+, -50 is folded to Constant(-50).
        # In earlier 3.10/3.11, it may be UnaryOp.
        # We accept either outcome.
        if params:
            assert params[0]['value'] == -50


# ============================================================================
# perturb_code
# ============================================================================

class TestPerturbCode:

    def test_simple_int(self):
        code = "capacity = 500\n"
        perturbed = perturb_code(code, 'capacity', 1.2)
        assert '600' in perturbed
        # Should remain int, not float
        assert '600.0' not in perturbed

    def test_simple_float(self):
        code = "rate = 0.05\n"
        perturbed = perturb_code(code, 'rate', 2.0)
        assert '0.1' in perturbed

    def test_list_element(self):
        code = "costs = [10, 20, 15]\n"
        perturbed = perturb_code(code, 'costs.0', 1.5)
        # 10 * 1.5 = 15
        assert '15' in perturbed

    def test_dict_value(self):
        code = 'demand = {"A": 100, "B": 200}\n'
        perturbed = perturb_code(code, 'demand.A', 2.0)
        assert '200' in perturbed

    def test_nested_dict_value(self):
        code = 'cfg = {"costs": {"x": 50}}\n'
        perturbed = perturb_code(code, 'cfg.costs.x', 3.0)
        assert '150' in perturbed

    def test_dict_list_value(self):
        code = 'info = {"prices": [100, 200]}\n'
        perturbed = perturb_code(code, 'info.prices.0', 0.5)
        assert '50' in perturbed

    def test_not_found_returns_original(self):
        code = "capacity = 500\n"
        perturbed = perturb_code(code, 'nonexistent', 1.2)
        assert perturbed == code

    def test_preserves_int_type(self):
        code = "capacity = 500\n"
        perturbed = perturb_code(code, 'capacity', 1.2)
        # 500 * 1.2 = 600, should be int
        assert '600' in perturbed
        assert '600.0' not in perturbed


# ============================================================================
# detect_perturbation_mode
# ============================================================================

class TestDetectPerturbationMode:

    def test_data_dict_mode(self):
        code = (
            'capacity = data["capacity"]\n'
            'demand = data.get("demand", 0)\n'
        )
        assert detect_perturbation_mode(code, {"capacity": 500}) == "data_dict"

    def test_source_code_mode_no_data(self):
        code = "capacity = 500\ndemand = 300\n"
        assert detect_perturbation_mode(code, None) == "source_code"

    def test_source_code_mode_hardcoded(self):
        code = (
            "a = 10\nb = 20\nc = 30\nd = 40\ne = 50\nf = 60\n"
            "g = 70\nh_val = 80\n"
        )
        assert detect_perturbation_mode(code, {"x": 1}) == "source_code"

    def test_hybrid_mode(self):
        code = (
            'x = data["x"]\n'
            "a = 10\nb = 20\nc = 30\nd = 40\ne = 50\nf = 60\n"
        )
        assert detect_perturbation_mode(code, {"x": 1}) == "hybrid"


# ============================================================================
# should_perturb
# ============================================================================

class TestShouldPerturb:

    def test_normal_param(self):
        assert should_perturb("capacity", 500) is True

    def test_excluded_name(self):
        assert should_perturb("tolerance", 0.001) is False
        assert should_perturb("M", 1e6) is False
        assert should_perturb("BigM", 1e6) is False

    def test_zero_value(self):
        assert should_perturb("cost", 0) is False

    def test_very_small(self):
        assert should_perturb("eps", 1e-10) is False

    def test_very_large(self):
        assert should_perturb("huge", 1e12) is False

    def test_small_int_indices(self):
        assert should_perturb("x", 0) is False
        assert should_perturb("x", 1) is False
        assert should_perturb("x", 2) is False
        assert should_perturb("x", 3) is True


# ============================================================================
# _match_param
# ============================================================================

class TestMatchParam:

    def test_exact_match(self):
        params = [{'name': 'capacity', 'access_path': 'capacity', 'value': 500}]
        assert _match_param(params, 'capacity') is not None

    def test_fuzzy_match(self):
        params = [{'name': 'production_cap', 'access_path': 'production_cap', 'value': 800}]
        result = _match_param(params, 'production')
        assert result is not None

    def test_no_match(self):
        params = [{'name': 'capacity', 'access_path': 'capacity', 'value': 500}]
        assert _match_param(params, 'totally_different') is None


# ============================================================================
# _apply_data_perturbation
# ============================================================================

class TestApplyDataPerturbation:

    def test_simple_scalar(self):
        data = {"capacity": 500, "demand": 300}
        result = _apply_data_perturbation(data, "capacity", 1.2)
        assert result is True
        assert data["capacity"] == 600  # 500 * 1.2 = 600, int preserved by param_utils

    def test_missing_param(self):
        data = {"capacity": 500}
        result = _apply_data_perturbation(data, "nonexistent", 1.2)
        assert result is False


# ============================================================================
# get_source_code_param_names
# ============================================================================

class TestGetSourceCodeParamNames:

    def test_basic(self):
        code = "capacity = 500\ndemand = 300\n"
        names = get_source_code_param_names(code)
        assert 'capacity' in names
        assert 'demand' in names

    def test_max_cap(self):
        # Generate many params
        lines = [f"p{i} = {i * 10}" for i in range(3, 50)]  # start at 3 to avoid small int filter
        code = "\n".join(lines) + "\n"
        names = get_source_code_param_names(code, max_params=10)
        assert len(names) == 10


# ============================================================================
# run_perturbation (unit-level with mock executor)
# ============================================================================

class TestRunPerturbation:

    def test_source_code_mode(self):
        code = "capacity = 500\n"

        def mock_executor(c, d):
            # Parse capacity from code
            for line in c.split('\n'):
                if line.startswith('capacity'):
                    val = int(line.split('=')[1].strip())
                    return float(val * 2)  # dummy objective = capacity * 2
            return None

        result = run_perturbation(
            code=code,
            data=None,
            param_name="capacity",
            factor=1.2,
            executor_fn=mock_executor,
            mode="source_code",
        )
        # 500 * 1.2 = 600, objective = 600 * 2 = 1200
        assert result == 1200.0

    def test_data_dict_mode_with_change(self):
        code = 'x = data["capacity"]\n'

        call_count = [0]

        def mock_executor(c, d):
            call_count[0] += 1
            if d and "capacity" in d:
                return float(d["capacity"] * 2)
            return 100.0

        result = run_perturbation(
            code=code,
            data={"capacity": 500},
            param_name="capacity",
            factor=1.2,
            executor_fn=mock_executor,
            mode="data_dict",
            baseline_obj=1000.0,
        )
        # 500 * 1.2 = 600, objective = 600 * 2 = 1200
        assert result == 1200.0
