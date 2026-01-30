import pytest

try:
    from reloop.agents.schemas import (
        ConstraintTemplate,
        DataFieldProfile,
        DataProfile,
        EquationTemplate,
        SoftViolation,
        SpecSheet,
        Step0Contract,
    )
    from reloop.agents.tools.data_profiler import profile_data
    from reloop.agents.tools.sanity_checker import run_sanity_checks
    from reloop.agents.tools.static_auditor import audit_script
except ImportError:  # pragma: no cover - optional dependency absent
    pytest.skip("reloop package not installed; skipping legacy tests", allow_module_level=True)


def test_data_profiler_handles_nested_shapes_without_values():
    data = {
        "periods": 4,
        "products": ["A", "B"],
        "shelf_life": {"A": 7, "B": 5},
        "demand": {"A": {"L1": [1, 2]}, "B": {"L1": [3, 4]}},
    }
    profile = profile_data(data)
    paths = {f.path for f in profile.fields}
    assert "<root>" in paths
    assert "products" in paths
    assert "shelf_life" in paths
    dump = profile.model_dump()
    assert "7" not in str(dump)  # numeric values are not leaked


def _empty_spec(decisions):
    return SpecSheet(
        sets=[],
        decisions=decisions,
        objective_terms=[],
        constraint_families=[],
        edge_cases=[],
        open_questions=[],
    )


def test_sanity_checker_flags_missing_lost_sales_slack():
    contract = Step0Contract(
        optimize="minimize cost",
        controls=[],
        hard_constraints=[],
        soft_violations=[SoftViolation(name="lost sales", penalty_source="costs.lost_sales")],
        contract_summary="",
    )
    spec = _empty_spec([])
    templates = [
        ConstraintTemplate(
            prefix="demand_route",
            template_type="NETWORK",
            applies_when="always",
            indices=[],
            equations=[
                EquationTemplate(
                    name_suffix="p_l_t",
                    sense="=",
                    lhs="d[p,l,t]",
                    rhs="demand[p,l,t]",
                )
            ],
            notes=[],
        )
    ]
    report = run_sanity_checks(contract, spec, templates, data_profile=None)
    assert report.overall_pass is False
    assert "demand_route" in report.blockers


def test_sanity_checker_flags_missing_shelf_life_indexing():
    contract = Step0Contract(
        optimize="minimize cost",
        controls=[],
        hard_constraints=[],
        soft_violations=[],
        contract_summary="",
    )
    spec = _empty_spec([])
    data_profile = DataProfile(
        summary="test",
        fields=[
            DataFieldProfile(
                path="shelf_life", kind="dict", type="dict", key_types="str"
            )
        ],
    )
    report = run_sanity_checks(contract, spec, [], data_profile=data_profile)
    failing = [c for c in report.checks if c.name == "unit_consistency"][0]
    assert failing.passed is False


def test_sanity_checker_flags_missing_substitution_coupling():
    contract = Step0Contract(
        optimize="minimize cost",
        controls=[],
        hard_constraints=[],
        soft_violations=[],
        contract_summary="",
    )
    spec = _empty_spec([])
    data_profile = DataProfile(
        summary="sub test",
        fields=[
            DataFieldProfile(
                path="network.sub_edges", kind="list", type="list", element_type="list"
            )
        ],
    )
    templates = [
        ConstraintTemplate(
            prefix="demand_route",
            template_type="NETWORK",
            applies_when="substitution",
            indices=[],
            equations=[
                EquationTemplate(
                    name_suffix="t", sense="=", lhs="d[p,l,t]", rhs="demand[p,l,t]"
                )
            ],
            notes=[],
        )
    ]
    report = run_sanity_checks(contract, spec, templates, data_profile=data_profile)
    failing = [c for c in report.checks if c.name == "conservation_closed"][0]
    assert failing.passed is False


def test_static_auditor_blocks_forbidden_io_and_missing_status():
    bad_code = """
import gurobipy as gp
from gurobipy import GRB
data={}
with open("file.txt") as f:
    f.read()
m = gp.Model()
m.Params.OutputFlag = 1
m.optimize()
"""
    audit = audit_script(bad_code)
    assert audit.passed is False
    assert any("OutputFlag" in f for f in audit.failures)
    assert any("Forbidden" in f for f in audit.failures)
