import pytest

from SelfEvolvedReLoop.controller import Controller
from SelfEvolvedReLoop.state import ExtractionConfig


def run_controller(text, data=None):
    controller = Controller()
    config = ExtractionConfig(use_llm=False, suggest_candidate=False)
    return controller.run(text, extraction_config=config, input_data=data)


def test_transport_flow_simple():
    scenario = """transport problem
    {
      "sources": [{"name": "S1", "supply": 10}],
      "sinks": [{"name": "D1", "demand": 10}],
      "costs": {"S1": {"D1": 5}}
    }
    """
    record = run_controller(scenario)
    assert record.final_status == "success"
    assert record.solve_result.objective == pytest.approx(50.0, rel=1e-6)
    assert record.audit_result.passed


def test_assignment_simple():
    scenario = """assignment problem
    {
      "workers": ["w1", "w2"],
      "tasks": ["t1", "t2"],
      "costs": {
        "w1": {"t1": 1, "t2": 5},
        "w2": {"t1": 4, "t2": 1}
      }
    }
    """
    record = run_controller(scenario)
    assert record.final_status == "success"
    assert record.audit_result.passed
    assert record.solve_result.objective == pytest.approx(2.0, rel=1e-6)


def test_facility_location_simple():
    scenario = """facility location
    {
      "facilities": [{"name": "F1", "open_cost": 1}, {"name": "F2", "open_cost": 100}],
      "customers": ["C1", "C2"],
      "ship_cost": {"F1": {"C1": 1, "C2": 1}, "F2": {"C1": 50, "C2": 50}},
      "demand": {"C1": 1, "C2": 1}
    }
    """
    record = run_controller(scenario)
    assert record.final_status == "success"
    assert record.audit_result.passed
    assert record.solve_result.objective == pytest.approx(3.0, rel=1e-6)


def test_retail_missing_data_nyi():
    scenario = "retail inventory planning without data"
    record = run_controller(scenario)
    assert record.final_status == "not_yet_implemented"
    assert record.not_implemented is not None
    assert "data_dict" in record.not_implemented.missing_inputs


def test_retail_with_data():
    scenario = "retail inventory planning with data"
    data = {
        "products": ["p1"],
        "locations": ["L1"],
        "periods": ["t1"],
        "demand": {"p1": [5]},
        "production_cap": {"p1": [10]},
        "purchase_cost": {"p1": 2.0},
        "hold_cost": {"p1": 0.0},
        "lost_penalty": {"p1": 10.0},
        "cold_usage": {"p1": 0.0},
        "cold_capacity": {"L1": 100.0},
    }
    record = run_controller(scenario, data=data)
    assert record.final_status == "success"
    assert record.audit_result.passed
    assert record.solve_result.objective == pytest.approx(10.0, rel=1e-6)
