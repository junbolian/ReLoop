import pytest

from SelfEvolvedReLoop.controller import Controller
from SelfEvolvedReLoop.state import AllocationSchema, AllocationItem, ExtractionConfig


class FakeExtractor:
    """Deterministic extractor for testing without LLM."""

    def run(self, inp):
        schema = AllocationSchema(
            items=[
                AllocationItem(name="bread", cost=0.5, features={"protein": 4, "calories": 50}),
                AllocationItem(name="milk", cost=0.8, features={"protein": 8, "calories": 40}),
            ],
            requirements={"protein": 10, "calories": 80},
        )

        class Out:
            def __init__(self, schema):
                self.schema = schema
                self.warnings = []
                self.error = None

        return Out(schema)


def test_diet_pipeline_optimal():
    controller = Controller(extractor=FakeExtractor())
    config = ExtractionConfig(use_llm=False, suggest_candidate=False)
    record = controller.run("Simple diet scenario", extraction_config=config)

    assert record.final_status == "success"
    assert record.audit_result and record.audit_result.passed is True
    assert record.solve_result.status.lower() == "optimal"
    assert record.solve_result.objective == pytest.approx(1.1, rel=1e-4)
