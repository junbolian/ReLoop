import os
import math
import pytest

from SelfEvolvedReLoop.controller import Controller
from SelfEvolvedReLoop.state import ExtractionConfig

requires_env = pytest.mark.skipif(
    not (os.environ.get("OR_LLM_BASE_URL") and os.environ.get("OR_LLM_API_KEY")),
    reason="Real LLM creds not set",
)


@requires_env
def test_real_llm_allocation_and_assignment():
    controller = Controller(runtime_kwargs={"budget_override": 2000})
    config = ExtractionConfig(use_llm=True, suggest_candidate=False)

    # allocation
    rec1 = controller.run(
        "allocation problem: {\"items\":[{\"name\":\"a\",\"cost\":1,\"features\":{\"cal\":5}}],\"requirements\":{\"cal\":5}}",
        extraction_config=config,
    )
    assert rec1.audit_result.passed
    assert rec1.solve_result.objective is not None and math.isfinite(rec1.solve_result.objective)
    assert "extract" in rec1.prompt_injections
    assert "model_plan" in rec1.prompt_injections

    # assignment
    rec2 = controller.run(
        "assignment problem: {\"workers\":[\"w1\",\"w2\"],\"tasks\":[\"t1\",\"t2\"],\"costs\":{\"w1\":{\"t1\":1,\"t2\":5},\"w2\":{\"t1\":4,\"t2\":1}}}",
        extraction_config=config,
    )
    assert rec2.audit_result.passed
    assert rec2.solve_result.objective is not None and math.isfinite(rec2.solve_result.objective)
    assert "extract" in rec2.prompt_injections
    assert "model_plan" in rec2.prompt_injections

    for text in [rec1.model_dump_json(), rec2.model_dump_json()]:
        assert "sk-" not in text
        assert "Bearer" not in text
