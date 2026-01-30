import json
from pathlib import Path

from SelfEvolvedReLoop.controller import Controller
from SelfEvolvedReLoop.state import ExtractionConfig


def test_run_snapshot_no_secrets():
    controller = Controller(runtime_kwargs={"budget_override": None})
    config = ExtractionConfig(use_llm=False, suggest_candidate=False)
    record = controller.run(
        "allocation test {\"items\":[{\"name\":\"a\",\"cost\":1,\"features\":{\"cal\":5}}],\"requirements\":{\"cal\":4}}",
        extraction_config=config,
    )
    run_path = Path(__file__).resolve().parent.parent / "memory" / "runs"
    path = run_path / f"{record.run_id}.json"
    assert path.exists(), "run snapshot not written"
    text = path.read_text(encoding="utf-8")
    forbidden = ["Bearer", "sk-", "OR_LLM_API_KEY", "OR_LLM_BASE_URL", "OPENAI_API_KEY"]
    for tok in forbidden:
        assert tok not in text
    data = json.loads(text)
    assert "prompt_injections" in data
