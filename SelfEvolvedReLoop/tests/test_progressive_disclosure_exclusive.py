import json
import pytest

from SelfEvolvedReLoop.skill_runtime import SkillRuntime


class DummyLLM:
    def __init__(self, bucket):
        self.bucket = bucket

    def invoke(self, prompt):
        self.bucket.append(prompt)

        class Resp:
            content = json.dumps({})

        return Resp()


def test_progressive_disclosure_exclusive(monkeypatch):
    captured = []

    monkeypatch.setattr(
        "SelfEvolvedReLoop.skill_runtime.get_llm_client",
        lambda *args, **kwargs: DummyLLM(captured),
    )

    runtime = SkillRuntime()
    state = type(
        "S",
        (),
        {
            "input_text": "allocation problem with calories",
            "archetype_id": "allocation_lp",
            "prompt_injections": {},
        },
    )()

    runtime.run_llm_skill("extract", state)

    assert captured, "LLM was not invoked"
    prompt = captured[0]
    assert "Skill Card: allocation_lp" in prompt
    assert "Skill Card: transport_flow" not in prompt

    meta = state.prompt_injections.get("extract")
    assert meta is not None
    paths = [f.get("path", "") for f in meta.get("files", [])]
    assert any("allocation_lp/skill.md" in p for p in paths)
    assert all("transport_flow/skill.md" not in p for p in paths)
