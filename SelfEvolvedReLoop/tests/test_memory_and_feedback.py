from pathlib import Path

from SelfEvolvedReLoop.skills.fixes import FixMemoryStoreSkill, FixMemoryRetrieveSkill, promote_candidate_to_trusted
from SelfEvolvedReLoop.state import FixRecord, RunRecord, ExtractionConfig
from SelfEvolvedReLoop.utils import memory_paths, load_jsonl


def _clear_memory():
    paths = memory_paths()
    for key in ["candidates", "trusted", "negative"]:
        p = Path(paths[key])
        if p.exists():
            p.unlink()
    run_dir = Path(paths["runs_dir"])
    if run_dir.exists():
        for f in run_dir.glob("*.json"):
            f.unlink()


def test_memory_store_and_retrieve():
    _clear_memory()
    store = FixMemoryStoreSkill()
    fix = FixRecord(
        fix_id="fix1",
        archetype_id="diet_lp",
        symptom_tags=["AUDIT_VIOLATION"],
        patch_type="prompt_patch",
        patch_payload={"prompt_version": "v2_strict"},
        preconditions=["LLM_available"],
        expected_effect="Improve extraction",
    )
    store.append_trusted(fix)
    retrieve = FixMemoryRetrieveSkill()
    res = retrieve.run(archetype_id="diet_lp", symptom_tags=["AUDIT_VIOLATION"])
    assert len(res.trusted_fixes) == 1
    assert res.trusted_fixes[0].fix_id == "fix1"


def test_feedback_promotion_flow():
    _clear_memory()
    store = FixMemoryStoreSkill()
    candidate_fix = FixRecord(
        fix_id="cand1",
        archetype_id="diet_lp",
        symptom_tags=["nutrient_shortfall"],
        patch_type="prompt_patch",
        patch_payload={"prompt_version": "v2_strict"},
        preconditions=[],
        expected_effect="Improve extraction",
    )
    record = RunRecord(
        run_id="run-test",
        input_text="scenario",
        archetype_id="diet_lp",
        extraction_config=ExtractionConfig(),
        candidate_fix=candidate_fix,
    )
    store.store_run(record)
    promoted = promote_candidate_to_trusted(run_id="run-test", label="correct", notes="looks good")
    assert promoted is not None
    trusted_entries = load_jsonl(Path(memory_paths()["trusted"]))
    assert len(trusted_entries) == 1
    assert trusted_entries[0]["human_label"] == "correct"

    promoted_wrong = promote_candidate_to_trusted(run_id="run-test", label="wrong")
    negative_entries = load_jsonl(Path(memory_paths()["negative"]))
    assert len(negative_entries) == 1
    assert negative_entries[0]["human_label"] == "wrong"

    promoted_skip = promote_candidate_to_trusted(run_id="run-test", label="skip")
    candidate_entries = load_jsonl(Path(memory_paths()["candidates"]))
    assert len(candidate_entries) == 1
    assert candidate_entries[0]["human_label"] == "skip"
