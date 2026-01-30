from pathlib import Path
import json

from SelfEvolvedReLoop.eval_min.runner import run_nl4opt_eval


def test_eval_min_offline(tmp_path):
    dataset = Path("SelfEvolvedReLoop/samples/nl4opt_3.jsonl")
    out_dir = tmp_path / "eval_out"
    result = run_nl4opt_eval(dataset, llm=False, tolerance=1e-6, max_examples=3, out_dir=out_dir)
    summary = result["summary"]
    assert summary["n"] == 3
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "results.jsonl").exists()
    assert (out_dir / "report.md").exists()
    rows = [json.loads(line) for line in (out_dir / "results.jsonl").read_text().splitlines() if line]
    assert len(rows) == 3
    assert any(r.get("objective") is not None for r in rows)
