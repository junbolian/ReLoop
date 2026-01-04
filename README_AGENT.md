# ReLoop Agent Runner

A clean-slate LangGraph orchestrator drives a strict Step0–Step6 workflow for retail MILP codegen (contract → tags → spec sheet → templates → sanity → codegen → audit/run/repair).

## Setup
1) Python ≥ 3.10 and Gurobi with a valid license.
2) `pip install -e .` (uses `pyproject.toml` + `requirements.txt`).
3) LLM options:
- **Mock**: `--mock-llm` (offline).
- **OpenAI-compatible**: set `OPENAI_API_KEY`, `OPENAI_MODEL`, optional `OPENAI_BASE_URL`.

# Set Environmental Variable
```bash
export OPENAI_API_KEY="sk-785..."
export OPENAI_MODEL="qwen-plus"  # or "qwen3-max", "qwen3-max-preview"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1" 
```

## Run one scenario
```bash
python -m reloop.agents.cli.run_one --scenario retail_f1_52_weeks_v0 --out artifacts
# or provide a custom base prompt: --base-prompt path/to/prompt.txt
# optional caps: --repair-limit 5 --max-turns 8
```

## Run a suite
```bash
python -m reloop.agents.cli.run_benchmark --suite suite.txt --out artifacts
```

Artifacts land under `artifacts/<run_id>/` with per-turn messages, step outputs, static audit, runtime stdout/stderr, solver/IIS reports, and code snapshots. Shims keep `python -m reloop.agents.run_one` and `python -m reloop.agents.run_benchmark` working (with deprecation warnings).
