# ReLoop Agent Runner

This folder adds an agent-driven solver loop for RetailOpt-190 with repair feedback, IIS-based hints, and reproducible artifacts for fine-tuning.

## Setup
1) Python ≥ 3.9, Gurobi with a valid license.
2) Install deps: `pip install -r requirements.txt` (include `requests`, `pytest` if missing).
3) LLM:
   - **Mock (offline, default)**: no key needed.
   - **Real**: set `OPENAI_API_KEY`, `OPENAI_MODEL`, optional `OPENAI_BASE_URL`.
   - Optional cost estimate: `export RELOOP_COST_PER_1K=0.01` (USD per 1k tokens).

## Single run (LangChain tool-calling agent)
```bash
# Qwen example (OpenAI-compatible)
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
OPENAI_MODEL=qwen-plus-latest \
OPENAI_API_KEY=sk-xxx \
python -m reloop.agents.run_one --scenario retail_f1_52_weeks_v0 --max_iters 6
```

## Batch run
```bash
python -m reloop.agents.run_benchmark --limit 3 --parallel 1
```

Results land under `runs/<scenario>/<timestamp>/` with `messages.jsonl` (tool calls), `training_trace.jsonl`, and `summary.json`. Aggregate CSV: `eval/agent_results.csv`.

## Notes
- Default data/prompts: `scenarios/retailopt_190/{data,prompts}`; override via `--data_dir/--prompts_dir`.
- LLM config via env: `OPENAI_API_KEY` (or `DASHSCOPE_API_KEY` for Qwen), `OPENAI_MODEL`, `OPENAI_BASE_URL`.
