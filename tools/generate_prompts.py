"""
Generate scenario prompt files for LLM evaluation on RetailOpt-190.

This script generates prompts for TWO evaluation modes:

1. Zero-shot Baseline:
   - Uses: {scenario_id}.scenario.txt (includes guardrails)
   - Single LLM call
   - For all baseline models (GPT-4, Claude, Qwen, etc.)

2. ReLoop Agent:
   - Uses: base_prompt + step_prompts (00-07)
   - Multi-step pipeline with probes
   - For Qwen2.5-Coder-14B primary experiments

DESIGN PRINCIPLE:
- Both Baseline and ReLoop receive the SAME semantic information
- Difference is only in GUIDANCE STYLE (one-shot vs multi-step)
- NO code templates given - only semantic descriptions
- Prompt must be UNAMBIGUOUS so a capable LLM can write correct code
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================
# DATA SCHEMA - JSON structure only
# ==============================================================================

DATA_SCHEMA = """
{
  "name": str,                          # scenario identifier
  "periods": int,                       # number of time periods
  "products": [str, ...],               # list of product IDs
  "locations": [str, ...],              # list of location IDs

  "shelf_life": {p: int},               # shelf life in periods per product
  "lead_time": {p: int},                # order lead time per product (0 = same-period arrival)

  "demand_curve": {p: [float, ...]},    # demand per product per period (0-indexed list)
  "demand_share": {l: float},           # fraction of total demand at each location

  "production_cap": {p: [float, ...]},  # max production per product per period (0-indexed list)
  "cold_capacity": {l: float},          # storage capacity per location
  "cold_usage": {p: float},             # storage units per unit of product

  "labor_cap": {l: [float, ...]},       # labor hours per location per period (0-indexed list)
  "labor_usage": {p: float},            # labor hours per unit sold
  "return_rate": {p: float},            # fraction of sales returned next period

  "costs": {
    "purchasing": {p: float},           # cost per unit ordered
    "inventory": {p: float},            # holding cost per unit per period
    "waste": {p: float},                # cost per unit expired
    "lost_sales": {p: float},           # penalty per unit of unmet demand
    "fixed_order": float,               # fixed cost per order placed
    "transshipment": float              # cost per unit transshipped
  },

  "constraints": {
    "moq": float,                       # minimum order quantity (0 = no MOQ)
    "pack_size": int,                   # order must be multiple of this (1 = no constraint)
    "budget_per_period": float|null,    # max purchasing cost per period
    "waste_limit_pct": float|null       # max waste as fraction of total demand
  },

  "network": {
    "sub_edges": [[p_from, p_to], ...], # substitution: p_from's demand can be served by p_to
    "trans_edges": [[l_from, l_to], ...]# transshipment: can ship from l_from to l_to
  }
}
""".strip()

DATA_ACCESS = """
- The variable `data` is pre-loaded. Do NOT use file I/O.
- Network data is nested: use data.get('network', {}).get('sub_edges', [])
- Lists are 0-indexed
""".strip()

OUTPUT_FORMAT = """
- Output ONLY Python code
- Use GurobiPy
- Print status and objective
""".strip()


# ==============================================================================
# SCENARIO TEMPLATE
# ==============================================================================

SCENARIO_TEMPLATE_ZEROSHOT = """[SCENARIO]
Family: {family_id} ({family_name})
Archetype: {archetype_id}
Scenario ID: {scenario_id}

[BUSINESS DESCRIPTION]
{description}

[DATA SCHEMA]
{data_schema}

[DATA ACCESS]
{data_access}

[OUTPUT FORMAT]
{output_format}

[TASK]
Write a GurobiPy script that models and solves this optimization problem.
""".strip()


# Template for ReLoop Agent (scenario description only)
SCENARIO_TEMPLATE_AGENT = """[SCENARIO]
Family: {family_id} ({family_name})
Archetype: {archetype_id}
Scenario ID: {scenario_id}

[BUSINESS DESCRIPTION]
{description}
""".strip()


def load_archetype_meta(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}

    if isinstance(meta, dict) and "archetypes" in meta and isinstance(meta["archetypes"], dict):
        meta = meta["archetypes"]

    if not isinstance(meta, dict):
        raise ValueError(f"Unexpected YAML structure in {path}: expected a mapping.")

    return meta


def infer_archetype_id(scenario_id: str, data: dict):
    if isinstance(data, dict) and data.get("archetype"):
        return str(data["archetype"])

    if "_v" in scenario_id:
        left, right = scenario_id.rsplit("_v", 1)
        if right.isdigit():
            return left

    return scenario_id


def choose_prompt_dir(root: Path, scenario_dir: Path, override: str | None):
    if override:
        return Path(override)

    shared = root / "scenarios" / "prompts"
    if shared.exists():
        return shared

    return scenario_dir / "prompts"




def is_scenario_set_dir(d: Path):
    if not d.is_dir():
        return False
    data_dir = d / "data"
    spec_file = d / "spec" / "archetypes.yaml"
    if not data_dir.exists() or not spec_file.exists():
        return False
    if not any(data_dir.glob("*.json")):
        return False
    return True


def auto_detect_scenario(root: Path):
    scenarios_root = root / "scenarios"
    if not scenarios_root.exists():
        return None, [], "missing_scenarios_root"

    if is_scenario_set_dir(scenarios_root):
        return "__ROOT__", [scenarios_root], "root_is_scenario_set"

    candidates = sorted(
        [p for p in scenarios_root.iterdir() if p.is_dir() and is_scenario_set_dir(p)],
        key=lambda x: x.name,
    )
    if not candidates:
        return None, [], "no_candidates"

    prefer = ["retailopt_190", "retail_comprehensive"]
    for name in prefer:
        for c in candidates:
            if c.name == name:
                return c.name, candidates, "preferred_candidate"

    return candidates[0].name, candidates, "first_candidate"


def main():
    parser = argparse.ArgumentParser(
        description="Generate scenario prompt files for LLM evaluation."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario folder name under reloop/scenarios/.",
    )
    parser.add_argument(
        "--prompts_root",
        type=str,
        default=None,
        help="Optional override output directory for prompts.",
    )
    parser.add_argument(
        "--no_guardrails",
        action="store_true",
        help="Exclude guardrails (for ablation study only).",
    )
    args = parser.parse_args()

    scenario_name = args.scenario
    detected_candidates = []
    detect_mode = None

    if scenario_name is None:
        scenario_name, detected_candidates, detect_mode = auto_detect_scenario(ROOT)
        if scenario_name is None:
            scenarios_root = ROOT / "scenarios"
            print("[ERROR] Could not auto-detect scenario folder. Use --scenario <n>.")
            if scenarios_root.exists():
                dirs = [p.name for p in scenarios_root.iterdir() if p.is_dir()]
                print(f"[HINT] Found subdirectories under {scenarios_root}: {dirs}")
            return
        else:
            if detect_mode == "root_is_scenario_set":
                print("[OK] Auto-detected layout: scenarios/ is the scenario set.")
            else:
                print(f"[OK] Auto-detected scenario folder: {scenario_name}")
                if len(detected_candidates) > 1:
                    print(f"[OK] Candidates: {[c.name for c in detected_candidates]}")

    if scenario_name == "__ROOT__":
        scenario_dir = ROOT / "scenarios"
    else:
        scenario_dir = ROOT / "scenarios" / scenario_name

    data_dir = scenario_dir / "data"
    spec_dir = scenario_dir / "spec"
    archetype_meta_file = spec_dir / "archetypes.yaml"

    if not archetype_meta_file.exists():
        print(f"[ERROR] archetypes.yaml not found at {archetype_meta_file}")
        return

    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        return

    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        print(f"[WARN] No JSON files found in {data_dir}")
        return

    arche_meta = load_archetype_meta(archetype_meta_file)

    prompt_dir = choose_prompt_dir(ROOT, scenario_dir, args.prompts_root)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    skip_count = 0

    # Content for prompts
    data_schema_content = "" if args.no_guardrails else DATA_SCHEMA
    data_access_content = "" if args.no_guardrails else DATA_ACCESS
    output_format_content = "" if args.no_guardrails else OUTPUT_FORMAT

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        scenario_id = str(data.get("scenario_id") or data.get("name") or jf.stem)
        archetype_id = infer_archetype_id(scenario_id, data)

        if archetype_id not in arche_meta:
            print(f"[WARN] Archetype {archetype_id} not found in archetypes.yaml for {scenario_id}")
            skip_count += 1
            continue

        meta = arche_meta[archetype_id]
        family_id = meta.get("family_id", "unknown")
        family_name = meta.get("family_name", "unknown")
        description = str(meta.get("description", "")).strip()

        # Generate Zero-shot prompt
        zeroshot_prompt = SCENARIO_TEMPLATE_ZEROSHOT.format(
            family_id=family_id,
            family_name=family_name,
            archetype_id=archetype_id,
            scenario_id=scenario_id,
            description=description,
            data_schema=data_schema_content,
            data_access=data_access_content,
            output_format=output_format_content,
        )

        out_file_zeroshot = prompt_dir / f"{scenario_id}.scenario.txt"
        with open(out_file_zeroshot, "w", encoding="utf-8") as f_out:
            f_out.write(zeroshot_prompt)
            f_out.write("\n")

        # Generate Agent base prompt (scenario only, no guardrails)
        agent_prompt = SCENARIO_TEMPLATE_AGENT.format(
            family_id=family_id,
            family_name=family_name,
            archetype_id=archetype_id,
            scenario_id=scenario_id,
            description=description,
        )

        out_file_agent = prompt_dir / f"{scenario_id}.base.txt"
        with open(out_file_agent, "w", encoding="utf-8") as f_out:
            f_out.write(agent_prompt)
            f_out.write("\n")

        print(f"[OK] {scenario_id} -> .scenario.txt (zero-shot) + .base.txt (agent)")

        ok_count += 1

    print(f"\nDone. Generated={ok_count}, Skipped={skip_count}")
    print(f"Output directory: {prompt_dir}")
    print("[NOTE] Generated two files per scenario:")
    print("  - {scenario_id}.scenario.txt : Zero-shot baseline (includes guardrails)")
    print("  - {scenario_id}.base.txt     : ReLoop Agent (scenario only, guardrails injected by step_prompts)")


if __name__ == "__main__":
    main()
