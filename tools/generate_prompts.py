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

Recommended Model: Qwen2.5-Coder-14B-Instruct
"""

import argparse
import json
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================
# GUARDRAILS (included in Zero-shot prompts for fair comparison)
# ==============================================================================

GUARDRAILS = """
[CORE RULES]
- `data` is a pre-loaded Python dict. Do not modify it.
- No file I/O. Never invent missing data.
- Never hard-code numeric values.
- Output must be plain Python code. No prose, no markdown, no comments.

[DATA FORMAT]
- sub_edges: [[p_from, p_to], ...] means p_from's demand can be served by p_to's inventory.
- trans_edges: [[loc_from, loc_to], ...] for transshipment.
- demand_share: {location: scalar}, NOT nested by product.
- demand = demand_curve[p][t-1] * demand_share[l]  (demand_curve is 0-indexed)
- Time indexing: 1-based (t = 1, 2, ..., T).

[SUBSTITUTION SEMANTICS - CRITICAL]
Edge [p_from, p_to] = "upward substitution": p_to can serve p_from's demand.
S[p_from, p_to, l, t] = quantity of p_from's demand served by p_to.

Build edge mappings BEFORE constraints:
  outgoing_edges = {p: [] for p in products}
  incoming_edges = {p: [] for p in products}
  for p_from, p_to in sub_edges:
      outgoing_edges[p_from].append(p_to)  # p_from sends demand OUT to p_to
      incoming_edges[p_to].append(p_from)  # p_to receives requests IN from p_from

Compute substitution flows for each product p:
  outbound = sum S[p, pt, l, t] for pt in outgoing_edges[p]  # demand p sends out
  inbound  = sum S[pf, p, l, t] for pf in incoming_edges[p]  # requests p receives

Substitution constraints:
  demand_route: outbound <= demand[p,l,t]  (can't substitute more than own demand)
  sales_conservation: sum_a(y[p,l,t,a]) + L[p,l,t] = demand[p,l,t] + inbound - outbound

[SHELF-LIFE / AGING]
- Life buckets: a = 1 (expiring) to a = shelf_life[p] (freshest)
- Aging: I[p,l,t+1,a] = I[p,l,t,a+1] - y[p,l,t,a+1]  (for a < shelf_life, t < T)
- Expiration: W[p,l,t] = I[p,l,t,1] - y[p,l,t,1]
- Availability: y[p,l,t,a] <= I[p,l,t,a]
- Holding cost: apply only to a >= 2 (not expiring bucket a=1)

[FRESH INFLOW - BOUNDARY CONDITIONS]
Fresh inventory enters at a = shelf_life[p]:
  if t > lead_time[p]:
      I[p,l,t,shelf_life] = Q[p,l,t-lead_time]
  else:
      I[p,l,t,shelf_life] = 0
NEVER access Q[p,l,0] or negative indices - they don't exist.

[INITIALIZATION at t=1]
  I[p,l,1,a] = 0  for a < shelf_life[p]  (non-fresh buckets empty)
  I[p,l,1,shelf_life] = Q[p,l,1] if lead_time=0, else 0

[AGING BOUNDARY at t=T]
Do NOT add aging constraints for t=T, as they would reference I[p,l,T+1,a] which doesn't exist.

[VARIABLE NAMING]
Use these exact names:
- I[p,l,t,a]: inventory by product, location, period, remaining life bucket
- y[p,l,t,a]: sales/consumption from life bucket
- W[p,l,t]: waste (expired inventory)
- Q[p,l,t]: orders/production
- L[p,l,t]: lost sales
- S[p_from,p_to,l,t]: substitution flow (only if sub_edges nonempty)

[SOLVING]
- Gurobi params: OutputFlag=0, Threads=1, Seed=0.
- Print: print(f"status: {m.Status}")
- If OPTIMAL: print(f"objective: {m.ObjVal}")
""".strip()


# ==============================================================================
# SCENARIO TEMPLATE
# ==============================================================================

SCENARIO_TEMPLATE_ZEROSHOT = """[SCENARIO]
Family: {family_id} ({family_name})
Archetype: {archetype_id}
Scenario ID: {scenario_id}

{description}

=============================================================================
MODELING GUIDELINES
=============================================================================
{guardrails}

=============================================================================
DATA ACCESS
=============================================================================

The evaluation harness loads the JSON into a Python variable called `data`.
Read all parameters from `data`. Do not use file I/O.

Key fields:
- data["periods"]: int (number of time periods)
- data["products"]: list of product IDs
- data["locations"]: list of location IDs
- data["shelf_life"][p]: int, life buckets per product
- data["lead_time"][p]: int, delivery delay (may be 0)
- data["demand_curve"][p]: list (0-indexed, use [t-1] for period t)
- data["demand_share"][l]: scalar share per location
- data["network"]["sub_edges"]: [[p_from, p_to], ...]
- data["network"]["trans_edges"]: [[l_from, l_to], ...]
- data["costs"]["inventory"][p], ["waste"][p], ["lost_sales"][p], ["purchasing"][p]
- data["production_cap"][p]: list or scalar (per period)
- data["cold_capacity"][l], data["cold_usage"][p]

{json_block}

[INSTRUCTION]
Write a complete GurobiPy script that:
1) Imports gurobipy (import gurobipy as gp; from gurobipy import GRB)
2) Reads all parameters from `data` (already loaded)
3) Builds edge mappings for substitution BEFORE creating constraints
4) Creates all decision variables with correct indices
5) Sets objective: minimize inventory + waste + lost_sales + purchasing costs
6) Adds all constraints respecting boundary conditions:
   - Initialization at t=1
   - Aging only for t < T
   - Fresh inflow with lead_time check
   - Substitution if sub_edges nonempty
7) Sets Gurobi params: OutputFlag=0, Threads=1, Seed=0
8) Prints status always; prints objective only if OPTIMAL

Return ONLY Python code. No markdown, no comments, no explanations.
""".strip()


# Template for ReLoop Agent (scenario description only, guardrails injected separately)
SCENARIO_TEMPLATE_AGENT = """[SCENARIO]
Family: {family_id} ({family_name})
Archetype: {archetype_id}
Scenario ID: {scenario_id}

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


def build_json_block(data: dict, indent: int):
    txt = json.dumps(data, ensure_ascii=False, indent=indent)
    return (
        "[JSON_PREVIEW]\n"
        "This JSON preview is for reference only. Do NOT hard-code numbers.\n"
        "Always read from `data` at runtime.\n"
        f"{txt}\n"
        "[/JSON_PREVIEW]"
    )


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
        "--include_json",
        action="store_true",
        help="Include a pretty-printed JSON preview in the prompt.",
    )
    parser.add_argument(
        "--json_indent",
        type=int,
        default=2,
        help="Indent level for JSON preview.",
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

    # Guardrails content
    guardrails_content = "" if args.no_guardrails else GUARDRAILS

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

        json_block = ""
        if args.include_json:
            json_block = build_json_block(data, args.json_indent)

        # Generate Zero-shot prompt (with guardrails)
        zeroshot_prompt = SCENARIO_TEMPLATE_ZEROSHOT.format(
            family_id=family_id,
            family_name=family_name,
            archetype_id=archetype_id,
            scenario_id=scenario_id,
            description=description,
            guardrails=guardrails_content,
            json_block=json_block,
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