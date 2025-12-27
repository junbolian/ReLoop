<<<<<<< HEAD
import argparse
=======
>>>>>>> 1980535 (Fix production_cap semantics in RetailOpt-190 system prompt (cap on Q))
import json
from pathlib import Path

import yaml  # pip install pyyaml


# Path settings: assume this script is inside reloop/tools/
# ROOT points to reloop/
ROOT = Path(__file__).resolve().parents[1]


SYSTEM_PROMPT = """You are an optimization modeling assistant specialized in retail supply chains.

Goal: semantic fidelity. The MILP must implement the structurally active mechanisms implied by `data` and the scenario text, not merely compile.

Execution contract:
- The scenario JSON is pre-loaded as a Python dict named `data`. Do NOT modify `data`.
- Do NOT perform any file I/O (no open(), json.load(), Path.read_text(), etc.).

Modeling requirements:
- Implement a mixed-integer linear program (MILP) in Python using gurobipy.
- Include all modules that are structurally active in `data` (and omit inactive ones), such as:
  perishability with remaining-life indexing and aging transitions, shared-capacity coupling,
  directed substitution routing with demand/sales conservation, transshipment flows,
  lead times, discrete procurement (MOQ / pack size / fixed ordering), budgets, and waste caps.

Data semantics (must follow):
- Demand allocation: if `data` contains `demand_curve` by product and period and `demand_share` by location, interpret `demand_curve[p][t]` as total demand for product p in period t, and allocate location demand as `Dem[p,l,t] = demand_curve[p][t] * demand_share[l]`.
- Production/procurement capacity: if `data` contains `production_cap[p][t]` with no location index, interpret it as a global (network-wide) capacity for product p in period t, enforced on orders/production decisions as `sum_l Q[p,l,t] <= production_cap[p][t]`. Lead times (if any) only shift arrivals across periods; do NOT reinterpret `production_cap` as an arrival/inflow cap.

Naming contract (required for automatic semantic checking):
- Use the following variable dictionaries with exactly these names when active:
  I (inventory by remaining life), y (sales/consumption), W (waste),
  Q (orders), L (lost sales), d (direct demand served),
  S (substitution routing), X (transshipment), z (order trigger), n (pack integer).
- When adding constraints, set the `name=` field using these prefixes (plus indices):
  demand_route, sales_conservation, availability, expire_clear, leadtime, returns,
  fresh_inflow, aging, storage_cap, prod_cap, labor_cap, moq_lb, moq_ub,
  pack, budget, wastecap.

Objective (reference semantics):
- Minimize total cost including all cost terms present in `data`: holding/inventory, waste, and lost-sales penalties;
  plus purchasing/ordering costs if provided; plus (if enabled) transshipment cost and fixed ordering cost.
- If a cost term is missing in `data`, treat it as zero; do not invent extra data.

Solving and output:
- Call the solver and print the solver status and the objective value (if available).

Return:
- Output a single Python script as plain text.
- No Markdown, no code fences, and no comments in the returned code.
""".strip()


USER_TEMPLATE = """[SCENARIO]
Family: {family_id} ({family_name})
Archetype: {archetype_id}
Scenario ID: {scenario_id}

{description}

Operational context:
- The JSON contains the number of time periods, the list of products, and the list of locations
  directly as top-level fields (for example: "periods", "products", "locations").
- Cost parameters such as holding, lost-sales, waste, purchasing, and any fixed ordering costs
  are stored in the "costs" section of the JSON.
- Capacity and operational limits such as storage capacity, production capacity, labor capacity,
  shelf life, lead times, minimum order quantities, pack sizes, and any waste or budget limits
  are stored in fields such as "cold_capacity", "production_cap", "labor_cap", "shelf_life",
  "lead_time", "constraints", and "network".
- Scenario-level control parameters such as global minimum order quantities, pack sizes, fixed
  ordering costs, per-period budgets, and waste caps are provided as scalar fields inside the
  "constraints" and "costs" sections and should be applied uniformly across products and locations
  unless the scenario description explicitly specifies otherwise.
- Substitution and transshipment structures are encoded in the "network" section, for example
  as substitution edges or transshipment edges between locations.
- The model should respect all of these fields exactly as given and interpret them in a way
  consistent with the scenario description.

JSON data (do not modify):
The evaluation harness loads the JSON for this scenario into a Python variable
called `data`. Your code should read all sets and parameters from `data` using
these fields and must not change any numeric values or perform any file I/O
(for example, do not call open or json.load).

[INSTRUCTION]
Using ONLY the information above, write a complete Python script that:

1) Imports gurobipy (import gurobipy as gp; from gurobipy import GRB),
2) Assumes the JSON has already been loaded into a Python variable called `data`,
3) Builds and solves a mixed-integer linear program that reflects the business
   description and the structure implied by the JSON fields (including capacities,
   shelf life, lead times, substitution edges, transshipment edges, and other keys/conditions present in `data`),
4) Prints the solver status and the optimal objective value.

Do not invent extra data. Do not change any numbers from the JSON.
Return ONLY the Python source code as plain text, with no comments and no Markdown.
""".strip()


def load_archetype_meta(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}

    # Allow either:
    #  (a) top-level mapping: {archetype_id: {...}}
    #  (b) wrapped mapping: {"archetypes": {archetype_id: {...}}}
    if isinstance(meta, dict) and "archetypes" in meta and isinstance(meta["archetypes"], dict):
        meta = meta["archetypes"]

    if not isinstance(meta, dict):
        raise ValueError(f"Unexpected YAML structure in {path}: expected a mapping.")

    return meta


def infer_archetype_id(scenario_id: str, data: dict):
    if isinstance(data, dict) and data.get("archetype"):
        return str(data["archetype"])

    # Default convention: drop trailing "_v{int}" if present
    if "_v" in scenario_id:
        left, right = scenario_id.rsplit("_v", 1)
        if right.isdigit():
            return left

    return scenario_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario folder name under reloop/scenarios/ (e.g., retailopt_190 or retail_comprehensive). "
             "If omitted, auto-detects retailopt_190 then retail_comprehensive.",
    )
    parser.add_argument(
        "--write_combined",
        action="store_true",
        help="Also write a combined file containing system+user prompt for each scenario.",
    )
    args = parser.parse_args()

    # Auto-detect scenario folder if not provided
    scenario_name = args.scenario
    if scenario_name is None:
        if (ROOT / "scenarios" / "retailopt_190").exists():
            scenario_name = "retailopt_190"
        elif (ROOT / "scenarios" / "retail_comprehensive").exists():
            scenario_name = "retail_comprehensive"
        else:
            print("[ERROR] Could not auto-detect scenario folder. Use --scenario <name>.")
            return

    scenario_dir = ROOT / "scenarios" / scenario_name
    data_dir = scenario_dir / "data"
    spec_dir = scenario_dir / "spec"
    prompt_dir = scenario_dir / "prompts"
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
    prompt_dir.mkdir(parents=True, exist_ok=True)

    # Write system prompt once
    sys_path = prompt_dir / "system_prompt.txt"
    with open(sys_path, "w", encoding="utf-8") as f_sys:
        f_sys.write(SYSTEM_PROMPT)
        f_sys.write("\n")
    print(f"[OK] Wrote system prompt -> {sys_path}")

    ok_count = 0
    skip_count = 0

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

        user_prompt = USER_TEMPLATE.format(
            family_id=family_id,
            family_name=family_name,
            archetype_id=archetype_id,
            scenario_id=scenario_id,
            description=description,
        )

        out_user = prompt_dir / f"{scenario_id}.user.txt"
        with open(out_user, "w", encoding="utf-8") as f_out:
            f_out.write(user_prompt)
            f_out.write("\n")

        if args.write_combined:
            out_combined = prompt_dir / f"{scenario_id}.txt"
            with open(out_combined, "w", encoding="utf-8") as f_out:
                f_out.write(SYSTEM_PROMPT)
                f_out.write("\n\n")
                f_out.write(user_prompt)
                f_out.write("\n")

        print(f"[OK] Wrote user prompt for {scenario_id} -> {out_user}")
        ok_count += 1

    print(f"Done. prompts_ok={ok_count}, skipped={skip_count}, scenario={scenario_name}")


if __name__ == "__main__":
    main()
