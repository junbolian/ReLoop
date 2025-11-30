import os
import json
from pathlib import Path

import yaml  # pip install pyyaml


# Path settings: assume this script is inside reloop/tools/
# Project root is one level above this file (reloop/)
ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "scenarios" / "retail_comprehensive"
DATA_DIR = SCENARIO_DIR / "data"
SPEC_DIR = SCENARIO_DIR / "spec"
PROMPT_DIR = SCENARIO_DIR / "prompts"

ARCHETYPE_META_FILE = SPEC_DIR / "archetypes.yaml"


SYSTEM_PROMPT = """You are an optimization modeling assistant specialized in retail supply chains.

Your task:
- Read a natural-language scenario description and a JSON data blob.
- Infer the correct mathematical optimization model (MILP) that matches the business logic.
- Implement that model as Python code using the Gurobi solver (gurobipy).
- Do NOT change the JSON data. Treat it as given inputs.

Requirements:
- Define all sets and parameters using the JSON fields.
- Define decision variables with clear, concise names.
- Add constraints that match the scenario description and the implied structure.
- Set an objective that minimizes total cost, including holding cost, lost-sales penalties,
  waste, ordering cost, and other costs implied by the JSON.
- At the end of the script, build the model, call the solver, and print the objective value
  and basic summaries of key decisions.

Return:
- A single Python script as plain text (no Markdown formatting, no code fences).
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
   shelf life, lead times, substitution edges, transshipment edges, and other flags),
4) Prints the solver status and the optimal objective value.

Do not invent extra data. Do not change any numbers from the JSON.
Return ONLY the Python source code as plain text, with no comments and no Markdown.
""".strip()


def load_archetype_meta(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    return meta


def main():
    if not ARCHETYPE_META_FILE.exists():
        print(f"[ERROR] archetypes.yaml not found at {ARCHETYPE_META_FILE}")
        return

    arche_meta = load_archetype_meta(ARCHETYPE_META_FILE)
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.exists():
        print(f"[ERROR] Data directory not found: {DATA_DIR}")
        return

    json_files = sorted(DATA_DIR.glob("*.json"))

    if not json_files:
        print(f"[WARN] No JSON files found in {DATA_DIR}")
        return

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        scenario_id = data.get("scenario_id") or jf.stem

        if data.get("archetype"):
            archetype_id = data["archetype"]
        else:
            if "_v" in scenario_id:
                archetype_id = scenario_id.rsplit("_v", 1)[0]
            else:
                archetype_id = scenario_id

        if archetype_id not in arche_meta:
            print(f"[WARN] Archetype {archetype_id} not found in archetypes.yaml for {scenario_id}")
            continue

        meta = arche_meta[archetype_id]
        family_id = meta["family_id"]
        family_name = meta["family_name"]
        description = meta["description"].strip()

        user_prompt = USER_TEMPLATE.format(
            family_id=family_id,
            family_name=family_name,
            archetype_id=archetype_id,
            scenario_id=scenario_id,
            description=description,
        )

        out_path = PROMPT_DIR / f"{scenario_id}.txt"
        with open(out_path, "w", encoding="utf-8") as f_out:
            f_out.write("### SYSTEM PROMPT ###\n")
            f_out.write(SYSTEM_PROMPT)
            f_out.write("\n\n### USER PROMPT ###\n")
            f_out.write(user_prompt)

        print(f"[OK] Wrote prompt for {scenario_id} -> {out_path}")


if __name__ == "__main__":
    main()
