# Retail Benchmark – LLM Prompt Templates

> File: `reloop/scenarios/retail_comprehensive/spec/retail_prompts.md`  
> Goal: Document how each JSON instance is converted into a text-based prompt for LLM-driven optimization agents, consistent with the current code in `reloop/tools` and the JSON schema used by `universal_retail_solver.py`.

---

## 1. System Prompt Template

All models evaluated on this benchmark are expected to receive the same **system prompt**.  
It matches the `SYSTEM_PROMPT` string used by the prompt-generation script in `reloop/tools/`.

```text
You are an optimization modeling assistant specialized in retail supply chains.

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
```

### 1.1 Execution context for the system prompt

In the benchmark pipeline, the phrase “JSON data blob” refers to the parsed contents of the scenario file that are exposed to the agent as a Python dictionary named `data`. The raw JSON text is **not** embedded in the prompt itself. Instead, for each scenario the evaluation harness:

1. Reads the corresponding JSON file from disk.
2. Parses it into a Python dictionary.
3. Binds that dictionary to a variable called `data`.
4. Executes the Python script returned by the model in an environment where `data` is already defined.

Agents are therefore expected to **read all parameters from `data` only**, without performing any file I/O (no `open`, `json.load`, etc.) and without modifying `data` in place.

The benchmark harness always sends this system prompt, followed by an instance-specific **user prompt** defined below.

---

## 2. User Prompt Template (Per JSON Instance)

For each JSON instance in `scenarios/retail_comprehensive/data/`, the prompt builder script combines:

- The archetype-level description from `archetypes.yaml` / `retail_spec.md`, and  
- The file name / scenario ID of the specific JSON instance.

The JSON blob itself is **not** pasted into the prompt; instead, the model is told that a Python variable called `data` already contains the parsed JSON, as described in Section 1.1.

The generic **user prompt template** is:

```text
[SCENARIO]
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
- Substitution and transshipment structures are encoded in the "network" section, for example
  as substitution edges or transshipment edges between locations.
- The model should respect all of these fields exactly as given and interpret them in a way
  consistent with the scenario description.

JSON data (do not modify):
The evaluation harness loads the JSON for this scenario into a Python variable
called `data`. Your code should read all sets and parameters from `data` using
these fields and must not change any numeric values or perform any file I/O.

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
```

This text is exactly the `USER_TEMPLATE` implemented in the prompt-generation script, with `{family_id}`, `{family_name}`, `{archetype_id}`, `{scenario_id}`, and `{description}` filled from `archetypes.yaml`.

Agents should treat `data` as a read-only source of inputs conforming to the JSON schema described in `retail_spec.md` and `universal_retail_solver.py`.

---

## 3. Example Prompt File Layout

Each generated prompt file under:

```text
reloop/scenarios/retail_comprehensive/prompts/
```

has the following outer structure:

```text
### SYSTEM PROMPT ###
<SYSTEM_PROMPT_TEXT>

### USER PROMPT ###
<USER_PROMPT_TEXT_FOR_THIS_INSTANCE>
```

This is purely a file layout convention. The actual API call to the LLM uses:

- `SYSTEM_PROMPT_TEXT` as the system message, and  
- `USER_PROMPT_TEXT_FOR_THIS_INSTANCE` as the user message.

The corresponding JSON is **never** inlined inside these `.txt` files. For each scenario, the evaluation harness separately loads the matching JSON file into the Python variable `data` before executing the code returned by the model.

---

## 4. Example User Prompt (Instance: `retail_f3_storage_bottleneck_v0`)

Below is a concrete example of the **user** portion of the prompt for one instance  
(`retail_f3_storage_bottleneck_v0`). It follows the generic template above and assumes:

- JSON file: `scenarios/retail_comprehensive/data/retail_f3_storage_bottleneck_v0.json`
- The JSON has already been parsed into a Python variable named `data`.

```text
[SCENARIO]
Family: F3 (Shared Resources and Capacity)
Archetype: retail_f3_storage_bottleneck
Scenario ID: retail_f3_storage_bottleneck_v0

Business narrative:
All locations share a tight storage capacity in each period. The storage space
represents a generic mix of dry, ambient, or temperature-controlled storage.
At each (location, period), inventory for all SKUs together must fit into a single
shared capacity. Higher-volume items consume more storage than smaller items.
Demand is seasonal but feasible if the model correctly couples inventory across
products via a single storage-capacity constraint at each (location, period).

Structure cues:
- Use the same single-echelon inventory and lost-sales logic as in the core
  operations family: multiple products, multiple locations, and exogenous
  seasonal demand per product.
- At every location and period, a single shared storage-capacity constraint
  limits the volume-weighted sum of on-hand inventory across all products using
  product-specific storage usage ("cold_usage") and the location capacity
  ("cold_capacity") from the JSON.
- There is no transshipment and lead times are zero in this archetype.
- Substitution behavior remains as defined by the "sub_edges" field in the JSON
  (which may be empty or include a small number of arcs).
- Labor capacity and production capacity are as in the base scenario, but must
  still be respected wherever specified in the JSON.

Operational context:
- The JSON provides top-level keys "periods", "products", and "locations".
- Demand for each product is given as a time series in "demand_curve" and is
  allocated across locations via "demand_share".
- Storage capacity per location is given in "cold_capacity", and per-unit
  storage usage per product is given in "cold_usage".
- Costs for inventory, waste, and lost sales are provided in the "costs" object.

JSON data (do not modify):
The evaluation harness provides a Python variable called `data` that contains the
entire JSON object for this instance, using the top-level fields described above.

[INSTRUCTION]
Using ONLY the information above, write a complete Python script that:

1) Imports gurobipy as gp and from gurobipy import GRB,
2) Assumes the JSON has already been loaded into a Python variable called `data`,
3) Builds a mixed-integer linear program with at least the following elements:

   - Decision variables for:
     * Inventory by product, location, period, and (if needed) vintage or shelf-life age,
     * Orders or inbound quantities by product, location, and period (respecting production caps),
     * Lost sales by product, location, and period,
     * Any additional variables needed to represent the shared storage constraint.

   - Inventory balance constraints that track how inventory evolves over time.
   - A single shared storage-capacity constraint at each (location, period) that
     couples all SKUs using their per-unit storage usage and the "cold_capacity"
     values from the JSON.
   - No transshipment between locations and no positive lead times; inbound flow
     becomes available immediately unless the JSON specifies otherwise.

4) Sets an objective that minimizes total cost, including:
   - Inventory holding cost,
   - Waste cost (if applicable),
   - Lost-sales penalties.

5) Builds the model, calls the solver, and prints:
   - The solver status,
   - The optimal objective value.

Return ONLY the Python source code as plain text, with no comments and no Markdown.
```

This example is illustrative; all other archetypes follow the same structure but use  
their own business narrative and structure cues from `archetypes.yaml` / `retail_spec.md`.

---

## 5. Repair Prompt Template (ReLoop-style IIS Feedback)

When a solver (for example, the reference `universal_retail_solver.py`) detects infeasibility
or structural inconsistencies in the LLM-generated model, a **repair prompt** can be built.
This prompt reuses the same scenario description and JSON semantics, but augments them with
diagnostics about what went wrong.

A generic repair template is:

```text
[SCENARIO]
(same text as the original user prompt for this instance)

[PREVIOUS MODEL]
Here is the previous Python model you wrote:

<MODEL_CODE_SNIPPET>

[DIAGNOSTICS]
The solver reports that the model is infeasible or structurally inconsistent.
An IIS (Irreducible Infeasible Subset) or conflict refinement points to the
following constraints or relationships as problematic:

<PLAIN_LANGUAGE_SUMMARY_OF_IIS>

[INSTRUCTION]
Revise the model to resolve these conflicts while keeping the JSON data and the
scenario description unchanged. In particular, consider:

- Whether constraints that require strict demand satisfaction should be relaxed
  by introducing lost-sales variables with appropriate penalties.
- Whether shared capacity constraints couple SKUs correctly at each (location, period).
- Whether any logical impossibility (for example, demand > maximum possible supply)
  should be handled through slack variables or lost sales instead of hard equalities.

Return the corrected Python script as plain text, replacing the previous model.
Do not change any numbers in the JSON or assume access to external data.
```

The IIS summary is generated by the evaluation framework using the reference solver and
its conflict-refinement APIs; the LLM only sees a plain-language description.

---

## 6. Relation to the Reference Solver and Time Limits

- `universal_retail_solver.py` is the **canonical reference model** for this benchmark.
- It uses a 60-second time limit (`TimeLimit = 60`) and a 1% relative MIP gap
  (`MIPGap = 0.01`) for all instances, and suppresses solver logs (`OutputFlag = 0`).
- `run_benchmark.py` runs the reference solver on all 190 JSON files and records:

  - `OPTIMAL` when the solver proves optimality within the time limit and gap,
  - `OPTIMAL (TL)` when it hits the time limit but returns a feasible incumbent,
  - `TIMEOUT` when it hits the time limit with no incumbent solution,
  - `INFEASIBLE` when the model is proven infeasible.

These labels, together with objective values, form the baseline in
`reloop/eval/benchmark_results.csv`. LLM-generated models are evaluated by
running the same instances with the same solver settings and comparing
feasibility rates and objective gaps to this reference.

---

## 7. Summary

- `retail_spec.md` defines the **archetypes and structural intent** of the 38 retail scenarios.
- `retail_prompts.md` (this file) defines the **system and user prompt formats** used to
  query LLM-based optimization agents.
- `universal_retail_solver.py` and `run_benchmark.py` provide the **reference implementation**
  and evaluation pipeline.

Together, these components ensure that:

- Every JSON instance has a clear business meaning.
- Prompt text and JSON semantics are tightly aligned with the actual code.
- Results from different LLMs are comparable under a fixed 60-second time limit and 1% MIP gap.
