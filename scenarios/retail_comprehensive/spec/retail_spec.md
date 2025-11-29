# Comprehensive Retail Supply Chain Benchmark

> File: `reloop/scenarios/retail_comprehensive/spec/retail_spec.md`
> Depends on:
>
> * `reloop/solvers/universal_retail_solver.py`
> * `reloop/tools/retail_benchmark_generator.py`
> * `reloop/eval/run_benchmark.py`

---

## 1. Overview

This benchmark evaluates **text-to-optimization agents** on **38 retail operations archetypes**, expanded into **190 solver-validated instances** via controlled perturbations. Together, these instances cover a broad spectrum of modern retail supply chains.

All instances share a **single JSON schema** and are solved by a **single universal MILP formulation** implemented in:

* `reloop/solvers/universal_retail_solver.py` (the *Universal Retail Solver*, URS).

The benchmark uses three generic SKUs:

* `SKU_Basic`
* `SKU_Premium`
* `SKU_ShortLife`

These abstract SKUs represent a wide range of real categories (grocery, fresh food, FMCG, apparel, electronics, pharmaceuticals). Structural building blocks include:

* Operations and assortment (multi-product demand, substitution, cannibalization)
* Shared resources (storage, production, labor)
* Dynamic disruptions (demand and supply shocks)
* Logistics and MIP logic (MOQ, pack sizes, lead times, fixed order cost)
* Network and multi-echelon flows (transshipment, hub-and-spoke, ring routing)
* Omni-channel operations (reverse logistics, ship-from-store, labor constraints)
* Feasibility traps and model repair (slack variables / lost-sales logic)

Cold-chain storage is treated as **one possible shared capacity type** inside the generic storage-capacity module; it is not the central topic of the benchmark.

---

## 2. Scenario Families (The Retail 3×3 Matrix)

The benchmark is organized into eight Scenario Families. Some families contain four archetypes, others six, to cover pricing/promotions, robust-style planning, and multi-echelon routing.

Each **archetype name** (e.g., `retail_f3_storage_bottleneck`) corresponds to:

* a generator function in `retail_benchmark_generator.py`, and
* a JSON file prefix under `scenarios/retail_comprehensive/data/`.

| Family | Name         | Retail Function                                          | Mathematical Logic                               | # Archetypes |
| ------ | ------------ | -------------------------------------------------------- | ------------------------------------------------ | -----------: |
| 1      | Operations   | Inventory Planning, Cost Control                         | Basic LP, Horizon Scaling                        |            4 |
| 2      | Assortment   | Merchandising, Substitution, Cannibalization, Promotions | Substitution Logic, Budget Coupling              |            6 |
| 3      | Resources    | Sourcing, Warehousing, Shared Resources                  | Capacity Constraints (Knapsack)                  |            4 |
| 4      | Dynamics     | Risk & Disruption Management, Robust-Style Planning      | Time-dependent Constraints, Demand/Supply Shocks |            6 |
| 5      | Feasibility  | Model Repair (Stress Tests)                              | Logical Inconsistency (requires Slack)           |            4 |
| 6      | Logistics    | Procurement & Transport                                  | MIP (Binary, Integer, Lead Time)                 |            4 |
| 7      | Network      | Distribution Strategy, Multi-echelon, Routing Proxies    | Network Flow (Transshipment)                     |            6 |
| 8      | Omni-channel | Store Ops, Returns, Online Fulfillment                   | Reverse Flows, Labor Constraints                 |            4 |

Overall:

* **38 structural archetypes**
* **5 numerical variations per archetype**
* **190 JSON instances** in total

All instances are solved by `universal_retail_solver.py`, so **the mathematical structure is fixed**; archetypes differ only through data and which logical features are effectively active.

---

## 3. Instance Construction

### 3.1 Archetypes and Variations

Each archetype is a deterministic *structural pattern* (e.g., “peak supply failure”, “MOQ binary logic”, “three-echelon Plant–DC–Store network”).

For each archetype, `retail_benchmark_generator.py` produces **5 JSON instances**:

* **v0**: Base configuration as defined by the archetype function.
* **v1–v4**: Perturbed variants obtained by changing demand and capacity within a controlled range, while keeping the **logical structure** identical.

### 3.2 Perturbation Scheme

For each SKU and period, the base demand curve is multiplied by an i.i.d. factor in a symmetric range around 1 (default ±15%). For each location, the storage capacity is similarly scaled within the same range. All other structural fields remain unchanged.

Formally, for each SKU p and period t:

* D_{p,t}^{(v)} = D_{p,t}^{(0)} × ε_{p,t}^{(v)}, with ε_{p,t}^{(v)} in [1 − δ, 1 + δ], default δ = 0.15.

For each location l:

* Cap_{l}^{(v)} = Cap_{l}^{(0)} × η_{l}^{(v)}, with η_{l}^{(v)} in [1 − δ, 1 + δ].

The **set of active constraint mechanisms** for an archetype (e.g., “shared storage capacity”, “MOQ”, “lead time”, “one-way substitution”) does not change across its 5 variants; only numerical parameters change. This ensures:

* Structural difficulty is fixed at the archetype level;
* Numerical variations prevent overfitting to a single deterministic instance.

---

## 4. Directory Structure

The benchmark lives under the `reloop` project tree:

```text
reloop/
├── solvers/
│   └── universal_retail_solver.py       # Universal Retail Solver (URS)
├── tools/
│   └── retail_benchmark_generator.py    # Retail instance generator (38 archetypes × 5 variations)
├── scenarios/
│   └── retail_comprehensive/           # Comprehensive retail scenarios
│       ├── spec/
│       │   ├── retail_spec.md          # Structural and semantic specification (this file)
│       │   └── retail_prompts.md       # Prompt design and templates for LLM-based agents
│       ├── data/                       # 190 JSON instances (38 archetypes × 5 variations)
│       └── prompts/                    # Per-instance text prompts (one .txt per JSON)
└── eval/
    ├── run_benchmark.py                # Batch evaluation script (writes benchmark_results.csv)
    └── benchmark_results.csv           
```

The `prompts/` directory is generated automatically by the prompt builder script and contains one `.txt` file per JSON instance. Each file concatenates a system prompt and a user prompt and assumes that the JSON content has already been loaded into a Python variable called `data` inside the evaluation harness.

### 4.1 Agent Interface and Execution Model

For each JSON instance `<scenario>.json`, there is a companion prompt file `<scenario>.txt` under `scenarios/retail_comprehensive/prompts/`. The prompt file **does not embed the JSON payload**; instead, it defines how the modeling agent should interpret a pre-loaded Python dictionary called `data`.

During evaluation, `reloop/eval/run_benchmark.py` performs the following steps for each scenario:

1. Load the JSON file into a Python dictionary:

   ```python
   with open(f".../data/{scenario}.json", "r") as f:
       data = json.load(f)
   ```
2. Read the corresponding prompt file and send its contents (system + user prompt) to the text-to-optimization agent.
3. Receive from the agent a **plain-text Python script** that is expected to build and solve a model **using the pre-loaded `data` object**.
4. Execute the returned script in a Python environment where `data` is available in the global namespace:

   ```python
   exec(code_str, {"data": data, ...})
   ```

The agent is therefore **not** expected to read JSON files from disk or modify the data; it must treat `data` as a read-only input that conforms to the schema described in Sections 5–6.

---

## 5. JSON Schema (High-Level)

The exact schema is implemented in `universal_retail_solver.py` and `retail_benchmark_generator.py`. This section documents the **intended meaning** of the fields so that agents and evaluators understand how the data map into the universal model.

Each instance JSON has the following top-level structure (conceptual example):

```jsonc
{
  "name": "retail_f3_storage_bottleneck_v0",
  "description": "Human-readable scenario description (not used by URS)",
  "periods": 20,
  "products": ["SKU_Basic", "SKU_Premium", "SKU_ShortLife"],
  "locations": ["DC1", "DC2", "DC3", "DC4", "DC5"],

  "shelf_life": { "SKU_Basic": 10, "SKU_Premium": 8, "SKU_ShortLife": 4 },
  "lead_time":  { "SKU_Basic": 0,  "SKU_Premium": 0, "SKU_ShortLife": 0 },

  "cold_capacity": { "DC1": 4000, "DC2": 3500, "DC3": 3000, "DC4": 3000, "DC5": 2500 },
  "cold_usage":    { "SKU_Basic": 1.0, "SKU_Premium": 3.0, "SKU_ShortLife": 1.2 },

  "production_cap": {
    "SKU_Basic":   [ ... per-period caps ... ],
    "SKU_Premium": [ ... ],
    "SKU_ShortLife": [ ... ]
  },

  "labor_cap": {
    "DC1": [ ... per-period caps ... ],
    "DC2": [ ... ],
    "...": [ ... ]
  },
  "labor_usage": { "SKU_Basic": 0.0, "SKU_Premium": 0.0, "SKU_ShortLife": 0.0 },

  "return_rate": {
    "SKU_Basic": 0.0,
    "SKU_Premium": 0.0,
    "SKU_ShortLife": 0.0
  },

  "demand_curve": {
    "SKU_Basic":   [ ... length periods ... ],
    "SKU_Premium": [ ... ],
    "SKU_ShortLife": [ ... ]
  },
  "demand_share": {
    "DC1": 0.25, "DC2": 0.20, "DC3": 0.20, "DC4": 0.20, "DC5": 0.15
  },

  "costs": {
    "lost_sales": {
      "SKU_Basic": 50.0,
      "SKU_Premium": 80.0,
      "SKU_ShortLife": 40.0
    },
    "inventory": {
      "SKU_Basic": 1.0,
      "SKU_Premium": 1.5,
      "SKU_ShortLife": 1.0
    },
    "waste": {
      "SKU_Basic": 2.0,
      "SKU_Premium": 3.0,
      "SKU_ShortLife": 2.0
    },
    "fixed_order": 0.0,
    "transshipment": 0.5,
    "purchasing": {
      "SKU_Basic": 10.0,
      "SKU_Premium": 20.0,
      "SKU_ShortLife": 15.0
    }
  },

  "constraints": {
    "moq": 0,
    "pack_size": 1,
    "budget_per_period": null,
    "waste_limit_pct": null
  },

  "network": {
    "sub_edges":  [["SKU_Basic", "SKU_Premium"]],
    "trans_edges": []
  }
}
```

URS interprets these fields as follows:

* `periods`, `products`, `locations` define the index sets for time, SKUs, and nodes.
* `shelf_life` controls how many age vintages are tracked for each product.
* `lead_time` controls when orders placed at time t become available as inventory.
* `cold_capacity` and `cold_usage` together define per-period storage constraints at each location.
* `production_cap` limits the total order quantity per product in each period (summed across locations).
* `labor_cap` and `labor_usage` define per-period labor-hour constraints by location (when binding).
* `return_rate` defines product-specific reverse logistics: a fraction of prior-period sales re-enters as inventory.
* `demand_curve` encodes the aggregate demand over time by product; `demand_share` splits that demand across locations. URS multiplies these to obtain a demand tensor indexed by (product, location, period).
* `costs` provide all unit costs used in the objective: purchasing, holding, waste, lost sales, fixed ordering, and transshipment.
* `constraints.moq` and `constraints.pack_size` trigger minimum-order and pack-size logic in procurement.
* `constraints.budget_per_period` optionally activates a per-period spend cap (open-to-buy style budget).
* `constraints.waste_limit_pct` optionally activates a global waste cap relative to total demand volume.
* `network.sub_edges` defines directed arcs (pf, pt) meaning “demand for pf may be served using inventory of pt” (one-way substitution).
* `network.trans_edges` defines directed arcs between locations representing transshipment or distribution edges.

All structural variety across the 190 instances is expressed through **these fields and their numerical values**, not through separate model code.

---

## 6. Detailed Archetypes by Family

Below we list all **38 archetypes**, grouped by Scenario Family, along with their retail interpretation and **structural intent**.

This section is for human readers and code alignment; it is not meant to be passed verbatim to modeling agents.

---

### 6.1 Family 1 – Core Operations (F1)

Tests fundamental multi-period inventory balancing and cost trade-offs using basic LP-style logic.

#### `retail_f1_base`

* **Retail story**: Standard seasonal demand for a generic multi-product category served from several distribution centers. Inventory is non-perishable at the horizon scale considered, and there is a simple one-way substitution from the basic product into the premium product.

* **Active mechanisms**:

  * Multi-period inventory balance
  * Shared storage capacity by location and period
  * One-way substitution from `SKU_Basic` to `SKU_Premium`
  * Lost sales penalty (no backorders)

* **Structural intent**:

  * Single-echelon stock: ending inventory per product, location, and period equals previous inventory plus inbound production minus units sold and discarded.
  * Storage capacity couples all SKUs at each location-period through their storage usage coefficients, although capacity is not necessarily tight in this archetype.
  * When basic inventory is insufficient, some basic demand may be served from premium inventory via the one-way substitution edge; any remaining unmet demand is lost and penalized.
  * There is no transshipment between locations, no lead times, and labor capacity is effectively non-binding.
  * The objective is to minimize total cost over the horizon, including purchasing, holding, waste, and lost sales penalties.

#### `retail_f1_high_waste`

* **Retail story**: Fresh food or short-life products where waste is very expensive. The retailer faces strong pressure to avoid overstocking and unnecessary disposal of inventory.

* **Active mechanisms**:

  * Same base structure as `retail_f1_base`.
  * Short effective shelf life.
  * Waste cost much larger than holding cost.

* **Structural intent**:

  * Inventory aging and waste tracking matter more because of the higher waste penalty.
  * The optimizer has a strong incentive to keep inventories lean and avoid overproduction, even at the risk of lost sales.

#### `retail_f1_jit_logic`

* **Retail story**: High holding-cost regime mimicking just-in-time or fast fashion, where inventory carried forward is very expensive.

* **Active mechanisms**:

  * Same base structure as `retail_f1_base`.
  * Holding costs multiplied significantly.

* **Structural intent**:

  * The objective discourages carrying inventory; optimal policies rely more on frequent production subject to caps and accept some lost sales when necessary.

#### `retail_f1_52_weeks`

* **Retail story**: Full-year replenishment schedule (52-period weekly horizon). Operational logic is otherwise identical to the core operations baseline.

* **Active mechanisms**:

  * Same constraints as `retail_f1_base`.
  * Time horizon extended from 20 to 52 periods by repeating and truncating the base demand and capacity patterns.

* **Structural intent**:

  * Tests scalability of time indexing and the cumulative effect of decisions over a longer horizon.

---

### 6.2 Family 2 – Assortment, Substitution, Pricing & Promotions (F2)

Tests multi-product interactions: substitution, cannibalization, price-like logic, and promotional lifts.

#### `retail_f2_no_substitution`

* **Retail story**: Independent demand; baseline assortment without substitution.

* **Active mechanisms**:

  * Same stock-flow structure as F1 but with an empty substitution graph.

* **Structural intent**:

  * Demand for each product can only be served by its own inventory; there are no cross-SKU flows.

#### `retail_f2_circular_sub`

* **Retail story**: Circular substitution ring among `SKU_Basic`, `SKU_Premium`, and `SKU_ShortLife` (brand switching).

* **Active mechanisms**:

  * Directed ring substitution: `Basic → Premium → ShortLife → Basic`.

* **Structural intent**:

  * Demand balance includes substitution flows; total sold units remain capacity-constrained.
  * The model must route demand along allowed substitution edges without double-counting sales.

#### `retail_f2_cannibalization`

* **Retail story**: A low-margin basic product is heavily promoted, doubling its demand and making its lost sales penalty relatively cheap, while storage is tightened. Basic can crowd out inventory for higher-margin items.

* **Active mechanisms**:

  * Demand lift on `SKU_Basic`.
  * Asymmetric margins and lost-sales penalties.
  * Tighter storage capacity.

* **Structural intent**:

  * The scenario creates cannibalization pressure where stocking more basic units can displace premium and short-life units due to limited storage.

#### `retail_f2_ultra_fresh`

* **Retail story**: Extremely short shelf life (e.g., daily bakery). Units quickly expire if not sold.

* **Active mechanisms**:

  * Shelf life reduced to just a few periods.
  * Waste cost active.

* **Structural intent**:

  * Perishable inventory with fast expiration and significant waste; the model must track vintage ages and ensure expired units are discarded and penalized.

#### `retail_f2_price_band_tight`

* **Retail story**: Price-band-style logic with a “premium” anchor. Premium has higher effective margin and stricter service requirements; basic is cheaper but more price-sensitive. Basic demand can partially shift to premium when basic is unavailable.

* **Active mechanisms**:

  * One-way substitution from basic to premium.
  * Higher lost-sales cost for premium.
  * Adjusted purchasing costs to emulate different price bands.

* **Structural intent**:

  * Tests whether agents correctly protect premium inventory and use basic inventory as a buffer without altering the substitution structure.

#### `retail_f2_promo_budget`

* **Retail story**: Promotions in the final periods double demand for basic and short-life products. An open-to-buy budget couples spending across periods.

* **Active mechanisms**:

  * Promo demand lift in the last few periods.
  * Per-period budget constraint on purchasing and fixed-cost spending.

* **Structural intent**:

  * Promo periods have higher demand; the budget constraint limits how aggressively promotions can be supported, creating intertemporal trade-offs.

---

### 6.3 Family 3 – Shared Resources & Capacity Coupling (F3)

Tests shared capacities such as storage, volume, and upstream production.

#### `retail_f3_storage_bottleneck`

* **Retail story**: All locations face tight storage capacity; the same storage resource must be shared across all products at each location-period.

* **Active mechanisms**:

  * Shared storage capacity by location and period.

* **Structural intent**:

  * Storage constraints couple all SKUs via their storage usage factors and are designed to be binding.

#### `retail_f3_volumetric_constraint`

* **Retail story**: Bulky premium items consume more storage volume, making premium the main driver of storage pressure.

* **Active mechanisms**:

  * Shared storage capacity.
  * Higher storage usage for `SKU_Premium`.

* **Structural intent**:

  * Encourages careful trade-offs between stocking bulky high-margin items and smaller items under a common storage cap.

#### `retail_f3_supply_bottleneck`

* **Retail story**: Storage is abundant but upstream production capacity is tight. The bottleneck is the ability to produce units each period, not to store them.

* **Active mechanisms**:

  * Reduced production capacity per product.
  * Very large storage capacities.

* **Structural intent**:

  * Lost sales penalties dominate because supply constraints, not storage, drive infeasibility.

#### `retail_f3_unbalanced_network`

* **Retail story**: One location acts as a high-capacity hub; other locations have very limited storage. Demand is local to each location with no lateral transshipments.

* **Active mechanisms**:

  * Highly asymmetric storage capacities across locations.

* **Structural intent**:

  * Tests how agents handle extreme capacity asymmetry when inventory cannot move between locations.

---

### 6.4 Family 4 – Dynamics, Disruption, and Robust-Style Planning (F4)

Tests resilience to time-dependent demand and supply shocks.

#### `retail_f4_early_stockout`

* **Retail story**: An upstream supply failure at launch: production is zero for several initial periods while demand already exists.

* **Active mechanisms**:

  * Production capacity set to zero in early periods.

* **Structural intent**:

  * Forces planning with pre-positioned inventory and careful rationing during the early disruption window.

#### `retail_f4_peak_failure`

* **Retail story**: An upstream supply failure during a high-demand peak (e.g., holiday season).

* **Active mechanisms**:

  * Demand peaks mid-horizon.
  * Production capacity is zero during a mid-horizon block of periods.

* **Structural intent**:

  * Creates a strong trade-off between building early inventory and accepting lost sales during the peak.

#### `retail_f4_demand_surge`

* **Retail story**: A single-period viral demand spike where demand multiplies sharply for all products.

* **Active mechanisms**:

  * One period’s demand multiplied by a large factor.

* **Structural intent**:

  * Tests whether agents anticipate the spike and balance holding costs against the risk of lost sales in the surge period.

#### `retail_f4_quality_hold`

* **Retail story**: A quality issue forces the retailer to halt production of the basic product after a certain time; existing units can still be sold until exhausted.

* **Active mechanisms**:

  * Production capacity for `SKU_Basic` set to zero after a cutoff period.

* **Structural intent**:

  * Asymmetric disruption requires reallocation of inventory and possible use of substitution if available.

#### `retail_f4_robust_variance`

* **Retail story**: Alternating high/low demand with very expensive lost sales, mimicking a robust-planning environment.

* **Active mechanisms**:

  * Demand oscillates across periods.
  * Lost-sales penalties are increased for all products.

* **Structural intent**:

  * Encourages safety stock and robust planning against variance.

#### `retail_f4_supply_risk`

* **Retail story**: Mid-season reduction in production capacity plus more expensive waste, capturing persistent but moderate supply risk.

* **Active mechanisms**:

  * Production caps reduced over a mid-horizon band.
  * Waste cost increased for all products.

* **Structural intent**:

  * Encourages balancing early builds against the risk of obsolescence and waste when supply becomes tighter.

---

### 6.5 Family 5 – Feasibility Traps & Model Repair (F5)

Tests detection of logical impossibilities and repair via lost sales and waste decisions.

#### `retail_f5_impossible_demand`

* **Retail story**: Demand is scaled so high that it is mathematically impossible to fully satisfy it given production and storage capacities.

* **Active mechanisms**:

  * Demand curves multiplied by a large factor.

* **Structural intent**:

  * Forces the model to rely heavily on lost sales; agents that implicitly enforce full service will fail or produce infeasible plans.

#### `retail_f5_strict_service_trap`

* **Retail story**: Storage capacity is reduced so much that high service levels are incompatible with physical constraints. This is a trap for models that assume near-perfect service.

* **Active mechanisms**:

  * Storage capacities scaled down sharply.

* **Structural intent**:

  * Tests whether agents gracefully allow lost sales instead of enforcing unrealistic service targets.

#### `retail_f5_storage_overflow`

* **Retail story**: Storage capacities are set extremely close to zero so that almost any inbound inventory risks violating capacity.

* **Active mechanisms**:

  * Storage capacities near zero.

* **Structural intent**:

  * Feasible plans must keep on-hand inventory extremely low, using waste and lost sales to remain within capacity.

#### `retail_f5_ultimate_stress`

* **Retail story**: Composite stress test combining tight storage, a peak-period supply failure, and no substitution. Intended as a “boss level” stress scenario.

* **Active mechanisms**:

  * Tight storage capacity.
  * Mid-horizon production failure.
  * Empty substitution graph.

* **Structural intent**:

  * Tests whether agents can maintain feasibility under simultaneous capacity, supply, and assortment stress.

---

### 6.6 Family 6 – Logistics & Mixed-Integer Procurement (F6)

Tests discrete logic in procurement and pipeline inventory.

#### `retail_f6_lead_time`

* **Retail story**: Non-zero, product-specific lead times; orders placed now arrive after several periods.

* **Active mechanisms**:

  * Positive lead times for each product.
  * Pipeline inventory representation.

* **Structural intent**:

  * Inflow at period t depends on orders placed at t − lead_time[p]; newly ordered units cannot be used immediately.

#### `retail_f6_moq_binary`

* **Retail story**: Minimum order quantity (MOQ). Each ordering decision is all-or-nothing: either do not order or order at least a fixed minimum quantity.

* **Active mechanisms**:

  * Binary order trigger per product/location/period.
  * MOQ constraint linked to the binary trigger.

* **Structural intent**:

  * Classic lot-sizing structure where small orders are disallowed.

#### `retail_f6_fixed_order_cost`

* **Retail story**: Fixed ordering or transportation cost per period encourages batching rather than many small orders.

* **Active mechanisms**:

  * Fixed cost charged when any positive order quantity is placed.

* **Structural intent**:

  * Lot-sizing behavior where binary order indicators determine whether the fixed cost is incurred.

#### `retail_f6_pack_size_integer`

* **Retail story**: Orders must be placed in integer multiples of a pack or pallet size.

* **Active mechanisms**:

  * Integer decision on “number of packs”.
  * Order quantity constrained to pack_size × integer.

* **Structural intent**:

  * Integrality couples order quantities across products and periods via capacity and budget constraints.

---

### 6.7 Family 7 – Network, Multi-echelon, and Routing Proxies (F7)

Tests multi-echelon distribution and network-flow complexity.

#### `retail_f7_transshipment`

* **Retail story**: Fully connected lateral transshipment network among locations. Inventory can be moved between any pair of locations at a transshipment cost.

* **Active mechanisms**:

  * Directed transshipment edges between all ordered pairs of distinct locations.

* **Structural intent**:

  * Inventory balance includes inbound and outbound transshipment; flows are capacity-limited by local inventory and charged per unit.

#### `retail_f7_hub_and_spoke`

* **Retail story**: One hub DC with large storage ships to several low-capacity spokes. All lateral movements must pass through the hub.

* **Active mechanisms**:

  * Storage capacity concentrated at hub.
  * Transshipment edges only from hub to spokes.

* **Structural intent**:

  * Captures a centralized distribution strategy; spokes rely on the hub to meet local demand.

#### `retail_f7_budget_limit`

* **Retail story**: Per-period open-to-buy budget limits total spending on inventory-related activities across locations.

* **Active mechanisms**:

  * Budget constraint per period on purchasing (and fixed ordering costs when present).

* **Structural intent**:

  * Couples decisions across locations and products within each period through a spending cap.

#### `retail_f7_multi_sourcing`

* **Retail story**: Products behave as if they came from different sourcing regimes: some slow but cheap, others fast but expensive, via heterogeneous lead times and holding costs.

* **Active mechanisms**:

  * Product-specific lead times and holding costs.

* **Structural intent**:

  * Mimics multi-speed sourcing channels; agents must trade off fast vs. slow inventory dynamics.

#### `retail_f7_multiechelon_chain`

* **Retail story**: Explicit three-echelon chain consisting of a plant, distribution centers, and stores. Demand occurs only at stores; upstream nodes hold and move inventory.

* **Active mechanisms**:

  * Locations redefined as plant, DCs, and stores.
  * Directed transshipment edges from plant to DCs and from DCs to stores.
  * Demand shares concentrated at stores.

* **Structural intent**:

  * Tests whether agents correctly model multi-echelon flows and restrict demand and lost sales to the store level.

#### `retail_f7_ring_routing`

* **Retail story**: Locations are connected in a directed ring, mimicking a truck route that visits each node in a fixed cycle. Storage capacities are tightened to make the network structure matter.

* **Active mechanisms**:

  * Transshipment edges forming a directed cycle over locations.
  * Slightly reduced storage capacities.

* **Structural intent**:

  * Inventory can only move along ring edges; capacity along the ring proxies for vehicle capacity.

---

### 6.8 Family 8 – Omni-channel & Store Operations (F8)

Tests omni-channel flows and in-store resource constraints.

#### `retail_f8_reverse_logistics`

* **Retail story**: A fraction of units sold in each period are returned by customers in the next period and re-enter inventory as reverse flow.

* **Active mechanisms**:

  * Product-specific return rates.
  * Reverse flow added to next-period arrivals.

* **Structural intent**:

  * Inventory balance includes both new production and returns; returned units are subject to the same shelf-life and capacity constraints as new units.

#### `retail_f8_labor_constraint`

* **Retail story**: Store and warehouse labor capacity limits how much inventory can be handled in each period. Labor is required for activities such as receiving, picking, and replenishment.

* **Active mechanisms**:

  * Labor capacity per location-period.
  * Per-unit labor usage by product.

* **Structural intent**:

  * Labor-hour constraints couple operational decisions across products and activities in each period.

#### `retail_f8_ship_from_store`

* **Retail story**: Stores fulfill both walk-in traffic and online orders (ship-from-store). Storage and labor capacities at stores are scaled up, but labor usage per unit is higher.

* **Active mechanisms**:

  * Increased storage and labor capacity.
  * Higher per-unit labor usage.

* **Structural intent**:

  * Captures the higher operational load of omni-channel fulfillment at the store level.

#### `retail_f8_sustainability`

* **Retail story**: The retailer operates under a sustainability or regulatory constraint that limits total waste. Over the entire planning horizon, total units discarded as waste must not exceed a fixed fraction of total demand.

* **Active mechanisms**:

  * Global waste cap relative to total demand volume.

* **Structural intent**:

  * The model must trade off holding more inventory, losing sales, and discarding units while respecting a cumulative waste constraint.

---

## 7. Evaluation Pipeline and Solver Settings

`run_benchmark.py` uses `universal_retail_solver.py` to compute reference solutions for all 190 JSON instances and writes a CSV summary:

* Output file: `reloop/eval/benchmark_results.csv`

Each row contains:

* `scenario` – JSON file name without `.json` (for example, `retail_f6_moq_binary_v2`)
* `status` – solver status as mapped by URS
* `objective` – objective value (total cost), or `N/A` if no feasible solution is available

### 7.1 Universal Retail Solver (URS) configuration

`universal_retail_solver.py` is implemented with Gurobi and uses the following global settings:

* `TimeLimit = 60` seconds
  Hard cap on wall-clock time for each instance, preventing stalling on hard MIP cases (especially Family 6 and some network instances).
* `MIPGap = 0.01`
  URS stops once it finds a solution within 1% relative optimality gap, treating this as sufficiently close to optimal.
* `OutputFlag = 0`
  Gurobi log output is suppressed in normal runs.

These settings define the **canonical baseline** against which text-to-optimization agents are evaluated. If an agent uses different settings, its results should be interpreted relative to this reference.

### 7.2 Status mapping

For each instance, URS reports one of the following high-level statuses:

* `OPTIMAL` – Gurobi reached proven optimality within the 60-second limit and 1% gap.
* `OPTIMAL (TL)` – Gurobi hit the time limit (`TIME_LIMIT`) but had at least one incumbent solution; the best incumbent is recorded and treated as a near-optimal baseline.
* `TIMEOUT` – Gurobi hit the time limit with no incumbent solution; no objective value is reported.
* `INFEASIBLE` – The model is mathematically infeasible under the given data; no objective value is reported.
* `Code <k>` – Any other Gurobi status code k that may occur (rare in practice). In downstream tooling, this may optionally be wrapped as `CRASH:<k>` or similar for logging purposes.

The `objective` field in `benchmark_results.csv` is:

* the optimal or best-incumbent objective value when `status` is `OPTIMAL` or `OPTIMAL (TL)`, and
* `N/A` otherwise.

These values serve as the **reference baseline** for evaluating text-to-optimization agents on:

* Feasibility rate across all instances;
* Objective gap relative to URS on instances where URS has a solution;
* Robustness under the full mix of Families F1–F8 (operations, assortment, disruptions, logistics, network, and omni-channel scenarios).
