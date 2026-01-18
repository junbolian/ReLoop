"""
Generate Synthetic Training Data V2 for ReLoop Agent.

This script generates high-quality supervised fine-tuning (SFT) data
covering all 8 step prompts (00-07) of the ReLoop pipeline:
- Step 00: Global Guardrails (embedded in system prompts)
- Step 01: Contract Extraction
- Step 02: Model Spec Sheet Generation
- Step 03: Constraint Templates Generation
- Step 04: GurobiPy Code Generation
- Step 05: JSON Format Repair
- Step 06: Runtime Error Repair
- Step 07: Semantic Probe Failure Repair

Features:
- 18 industry verticals for diverse scenarios
- 48 feature combinations for complexity variation
- 8 error injection types for repair samples
- Semantic probe compatible code generation
- Sanity checker compatible spec sheet generation

Usage:
    python generate_synthetic_training_data_v2.py

Output:
    - v2_train_all.jsonl (complete training set)
    - v2_step1_contract.jsonl
    - v2_step2_spec_sheet.jsonl
    - v2_step3_templates.jsonl
    - v2_step4_codegen.jsonl
    - v2_step05_repair_json.jsonl
    - v2_step06_repair_runtime.jsonl
    - v2_step07_repair_probe.jsonl
"""

import json
import random
from pathlib import Path
from typing import Any

# ============================================================================
# INDUSTRY CONFIGURATIONS
# ============================================================================

INDUSTRIES = {
    "dairy": {
        "products": ["whole_milk", "skim_milk", "yogurt", "butter", "cream", "cheese"],
        "shelf_life_range": (3, 7),
        "lead_time_range": (1, 2),
        "description": "dairy products with short shelf life"
    },
    "bakery": {
        "products": ["bread", "croissant", "muffin", "bagel", "cake", "pastry"],
        "shelf_life_range": (2, 5),
        "lead_time_range": (0, 1),
        "description": "fresh baked goods"
    },
    "produce": {
        "products": ["lettuce", "tomato", "cucumber", "carrot", "spinach", "bell_pepper"],
        "shelf_life_range": (4, 10),
        "lead_time_range": (1, 3),
        "description": "fresh fruits and vegetables"
    },
    "meat": {
        "products": ["beef", "pork", "chicken", "lamb", "turkey", "sausage"],
        "shelf_life_range": (3, 7),
        "lead_time_range": (1, 2),
        "description": "fresh meat products"
    },
    "seafood": {
        "products": ["salmon", "tuna", "shrimp", "cod", "crab", "lobster"],
        "shelf_life_range": (2, 5),
        "lead_time_range": (1, 2),
        "description": "fresh seafood"
    },
    "deli": {
        "products": ["ham", "turkey_deli", "salami", "roast_beef", "pastrami", "bologna"],
        "shelf_life_range": (5, 14),
        "lead_time_range": (1, 3),
        "description": "deli meats and prepared foods"
    },
    "frozen": {
        "products": ["ice_cream", "frozen_pizza", "frozen_veg", "frozen_fish", "frozen_meal", "popsicle"],
        "shelf_life_range": (30, 90),
        "lead_time_range": (2, 5),
        "description": "frozen food products"
    },
    "beverage": {
        "products": ["fresh_juice", "smoothie", "cold_brew", "kombucha", "lemonade", "iced_tea"],
        "shelf_life_range": (7, 21),
        "lead_time_range": (1, 3),
        "description": "fresh beverages"
    },
    "prepared_foods": {
        "products": ["salad", "sandwich", "soup", "sushi", "wrap", "pasta_salad"],
        "shelf_life_range": (2, 5),
        "lead_time_range": (0, 1),
        "description": "ready-to-eat prepared foods"
    },
    "floral": {
        "products": ["roses", "tulips", "lilies", "carnations", "sunflowers", "orchids"],
        "shelf_life_range": (5, 10),
        "lead_time_range": (1, 2),
        "description": "fresh cut flowers"
    },
    "pharmacy": {
        "products": ["insulin", "vaccine", "eye_drops", "probiotics", "serum", "biologics"],
        "shelf_life_range": (14, 60),
        "lead_time_range": (2, 5),
        "description": "temperature-sensitive pharmaceuticals"
    },
    "cosmetics": {
        "products": ["face_cream", "serum", "mask", "eye_cream", "lip_balm", "sunscreen"],
        "shelf_life_range": (30, 180),
        "lead_time_range": (3, 7),
        "description": "skincare and cosmetic products"
    },
    "pet_food": {
        "products": ["raw_dog_food", "raw_cat_food", "fresh_treats", "pet_milk", "wet_food", "fresh_kibble"],
        "shelf_life_range": (7, 21),
        "lead_time_range": (2, 4),
        "description": "fresh and refrigerated pet food"
    },
    "baby_food": {
        "products": ["infant_formula", "baby_puree", "baby_yogurt", "baby_cereal", "toddler_meal", "baby_juice"],
        "shelf_life_range": (7, 30),
        "lead_time_range": (2, 4),
        "description": "infant and toddler nutrition"
    },
    "specialty_cheese": {
        "products": ["brie", "camembert", "gouda", "cheddar", "mozzarella", "feta"],
        "shelf_life_range": (14, 60),
        "lead_time_range": (3, 7),
        "description": "artisan and specialty cheeses"
    },
    "organic": {
        "products": ["organic_milk", "organic_eggs", "organic_butter", "organic_yogurt", "organic_juice", "organic_tofu"],
        "shelf_life_range": (5, 14),
        "lead_time_range": (1, 3),
        "description": "organic and natural products"
    },
    "meal_kit": {
        "products": ["protein_kit", "veggie_kit", "pasta_kit", "stir_fry_kit", "salad_kit", "breakfast_kit"],
        "shelf_life_range": (3, 7),
        "lead_time_range": (1, 2),
        "description": "meal kit components"
    },
    "wine": {
        "products": ["red_wine", "white_wine", "rose", "sparkling", "champagne", "dessert_wine"],
        "shelf_life_range": (30, 365),
        "lead_time_range": (5, 14),
        "description": "wine and fine beverages requiring temperature control"
    }
}

# ============================================================================
# FEATURE COMBINATIONS
# ============================================================================

FEATURE_SETS = [
    {"substitution": False, "transshipment": False, "moq": False, "labor": False},
    {"substitution": True, "transshipment": False, "moq": False, "labor": False},
    {"substitution": False, "transshipment": True, "moq": False, "labor": False},
    {"substitution": True, "transshipment": True, "moq": False, "labor": False},
    {"substitution": False, "transshipment": False, "moq": True, "labor": False},
    {"substitution": True, "transshipment": False, "moq": True, "labor": False},
    {"substitution": False, "transshipment": True, "moq": True, "labor": False},
    {"substitution": True, "transshipment": True, "moq": True, "labor": False},
    {"substitution": False, "transshipment": False, "moq": False, "labor": True},
    {"substitution": True, "transshipment": False, "moq": False, "labor": True},
    {"substitution": False, "transshipment": True, "moq": False, "labor": True},
    {"substitution": True, "transshipment": True, "moq": False, "labor": True},
    {"substitution": True, "transshipment": True, "moq": True, "labor": True},
]

EXTENDED_FEATURE_SETS = []
for base in FEATURE_SETS:
    EXTENDED_FEATURE_SETS.append({**base, "pack_size": False, "budget": False, "waste_limit": False})
    EXTENDED_FEATURE_SETS.append({**base, "pack_size": True, "budget": False, "waste_limit": False})
    EXTENDED_FEATURE_SETS.append({**base, "pack_size": False, "budget": True, "waste_limit": False})
    EXTENDED_FEATURE_SETS.append({**base, "pack_size": False, "budget": False, "waste_limit": True})

FEATURE_SETS = EXTENDED_FEATURE_SETS[:48]

# ============================================================================
# GLOBAL GUARDRAILS (Step 00)
# ============================================================================

GLOBAL_GUARDRAILS = """[IDENTITY]
You are an optimization modeling agent for retail supply-chain planning.

[CORE RULES]
- `data` is a pre-loaded Python dict. Do not modify it, and don't write data = DATA_PLACEHOLDER.
- No file I/O. Never invent missing data.
- Never hard-code numeric values.
- Output must be machine-parseable. No prose, no markdown, no comments in code.

[PHYSICAL VALIDITY]
- Inventory must obey flow conservation.
- Demand each period: fulfilled, substituted, or lost. No backorders.
- Expired inventory = waste.

[VARIABLE NAMING]
Core variables (always create):
- I[p,l,t,a]: START-OF-PERIOD inventory (before sales occur)
- y[p,l,t,a]: sales from each life bucket (during the period)
- W[p,l,t]: waste (expired inventory from bucket a=1)
- Q[p,l,t]: orders/production
- L[p,l,t]: lost sales (MUST be included - this is the slack variable)

Optional variables (only if feature is active):
- S[p_from,p_to,l,t]: substitution flow (only if sub_edges nonempty)
- X[p,src,dst,t]: transshipment flow (only if trans_edges nonempty)
- z[p,l,t]: binary order indicator (only if moq > 0 or fixed_order > 0)
- n[p,l,t]: integer pack count (only if pack_size > 1)

[SUBSTITUTION SEMANTICS]
Edge [p_from, p_to] means: "p_from's demand can be served by p_to's inventory"
S[p_from, p_to, l, t] = quantity of p_from's demand fulfilled by p_to

[SHELF-LIFE AND AGING]
- a = 1: expiring (will become waste if not sold this period)
- a = shelf_life[p]: freshest (just arrived)
- Holding cost applies ONLY to buckets a >= 2

[BOUNDARY CONDITIONS]
1. INITIALIZATION at t=1: All non-fresh inventory buckets must start empty.
2. FRESH INFLOW: If t <= lead_time, no fresh inventory arrives yet.
3. AGING at t=T: Do NOT create aging constraints for the final period.
4. EMPTY EDGES: If sub_edges is empty, no substitution variables or constraints.

[DATA ACCESS]
Network data is NESTED:
  sub_edges = data.get('network', {}).get('sub_edges', [])
  trans_edges = data.get('network', {}).get('trans_edges', [])
DO NOT use data['sub_edges'] directly - this will cause KeyError!
"""

# ============================================================================
# SCENARIO GENERATION
# ============================================================================

def generate_scenario(scenario_id: int, industry: str, features: dict) -> dict:
    """Generate a complete scenario with data profile."""
    config = INDUSTRIES[industry]

    num_products = random.randint(2, 4)
    num_locations = random.randint(2, 3)
    num_periods = random.randint(5, 10)

    products = random.sample(config["products"], min(num_products, len(config["products"])))
    locations = [f"loc_{i+1}" for i in range(num_locations)]

    shelf_life = {p: random.randint(*config["shelf_life_range"]) for p in products}
    lead_time = {p: random.randint(*config["lead_time_range"]) for p in products}

    base_demand = random.randint(50, 200)
    demand_curve = {
        p: [int(base_demand * random.uniform(0.7, 1.3)) for _ in range(num_periods)]
        for p in products
    }
    demand_share = {loc: round(1.0 / num_locations, 2) for loc in locations}

    production_cap = {p: [int(base_demand * 2.0) for _ in range(num_periods)] for p in products}
    cold_capacity = {loc: int(base_demand * num_products * 3) for loc in locations}
    cold_usage = {p: round(random.uniform(0.5, 2.0), 2) for p in products}

    costs = {
        "purchasing": {p: round(random.uniform(1.0, 10.0), 2) for p in products},
        "inventory": {p: round(random.uniform(0.1, 1.0), 2) for p in products},
        "waste": {p: round(random.uniform(0.5, 5.0), 2) for p in products},
        "lost_sales": {p: round(random.uniform(2.0, 20.0), 2) for p in products}
    }

    network = {"sub_edges": [], "trans_edges": []}

    if features.get("substitution") and len(products) >= 2:
        for i in range(len(products) - 1):
            network["sub_edges"].append([products[i], products[i + 1]])

    if features.get("transshipment") and len(locations) >= 2:
        for i in range(len(locations) - 1):
            network["trans_edges"].append([locations[i], locations[i + 1]])

    constraints = {}
    if features.get("moq"):
        constraints["moq"] = random.randint(5, 20)
    if features.get("pack_size"):
        constraints["pack_size"] = random.choice([6, 12, 24])
    if features.get("budget"):
        constraints["budget_per_period"] = int(base_demand * 15)
    if features.get("waste_limit"):
        constraints["waste_limit_pct"] = round(random.uniform(0.05, 0.15), 2)

    if features.get("labor"):
        costs["labor_usage"] = {p: round(random.uniform(0.1, 0.5), 2) for p in products}
        costs["labor_cap"] = {loc: [int(base_demand * 0.8) for _ in range(num_periods)] for loc in locations}

    if features.get("moq"):
        costs["fixed_order"] = round(random.uniform(10.0, 50.0), 2)

    if features.get("transshipment"):
        costs["transshipment"] = round(random.uniform(0.5, 2.0), 2)

    data_profile = {
        "products": products,
        "locations": locations,
        "periods": num_periods,
        "shelf_life": shelf_life,
        "lead_time": lead_time,
        "demand_curve": demand_curve,
        "demand_share": demand_share,
        "production_cap": production_cap,
        "cold_capacity": cold_capacity,
        "cold_usage": cold_usage,
        "costs": costs,
        "network": network,
        "constraints": constraints
    }

    narrative = f"""A {industry} retailer operates {num_locations} locations selling {num_products} perishable products.
Planning horizon: {num_periods} periods.
Products: {', '.join(products)}.
Shelf life varies from {min(shelf_life.values())} to {max(shelf_life.values())} periods.
{"Substitution is allowed between products." if features.get("substitution") else "No substitution allowed."}
{"Transshipment between locations is available." if features.get("transshipment") else "No transshipment."}
{"Minimum order quantities apply." if features.get("moq") else ""}
{"Labor capacity constraints are active." if features.get("labor") else ""}
Objective: Minimize total costs including purchasing, holding, waste, and lost sales penalties."""

    return {
        "scenario_id": f"synth_{industry}_{scenario_id:05d}",
        "industry": industry,
        "features": features,
        "narrative": narrative.strip(),
        "data_profile": data_profile
    }


def generate_step1_response(scenario: dict) -> str:
    features = scenario["features"]
    data = scenario["data_profile"]

    controls = ["order quantities Q[p,l,t]", "inventory allocation I[p,l,t,a]", "sales allocation y[p,l,t,a]"]
    if features.get("substitution"):
        controls.append("substitution flows S[p_from,p_to,l,t]")
    if features.get("transshipment"):
        controls.append("transshipment flows X[p,src,dst,t]")

    hard_constraints = [
        "inventory flow conservation",
        "capacity limits (storage, production)",
        "perishability/aging dynamics",
        "non-negativity"
    ]
    if features.get("substitution"):
        hard_constraints.append("substitution active (demand routing)")
    else:
        hard_constraints.append("no substitution (sub_edges empty)")
    if not features.get("transshipment"):
        hard_constraints.append("no transshipment (trans_edges empty)")
    if features.get("moq"):
        hard_constraints.append("minimum order quantity")
    if features.get("labor"):
        hard_constraints.append("labor capacity limits")

    soft_violations = [{"name": "lost_sales", "penalty": "costs.lost_sales"}]

    contract = {
        "optimize": f"Minimize total cost for {scenario['industry']} supply chain over {data['periods']} periods",
        "controls": controls,
        "hard_constraints": hard_constraints,
        "soft_violations": soft_violations,
        "summary": f"Multi-period inventory optimization with perishable {scenario['industry']} products"
    }

    return json.dumps(contract, indent=2)


def generate_step2_response(scenario: dict) -> str:
    features = scenario["features"]

    sets = [
        {"name": "P", "description": "Set of products", "source": "data.products"},
        {"name": "L", "description": "Set of locations", "source": "data.locations"},
        {"name": "T", "description": "Time periods 1 to T", "source": "range(1, data.periods + 1)"},
        {"name": "A", "description": "Remaining life buckets 1 to shelf_life[p]", "source": "range(1, shelf_life[p] + 1)"}
    ]

    decisions = [
        {"name": "I", "type": "continuous", "domain": ">=0", "indices": ["p", "l", "t", "a"],
         "meaning": "Start-of-period inventory by life bucket", "active_if": "always"},
        {"name": "y", "type": "continuous", "domain": ">=0", "indices": ["p", "l", "t", "a"],
         "meaning": "Sales from each life bucket", "active_if": "always"},
        {"name": "W", "type": "continuous", "domain": ">=0", "indices": ["p", "l", "t"],
         "meaning": "Waste (expired inventory)", "active_if": "always"},
        {"name": "Q", "type": "continuous", "domain": ">=0", "indices": ["p", "l", "t"],
         "meaning": "Order/production quantity", "active_if": "always"},
        {"name": "L", "type": "continuous", "domain": ">=0", "indices": ["p", "l", "t"],
         "meaning": "Lost sales (slack variable)", "active_if": "always"}
    ]

    if features.get("substitution"):
        decisions.append({"name": "S", "type": "continuous", "domain": ">=0",
                         "indices": ["p_from", "p_to", "l", "t"],
                         "meaning": "Substitution flow", "active_if": "sub_edges nonempty"})

    if features.get("transshipment"):
        decisions.append({"name": "X", "type": "continuous", "domain": ">=0",
                         "indices": ["p", "src", "dst", "t"],
                         "meaning": "Transshipment flow", "active_if": "trans_edges nonempty"})

    if features.get("moq"):
        decisions.append({"name": "z", "type": "binary", "domain": "{0,1}",
                         "indices": ["p", "l", "t"],
                         "meaning": "Order indicator", "active_if": "moq > 0 or fixed_order > 0"})

    objective_terms = [
        {"name": "purchasing_cost", "expression": "Cost incurred when ordering/producing units",
         "source": "costs.purchasing", "active_if": "always"},
        {"name": "inventory_holding", "expression": "Cost of holding end-of-period inventory (I - y) for buckets a >= 2",
         "source": "costs.inventory", "active_if": "always"},
        {"name": "waste_cost", "expression": "Penalty for units that expire from bucket a=1",
         "source": "costs.waste", "active_if": "always"},
        {"name": "lost_sales_penalty", "expression": "Penalty for demand that cannot be fulfilled",
         "source": "costs.lost_sales", "active_if": "always"}
    ]

    constraint_families = [
        {"prefix": "sales_conservation", "meaning": "Sales + lost = effective demand",
         "indices": ["p", "l", "t"], "sense": "=", "active_if": "always"},
        {"prefix": "availability", "meaning": "Sales from bucket cannot exceed inventory",
         "indices": ["p", "l", "t", "a"], "sense": "<=", "active_if": "always"},
        {"prefix": "aging", "meaning": "Inventory moves between life buckets",
         "indices": ["p", "l", "t", "a"], "sense": "=", "active_if": "t < T and a < shelf_life"},
        {"prefix": "expire_clear", "meaning": "Waste equals unsold from bucket a=1",
         "indices": ["p", "l", "t"], "sense": "=", "active_if": "always"},
        {"prefix": "fresh_inflow", "meaning": "Fresh inventory from orders arrives at highest bucket",
         "indices": ["p", "l", "t"], "sense": "=", "active_if": "always"},
        {"prefix": "init", "meaning": "Non-fresh buckets start empty at t=1",
         "indices": ["p", "l", "a"], "sense": "=", "active_if": "t = 1 and a < shelf_life"},
        {"prefix": "storage_cap", "meaning": "Weighted inventory sum <= cold_capacity",
         "indices": ["l", "t"], "sense": "<=", "active_if": "cold_capacity present"},
        {"prefix": "prod_cap", "meaning": "Total orders <= production_cap",
         "indices": ["p", "t"], "sense": "<=", "active_if": "production_cap present"}
    ]

    if features.get("substitution"):
        constraint_families.append({"prefix": "demand_route", "meaning": "Outbound substitution <= demand",
                                   "indices": ["p", "l", "t"], "sense": "<=", "active_if": "sub_edges nonempty"})

    if features.get("moq"):
        constraint_families.extend([
            {"prefix": "moq_lb", "meaning": "Q >= moq * z", "indices": ["p", "l", "t"], "sense": ">=", "active_if": "moq > 0"},
            {"prefix": "moq_ub", "meaning": "Q <= bigM * z", "indices": ["p", "l", "t"], "sense": "<=", "active_if": "moq > 0"}
        ])

    if features.get("labor"):
        constraint_families.append({"prefix": "labor_cap", "meaning": "Labor usage <= labor capacity",
                                   "indices": ["l", "t"], "sense": "<=", "active_if": "labor_usage present"})

    edge_cases = [
        {"case": "t=1 initialization", "handling": "Set I[p,l,1,a] = 0 for a < shelf_life[p]"},
        {"case": "t=T boundary", "handling": "Do not create aging constraints for t = T"},
        {"case": "lead_time > 0", "handling": "Fresh inflow = 0 for t <= lead_time"}
    ]

    spec_sheet = {
        "sets": sets,
        "decisions": decisions,
        "objective_terms": objective_terms,
        "constraint_families": constraint_families,
        "edge_cases": edge_cases,
        "open_questions": []
    }

    return json.dumps(spec_sheet, indent=2)


def generate_step3_response(scenario: dict) -> str:
    features = scenario["features"]

    templates = [
        {
            "prefix": "sales_conservation",
            "template_type": "BALANCE",
            "applies_when": "always",
            "indices": ["p", "l", "t"],
            "equations": [{
                "name_suffix": "",
                "sense": "=",
                "lhs": "sum(y[p,l,t,a] for a in 1..shelf_life[p]) + L[p,l,t]",
                "rhs": "demand_curve[p][t-1] * demand_share[l]" +
                       (" + sum(S[p_from,p,l,t] for edges where p is target) - sum(S[p,p_to,l,t] for edges where p is source)" if features.get("substitution") else "")
            }],
            "notes": ["L (lost sales) MUST appear on LHS", "demand computed from demand_curve and demand_share"]
        },
        {
            "prefix": "availability",
            "template_type": "BALANCE",
            "applies_when": "always",
            "indices": ["p", "l", "t", "a"],
            "equations": [{"name_suffix": "", "sense": "<=", "lhs": "y[p,l,t,a]", "rhs": "I[p,l,t,a]"}],
            "notes": ["Sales cannot exceed available inventory in each bucket"]
        },
        {
            "prefix": "aging",
            "template_type": "DYNAMICS",
            "applies_when": "t < T and a < shelf_life[p]",
            "indices": ["p", "l", "t", "a"],
            "equations": [{"name_suffix": "", "sense": "=", "lhs": "I[p,l,t+1,a]", "rhs": "I[p,l,t,a+1] - y[p,l,t,a+1]"}],
            "notes": ["CRITICAL: Do NOT create for t = T"]
        },
        {
            "prefix": "expire_clear",
            "template_type": "DYNAMICS",
            "applies_when": "always",
            "indices": ["p", "l", "t"],
            "equations": [{"name_suffix": "", "sense": "=", "lhs": "W[p,l,t]", "rhs": "I[p,l,t,1] - y[p,l,t,1]"}],
            "notes": ["Unsold inventory from bucket a=1 becomes waste"]
        },
        {
            "prefix": "fresh_inflow",
            "template_type": "DYNAMICS",
            "applies_when": "always",
            "indices": ["p", "l", "t"],
            "equations": [{"name_suffix": "", "sense": "=", "lhs": "I[p,l,t,shelf_life[p]]", "rhs": "Q[p,l,t-lead_time[p]] if t > lead_time[p] else 0"}],
            "notes": ["CRITICAL: If t <= lead_time, inflow is 0"]
        },
        {
            "prefix": "init",
            "template_type": "DYNAMICS",
            "applies_when": "t = 1",
            "indices": ["p", "l", "a"],
            "equations": [{"name_suffix": "", "sense": "=", "lhs": "I[p,l,1,a]", "rhs": "0"}],
            "notes": ["CRITICAL: For a < shelf_life[p], initial inventory = 0"]
        },
        {
            "prefix": "storage_cap",
            "template_type": "CAPACITY",
            "applies_when": "cold_capacity present",
            "indices": ["l", "t"],
            "equations": [{"name_suffix": "", "sense": "<=", "lhs": "sum(cold_usage[p] * I[p,l,t,a] for p, a)", "rhs": "cold_capacity[l]"}],
            "notes": ["Weighted inventory sum limited by storage capacity"]
        },
        {
            "prefix": "prod_cap",
            "template_type": "CAPACITY",
            "applies_when": "production_cap present",
            "indices": ["p", "t"],
            "equations": [{"name_suffix": "", "sense": "<=", "lhs": "sum(Q[p,l,t] for l)", "rhs": "production_cap[p][t-1]"}],
            "notes": ["production_cap is 0-indexed"]
        }
    ]

    if features.get("substitution"):
        templates.append({
            "prefix": "demand_route",
            "template_type": "BALANCE",
            "applies_when": "sub_edges nonempty",
            "indices": ["p", "l", "t"],
            "equations": [{"name_suffix": "", "sense": "<=", "lhs": "sum(S[p,p_to,l,t] for edges where p is source)", "rhs": "demand_curve[p][t-1] * demand_share[l]"}],
            "notes": ["Outbound substitution cannot exceed product's own demand"]
        })

    if features.get("moq"):
        templates.extend([
            {"prefix": "moq_lb", "template_type": "CAPACITY", "applies_when": "moq > 0", "indices": ["p", "l", "t"],
             "equations": [{"name_suffix": "", "sense": ">=", "lhs": "Q[p,l,t]", "rhs": "moq * z[p,l,t]"}], "notes": ["If ordering, must order at least MOQ"]},
            {"prefix": "moq_ub", "template_type": "CAPACITY", "applies_when": "moq > 0", "indices": ["p", "l", "t"],
             "equations": [{"name_suffix": "", "sense": "<=", "lhs": "Q[p,l,t]", "rhs": "bigM * z[p,l,t]"}], "notes": ["If not ordering, Q must be 0"]}
        ])

    if features.get("labor"):
        templates.append({
            "prefix": "labor_cap",
            "template_type": "CAPACITY",
            "applies_when": "labor_usage present",
            "indices": ["l", "t"],
            "equations": [{"name_suffix": "", "sense": "<=", "lhs": "sum(labor_usage[p] * y[p,l,t,a] for p, a)", "rhs": "labor_cap[l][t-1]"}],
            "notes": ["labor_cap is 0-indexed"]
        })

    return json.dumps(templates, indent=2)


def generate_step4_response(scenario: dict) -> str:
    features = scenario["features"]

    code_parts = []
    code_parts.append("import gurobipy as gp")
    code_parts.append("from gurobipy import GRB")
    code_parts.append("")
    code_parts.append("products = data['products']")
    code_parts.append("locations = data['locations']")
    code_parts.append("T = data['periods']")
    code_parts.append("shelf_life = data['shelf_life']")
    code_parts.append("lead_time = data['lead_time']")
    code_parts.append("demand_curve = data['demand_curve']")
    code_parts.append("demand_share = data['demand_share']")
    code_parts.append("production_cap = data['production_cap']")
    code_parts.append("cold_capacity = data['cold_capacity']")
    code_parts.append("cold_usage = data['cold_usage']")
    code_parts.append("costs = data['costs']")
    code_parts.append("")
    code_parts.append("sub_edges = data.get('network', {}).get('sub_edges', [])")
    code_parts.append("trans_edges = data.get('network', {}).get('trans_edges', [])")
    code_parts.append("")

    if features.get("moq") or features.get("pack_size"):
        code_parts.append("constraints = data.get('constraints', {})")
        if features.get("moq"):
            code_parts.append("moq = constraints.get('moq', 0)")
        if features.get("pack_size"):
            code_parts.append("pack_size = constraints.get('pack_size', 1)")
        code_parts.append("")

    if features.get("labor"):
        code_parts.append("labor_usage = costs.get('labor_usage', {})")
        code_parts.append("labor_cap = costs.get('labor_cap', {})")
        code_parts.append("")

    if features.get("substitution"):
        code_parts.append("outgoing = {p: [] for p in products}")
        code_parts.append("incoming = {p: [] for p in products}")
        code_parts.append("for edge in sub_edges:")
        code_parts.append("    p_from, p_to = edge")
        code_parts.append("    outgoing[p_from].append(p_to)")
        code_parts.append("    incoming[p_to].append(p_from)")
        code_parts.append("")

    code_parts.append("m = gp.Model('reloop')")
    code_parts.append("")

    # Variables
    code_parts.append("I = {}")
    code_parts.append("y = {}")
    code_parts.append("for p in products:")
    code_parts.append("    for l in locations:")
    code_parts.append("        for t in range(1, T + 1):")
    code_parts.append("            for a in range(1, shelf_life[p] + 1):")
    code_parts.append("                I[(p, l, t, a)] = m.addVar(lb=0.0, name=f'I_{p}_{l}_{t}_{a}')")
    code_parts.append("                y[(p, l, t, a)] = m.addVar(lb=0.0, name=f'y_{p}_{l}_{t}_{a}')")
    code_parts.append("")
    code_parts.append("W = {}")
    code_parts.append("Q = {}")
    code_parts.append("L = {}")
    code_parts.append("for p in products:")
    code_parts.append("    for l in locations:")
    code_parts.append("        for t in range(1, T + 1):")
    code_parts.append("            W[(p, l, t)] = m.addVar(lb=0.0, name=f'W_{p}_{l}_{t}')")
    code_parts.append("            Q[(p, l, t)] = m.addVar(lb=0.0, name=f'Q_{p}_{l}_{t}')")
    code_parts.append("            L[(p, l, t)] = m.addVar(lb=0.0, name=f'L_{p}_{l}_{t}')")
    code_parts.append("")

    if features.get("substitution"):
        code_parts.append("S = {}")
        code_parts.append("if sub_edges:")
        code_parts.append("    for edge in sub_edges:")
        code_parts.append("        p_from, p_to = edge")
        code_parts.append("        for l in locations:")
        code_parts.append("            for t in range(1, T + 1):")
        code_parts.append("                S[(p_from, p_to, l, t)] = m.addVar(lb=0.0, name=f'S_{p_from}_{p_to}_{l}_{t}')")
        code_parts.append("")

    if features.get("transshipment"):
        code_parts.append("X = {}")
        code_parts.append("if trans_edges:")
        code_parts.append("    for p in products:")
        code_parts.append("        for edge in trans_edges:")
        code_parts.append("            src, dst = edge")
        code_parts.append("            for t in range(1, T + 1):")
        code_parts.append("                X[(p, src, dst, t)] = m.addVar(lb=0.0, name=f'X_{p}_{src}_{dst}_{t}')")
        code_parts.append("")

    if features.get("moq"):
        code_parts.append("z = {}")
        code_parts.append("if moq > 0:")
        code_parts.append("    for p in products:")
        code_parts.append("        for l in locations:")
        code_parts.append("            for t in range(1, T + 1):")
        code_parts.append("                z[(p, l, t)] = m.addVar(vtype=GRB.BINARY, name=f'z_{p}_{l}_{t}')")
        code_parts.append("")

    # Objective
    code_parts.append("obj = gp.LinExpr()")
    code_parts.append("")
    code_parts.append("for p in products:")
    code_parts.append("    for l in locations:")
    code_parts.append("        for t in range(1, T + 1):")
    code_parts.append("            obj += costs['purchasing'][p] * Q[(p, l, t)]")
    code_parts.append("            obj += costs['waste'][p] * W[(p, l, t)]")
    code_parts.append("            obj += costs['lost_sales'][p] * L[(p, l, t)]")
    code_parts.append("            for a in range(2, shelf_life[p] + 1):")
    code_parts.append("                obj += costs['inventory'][p] * (I[(p, l, t, a)] - y[(p, l, t, a)])")
    code_parts.append("")

    if features.get("moq"):
        code_parts.append("if moq > 0 and 'fixed_order' in costs:")
        code_parts.append("    for p in products:")
        code_parts.append("        for l in locations:")
        code_parts.append("            for t in range(1, T + 1):")
        code_parts.append("                obj += costs['fixed_order'] * z[(p, l, t)]")
        code_parts.append("")

    code_parts.append("m.setObjective(obj, GRB.MINIMIZE)")
    code_parts.append("")

    # Constraints
    code_parts.append("for p in products:")
    code_parts.append("    for l in locations:")
    code_parts.append("        for a in range(1, shelf_life[p]):")
    code_parts.append("            m.addConstr(I[(p, l, 1, a)] == 0, name=f'init_{p}_{l}_{a}')")
    code_parts.append("")

    code_parts.append("for p in products:")
    code_parts.append("    for l in locations:")
    code_parts.append("        sl = shelf_life[p]")
    code_parts.append("        lt = lead_time[p]")
    code_parts.append("        if lt == 0:")
    code_parts.append("            m.addConstr(I[(p, l, 1, sl)] == Q[(p, l, 1)], name=f'fresh_inflow_{p}_{l}_1')")
    code_parts.append("        else:")
    code_parts.append("            m.addConstr(I[(p, l, 1, sl)] == 0, name=f'fresh_inflow_{p}_{l}_1')")
    code_parts.append("")

    code_parts.append("for p in products:")
    code_parts.append("    for l in locations:")
    code_parts.append("        for t in range(2, T + 1):")
    code_parts.append("            sl = shelf_life[p]")
    code_parts.append("            lt = lead_time[p]")
    code_parts.append("            if t > lt:")
    code_parts.append("                t_ord = t - lt")
    code_parts.append("                m.addConstr(I[(p, l, t, sl)] == Q[(p, l, t_ord)], name=f'fresh_inflow_{p}_{l}_{t}')")
    code_parts.append("            else:")
    code_parts.append("                m.addConstr(I[(p, l, t, sl)] == 0, name=f'fresh_inflow_{p}_{l}_{t}')")
    code_parts.append("")

    code_parts.append("for p in products:")
    code_parts.append("    for l in locations:")
    code_parts.append("        for t in range(1, T):")
    code_parts.append("            for a in range(1, shelf_life[p]):")
    code_parts.append("                m.addConstr(I[(p, l, t + 1, a)] == I[(p, l, t, a + 1)] - y[(p, l, t, a + 1)], name=f'aging_{p}_{l}_{t}_{a}')")
    code_parts.append("")

    code_parts.append("for p in products:")
    code_parts.append("    for l in locations:")
    code_parts.append("        for t in range(1, T + 1):")
    code_parts.append("            m.addConstr(W[(p, l, t)] == I[(p, l, t, 1)] - y[(p, l, t, 1)], name=f'expire_{p}_{l}_{t}')")
    code_parts.append("")

    code_parts.append("for p in products:")
    code_parts.append("    for l in locations:")
    code_parts.append("        for t in range(1, T + 1):")
    code_parts.append("            for a in range(1, shelf_life[p] + 1):")
    code_parts.append("                m.addConstr(y[(p, l, t, a)] <= I[(p, l, t, a)], name=f'avail_{p}_{l}_{t}_{a}')")
    code_parts.append("")

    code_parts.append("for p in products:")
    code_parts.append("    for l in locations:")
    code_parts.append("        for t in range(1, T + 1):")
    code_parts.append("            demand = demand_curve[p][t - 1] * demand_share[l]")
    code_parts.append("            sales_sum = gp.quicksum(y[(p, l, t, a)] for a in range(1, shelf_life[p] + 1))")

    if features.get("substitution"):
        code_parts.append("            inbound = gp.quicksum(S[(pf, p, l, t)] for pf in incoming.get(p, []) if (pf, p, l, t) in S)")
        code_parts.append("            outbound = gp.quicksum(S[(p, pt, l, t)] for pt in outgoing.get(p, []) if (p, pt, l, t) in S)")
        code_parts.append("            m.addConstr(sales_sum + L[(p, l, t)] == demand + inbound - outbound, name=f'sales_conservation_{p}_{l}_{t}')")
    else:
        code_parts.append("            m.addConstr(sales_sum + L[(p, l, t)] == demand, name=f'sales_conservation_{p}_{l}_{t}')")
    code_parts.append("")

    if features.get("substitution"):
        code_parts.append("if sub_edges:")
        code_parts.append("    for p in products:")
        code_parts.append("        if outgoing.get(p):")
        code_parts.append("            for l in locations:")
        code_parts.append("                for t in range(1, T + 1):")
        code_parts.append("                    demand = demand_curve[p][t - 1] * demand_share[l]")
        code_parts.append("                    outbound = gp.quicksum(S[(p, pt, l, t)] for pt in outgoing[p] if (p, pt, l, t) in S)")
        code_parts.append("                    m.addConstr(outbound <= demand, name=f'demand_route_{p}_{l}_{t}')")
        code_parts.append("")

    code_parts.append("for l in locations:")
    code_parts.append("    for t in range(1, T + 1):")
    code_parts.append("        inv_sum = gp.quicksum(cold_usage[p] * I[(p, l, t, a)] for p in products for a in range(1, shelf_life[p] + 1))")
    code_parts.append("        m.addConstr(inv_sum <= cold_capacity[l], name=f'storage_cap_{l}_{t}')")
    code_parts.append("")

    code_parts.append("for p in products:")
    code_parts.append("    for t in range(1, T + 1):")
    code_parts.append("        prod_sum = gp.quicksum(Q[(p, l, t)] for l in locations)")
    code_parts.append("        m.addConstr(prod_sum <= production_cap[p][t - 1], name=f'prod_cap_{p}_{t}')")
    code_parts.append("")

    if features.get("moq"):
        code_parts.append("if moq > 0:")
        code_parts.append("    bigM = max(production_cap[p][t] for p in products for t in range(len(production_cap[products[0]])))")
        code_parts.append("    for p in products:")
        code_parts.append("        for l in locations:")
        code_parts.append("            for t in range(1, T + 1):")
        code_parts.append("                m.addConstr(Q[(p, l, t)] >= moq * z[(p, l, t)], name=f'moq_lb_{p}_{l}_{t}')")
        code_parts.append("                m.addConstr(Q[(p, l, t)] <= bigM * z[(p, l, t)], name=f'moq_ub_{p}_{l}_{t}')")
        code_parts.append("")

    if features.get("labor"):
        code_parts.append("if labor_usage and labor_cap:")
        code_parts.append("    for l in locations:")
        code_parts.append("        for t in range(1, T + 1):")
        code_parts.append("            labor_sum = gp.quicksum(labor_usage.get(p, 0) * y[(p, l, t, a)] for p in products for a in range(1, shelf_life[p] + 1))")
        code_parts.append("            m.addConstr(labor_sum <= labor_cap[l][t - 1], name=f'labor_cap_{l}_{t}')")
        code_parts.append("")

    code_parts.append("m.Params.OutputFlag = 0")
    code_parts.append("m.Params.Threads = 1")
    code_parts.append("m.Params.Seed = 0")
    code_parts.append("m.optimize()")
    code_parts.append("")
    code_parts.append("print(f'Status: {m.Status}')")
    code_parts.append("if m.Status == GRB.OPTIMAL:")
    code_parts.append("    print(f'Objective: {m.ObjVal:.2f}')")

    return "\n".join(code_parts)


# Repair sample generators
JSON_REPAIR_INSTRUCTIONS = [
    "Re-output as valid JSON only. No prose, no markdown.",
    "Fix the JSON syntax error. Output valid JSON only.",
    "The previous output was not valid JSON. Correct it.",
]

JSON_ERROR_TYPES = [
    {"type": "trailing_comma", "description": "Trailing comma in array or object"},
    {"type": "single_quotes", "description": "Single quotes instead of double quotes"},
    {"type": "unquoted_key", "description": "Unquoted object key"},
    {"type": "missing_comma", "description": "Missing comma between elements"},
    {"type": "markdown_wrapper", "description": "JSON wrapped in markdown code block"},
]

RUNTIME_REPAIR_INSTRUCTIONS = [
    "Fix the runtime error and output corrected Python code only.",
    "The code raised an exception. Fix and re-output.",
]

RUNTIME_ERROR_TYPES = [
    {"type": "key_error_sub_edges", "error": "KeyError: 'sub_edges'", "cause": "Direct access instead of nested", "fix": "Use data.get('network', {}).get('sub_edges', [])"},
    {"type": "index_error_demand", "error": "IndexError: list index out of range", "cause": "Using t instead of t-1", "fix": "Use demand_curve[p][t-1]"},
]

PROBE_FAILURE_TEMPLATES = [
    {"probe": "initialization", "failure": "Non-fresh buckets not initialized to zero at t=1", "cause": "Missing init constraint", "fix": "For a < shelf_life[p]: I[p,l,1,a] = 0"},
    {"probe": "holding_cost", "failure": "Holding cost applied to I instead of (I - y)", "cause": "Wrong formula", "fix": "Use costs['inventory'] * (I[p,l,t,a] - y[p,l,t,a])"},
    {"probe": "lost_sales_slack", "failure": "L variable not in sales_conservation", "cause": "Missing L", "fix": "sales_conservation: sum(y) + L = demand"},
    {"probe": "aging_dynamics", "failure": "Aging constraint incorrect or missing", "cause": "Created for t=T", "fix": "Only create for t < T"},
]


def generate_step05_sample(scenario: dict) -> dict:
    error_type = random.choice(JSON_ERROR_TYPES)
    instruction = random.choice(JSON_REPAIR_INSTRUCTIONS)

    original_json = {"optimize": "Minimize total cost", "controls": ["order quantities", "inventory allocation"], "summary": "Supply chain optimization"}

    if error_type["type"] == "trailing_comma":
        malformed = '{"optimize": "Minimize total cost", "controls": ["order quantities",], "summary": "Test"}'
    elif error_type["type"] == "single_quotes":
        malformed = "{'optimize': 'Minimize total cost', 'controls': ['order quantities']}"
    elif error_type["type"] == "unquoted_key":
        malformed = '{optimize: "Minimize total cost", controls: ["order quantities"]}'
    elif error_type["type"] == "missing_comma":
        malformed = '{"optimize": "Minimize total cost" "controls": ["order quantities"]}'
    else:
        malformed = '```json\n{"optimize": "Minimize total cost", "controls": ["order quantities"]}\n```'

    corrected = json.dumps(original_json, indent=2)

    return {
        "messages": [
            {"role": "system", "content": GLOBAL_GUARDRAILS},
            {"role": "user", "content": f"FORMAT REPAIR — JSON\n\nYour output was not valid JSON.\n\nTask: {instruction}\n\nRules:\n- No prose, no markdown.\n- Double quotes for all strings.\n- No trailing commas.\n\nYour previous (invalid) output:\n{malformed}"},
            {"role": "assistant", "content": corrected}
        ],
        "metadata": {"scenario_id": f"json_repair_{scenario['scenario_id']}", "step": "step05_json_repair", "error_type": error_type["type"]}
    }


def generate_step06_sample(scenario: dict) -> dict:
    error_info = random.choice(RUNTIME_ERROR_TYPES)
    instruction = random.choice(RUNTIME_REPAIR_INSTRUCTIONS)

    if error_info["type"] == "key_error_sub_edges":
        buggy_code = "sub_edges = data['sub_edges']  # BUG"
        fixed_code = "sub_edges = data.get('network', {}).get('sub_edges', [])"
    else:
        buggy_code = "demand = demand_curve[p][t]  # BUG"
        fixed_code = "demand = demand_curve[p][t - 1]"

    return {
        "messages": [
            {"role": "system", "content": GLOBAL_GUARDRAILS},
            {"role": "user", "content": f"STEP 6 - RUNTIME ERROR REPAIR\n\n{instruction}\n\nRuntime error:\n{error_info['error']}\n\nCause: {error_info['cause']}\n\nPrevious code:\n{buggy_code}"},
            {"role": "assistant", "content": fixed_code}
        ],
        "metadata": {"scenario_id": f"runtime_repair_{scenario['scenario_id']}", "step": "step06_runtime_repair", "error_type": error_info["type"]}
    }


def generate_step07_sample(scenario: dict) -> dict:
    probe = random.choice(PROBE_FAILURE_TEMPLATES)

    if probe["probe"] == "initialization":
        buggy_snippet = "# Missing initialization constraint"
        fixed_snippet = "for p in products:\n    for l in locations:\n        for a in range(1, shelf_life[p]):\n            m.addConstr(I[(p, l, 1, a)] == 0, name=f'init_{p}_{l}_{a}')"
    elif probe["probe"] == "holding_cost":
        buggy_snippet = "obj += costs['inventory'][p] * I[(p, l, t, a)]  # BUG"
        fixed_snippet = "obj += costs['inventory'][p] * (I[(p, l, t, a)] - y[(p, l, t, a)])"
    elif probe["probe"] == "lost_sales_slack":
        buggy_snippet = "m.addConstr(sales_sum == demand)  # BUG: no L"
        fixed_snippet = "m.addConstr(sales_sum + L[(p, l, t)] == demand)"
    else:
        buggy_snippet = "for t in range(1, T + 1):  # BUG: includes T"
        fixed_snippet = "for t in range(1, T):  # FIXED: excludes T"

    return {
        "messages": [
            {"role": "system", "content": GLOBAL_GUARDRAILS},
            {"role": "user", "content": f"STEP 7 - AUDIT/PROBE REPAIR\n\nSemantic probe failure:\n- Probe: {probe['probe']}\n- Failure: {probe['failure']}\n- Cause: {probe['cause']}\n\nRelevant code:\n{buggy_snippet}"},
            {"role": "assistant", "content": fixed_snippet}
        ],
        "metadata": {"scenario_id": f"probe_repair_{scenario['scenario_id']}", "step": "step07_probe_repair", "probe": probe["probe"]}
    }


STEP1_PROMPT = "STEP 1 — LOCK THE TASK CONTRACT\n\nTask: Based on scenario narrative and data_profile, answer:\n1) What are we optimizing?\n2) What can we control?\n3) What are hard constraints?\n4) What are soft violations (penalized)?\n\nOutput: Single JSON object, no surrounding text."
STEP2_PROMPT = "STEP 2 - BUILD THE MODEL SPEC SHEET\n\nTask: Produce a structured model specification based on Step1 contract and data_profile.\n\nOutput: Single JSON object. No surrounding text, no markdown."
STEP3_PROMPT = "STEP 3 - CONSTRAINT TEMPLATES\n\nTask: Using spec_sheet, produce constraint templates as a JSON array.\n\nOutput: Single JSON array. No surrounding text, no markdown."
STEP4_PROMPT = "STEP 4 - GENERATE GUROBIPY SCRIPT\n\nOutput ONLY raw Python code. No markdown, no comments.\nFirst character must be: import"


def generate_training_sample(scenario: dict, step: int) -> dict:
    data = scenario["data_profile"]

    if step == 1:
        return {
            "messages": [
                {"role": "system", "content": GLOBAL_GUARDRAILS},
                {"role": "user", "content": f"{STEP1_PROMPT}\n\nScenario:\n{scenario['narrative']}\n\nData profile:\n{json.dumps(data, indent=2)}"},
                {"role": "assistant", "content": generate_step1_response(scenario)}
            ],
            "metadata": {"scenario_id": scenario["scenario_id"], "step": "step1", "industry": scenario["industry"]}
        }
    elif step == 2:
        step1_response = generate_step1_response(scenario)
        return {
            "messages": [
                {"role": "system", "content": GLOBAL_GUARDRAILS},
                {"role": "user", "content": f"{STEP2_PROMPT}\n\nStep 1 Contract:\n{step1_response}\n\nData profile:\n{json.dumps(data, indent=2)}"},
                {"role": "assistant", "content": generate_step2_response(scenario)}
            ],
            "metadata": {"scenario_id": scenario["scenario_id"], "step": "step2", "industry": scenario["industry"]}
        }
    elif step == 3:
        step2_response = generate_step2_response(scenario)
        return {
            "messages": [
                {"role": "system", "content": GLOBAL_GUARDRAILS},
                {"role": "user", "content": f"{STEP3_PROMPT}\n\nSpec Sheet:\n{step2_response}"},
                {"role": "assistant", "content": generate_step3_response(scenario)}
            ],
            "metadata": {"scenario_id": scenario["scenario_id"], "step": "step3", "industry": scenario["industry"]}
        }
    elif step == 4:
        step3_response = generate_step3_response(scenario)
        return {
            "messages": [
                {"role": "system", "content": GLOBAL_GUARDRAILS},
                {"role": "user", "content": f"{STEP4_PROMPT}\n\nConstraint Templates:\n{step3_response}\n\nData profile:\n{json.dumps(data, indent=2)}"},
                {"role": "assistant", "content": generate_step4_response(scenario)}
            ],
            "metadata": {"scenario_id": scenario["scenario_id"], "step": "step4", "industry": scenario["industry"]}
        }
    else:
        raise ValueError(f"Unknown step: {step}")


def generate_all_training_data(num_scenarios: int = 2500, output_dir: str = "train_set"):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    industries = list(INDUSTRIES.keys())
    scenarios = []
    for i in range(num_scenarios):
        industry = industries[i % len(industries)]
        features = FEATURE_SETS[i % len(FEATURE_SETS)]
        scenario = generate_scenario(i, industry, features)
        scenarios.append(scenario)

    step_files = {1: [], 2: [], 3: [], 4: []}

    print(f"Generating {num_scenarios} scenarios across {len(industries)} industries...")

    for scenario in scenarios:
        for step in [1, 2, 3, 4]:
            sample = generate_training_sample(scenario, step)
            step_files[step].append(sample)

    step_names = {1: "contract", 2: "spec_sheet", 3: "templates", 4: "codegen"}
    for step, samples in step_files.items():
        filepath = output_path / f"v2_step{step}_{step_names[step]}.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"  Step {step}: {len(samples)} samples -> {filepath}")

    repair_samples = {"step05": [], "step06": [], "step07": []}

    num_json_repairs = int(num_scenarios * 0.4)
    for i in range(num_json_repairs):
        sample = generate_step05_sample(scenarios[i % len(scenarios)])
        repair_samples["step05"].append(sample)

    num_runtime_repairs = int(num_scenarios * 0.25)
    for i in range(num_runtime_repairs):
        sample = generate_step06_sample(scenarios[i % len(scenarios)])
        repair_samples["step06"].append(sample)

    for i in range(num_scenarios):
        sample = generate_step07_sample(scenarios[i % len(scenarios)])
        repair_samples["step07"].append(sample)

    repair_files = {"step05": "v2_step05_repair_json.jsonl", "step06": "v2_step06_repair_runtime.jsonl", "step07": "v2_step07_repair_probe.jsonl"}
    for step_key, fname in repair_files.items():
        filepath = output_path / fname
        with open(filepath, "w", encoding="utf-8") as f:
            for sample in repair_samples[step_key]:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"  {step_key}: {len(repair_samples[step_key])} samples -> {filepath}")

    all_samples = []
    for step in [1, 2, 3, 4]:
        all_samples.extend(step_files[step])
    for step_key in ["step05", "step06", "step07"]:
        all_samples.extend(repair_samples[step_key])

    combined_path = output_path / "v2_train_all.jsonl"
    with open(combined_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_samples)} samples -> {combined_path}")

    return all_samples


if __name__ == "__main__":
    generate_all_training_data(num_scenarios=2500)
