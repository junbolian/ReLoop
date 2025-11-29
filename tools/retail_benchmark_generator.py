# ==============================================================================
# FILE: retail_benchmark_generator.py
# LOCATION: reloop/tools/
#
# DESCRIPTION:
#   Generates the "Comprehensive Retail Supply Chain" benchmark instances.
#   Each structural archetype is expanded into several perturbed variations.
# ==============================================================================

import json
import os
import copy
import numpy as np
from pathlib import Path

# ==========================================
# 0. Base Retail Configuration
# ==========================================
def get_base_scenario():
    periods = 20
    t = np.arange(periods)
    base_curve = 1000 * np.exp(-(t - 10)**2 / (2 * 3**2)) + 300 
    
    return {
        "name": "retail_base",
        "description": "Standard seasonal retail scenario.",
        "periods": periods,
        "products": ["SKU_Basic", "SKU_Premium", "SKU_ShortLife"],
        "locations": ["DC1", "DC2", "DC3", "DC4", "DC5"],
        "shelf_life": {"SKU_Basic": 10, "SKU_Premium": 8, "SKU_ShortLife": 4},
        "lead_time": {"SKU_Basic": 0, "SKU_Premium": 0, "SKU_ShortLife": 0}, 
        "cold_capacity": {
            "DC1": 4000, "DC2": 3500, "DC3": 3000, "DC4": 3000, "DC5": 2500
        },
        "cold_usage": {"SKU_Basic": 1.0, "SKU_Premium": 3.0, "SKU_ShortLife": 1.2},
        "production_cap": {
            "SKU_Basic": [800] * periods,
            "SKU_Premium": [400] * periods,
            "SKU_ShortLife": [500] * periods
        },
        "labor_cap": {l: [99999.0] * periods for l in ["DC1", "DC2", "DC3", "DC4", "DC5"]},
        "labor_usage": {"SKU_Basic": 0.0, "SKU_Premium": 0.0, "SKU_ShortLife": 0.0}, 
        "return_rate": {"SKU_Basic": 0.0, "SKU_Premium": 0.0, "SKU_ShortLife": 0.0}, 
        "demand_curve": {
            "SKU_Basic": [int(x) for x in base_curve],
            "SKU_Premium": [int(x * 0.5) for x in base_curve],
            "SKU_ShortLife": [int(x * 0.4) for x in base_curve]
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
            "budget_per_period": None,
            "waste_limit_pct": None  
        },
        "network": {
            "sub_edges": [["SKU_Basic", "SKU_Premium"]],
            "trans_edges": []               
        }
    }

# ==========================================
# Family Definitions
# ==========================================

def inst_01_base(data):
    data["name"] = "retail_f1_base"
    return data

def inst_02_high_waste_cost(data): 
    data["name"] = "retail_f1_high_waste"
    for p in data["products"]:
        data["costs"]["waste"][p] *= 20.0
    return data

def inst_03_high_holding_cost(data):
    data["name"] = "retail_f1_jit_logic"
    for p in data["products"]:
        data["costs"]["inventory"][p] *= 20.0
    return data

def inst_04_long_horizon(data):
    data["name"] = "retail_f1_52_weeks"
    data["periods"] = 52
    for p in data["products"]:
        data["production_cap"][p] = (data["production_cap"][p] * 3)[:52]
        data["demand_curve"][p] = (data["demand_curve"][p] * 3)[:52]
    for l in data["locations"]:
        data["labor_cap"][l] = (data["labor_cap"][l] * 3)[:52]
    return data

def inst_05_no_sub(data):
    data["name"] = "retail_f2_no_substitution"
    data["network"]["sub_edges"] = []
    return data

def inst_06_deep_sub(data):
    data["name"] = "retail_f2_circular_sub"
    data["network"]["sub_edges"] = [
        ["SKU_Basic", "SKU_Premium"],
        ["SKU_Premium", "SKU_ShortLife"],
        ["SKU_ShortLife", "SKU_Basic"]
    ]
    return data

def inst_07_cannibalization(data):
    data["name"] = "retail_f2_cannibalization"
    data["demand_curve"]["SKU_Basic"] = [
        int(x * 2.0) for x in data["demand_curve"]["SKU_Basic"]
    ]
    data["costs"]["lost_sales"]["SKU_Basic"] = 5.0
    for l in data["locations"]:
        data["cold_capacity"][l] *= 0.5
    return data

def inst_08_short_life(data):
    data["name"] = "retail_f2_ultra_fresh"
    data["shelf_life"] = {"SKU_Basic": 2, "SKU_Premium": 2, "SKU_ShortLife": 1}
    return data

def inst_09_cold_tight(data):
    data["name"] = "retail_f3_storage_bottleneck"
    for l in data["locations"]:
        data["cold_capacity"][l] *= 0.3
    return data

def inst_10_heavy_item(data):
    data["name"] = "retail_f3_volumetric_constraint"
    data["cold_usage"]["SKU_Premium"] = 15.0 
    return data

def inst_11_production_bottleneck(data):
    data["name"] = "retail_f3_supply_bottleneck"
    for l in data["locations"]:
        data["cold_capacity"][l] = 999999.0
    for p in data["products"]:
        data["production_cap"][p] = [
            x * 0.3 for x in data["production_cap"][p]
        ]
    return data

def inst_12_asymmetric(data):
    data["name"] = "retail_f3_unbalanced_network"
    total_cap = sum(data["cold_capacity"].values())
    data["cold_capacity"] = {l: total_cap * 0.01 for l in data["locations"]}
    data["cold_capacity"]["DC1"] = total_cap * 0.96
    return data

def inst_13_early_disruption(data):
    data["name"] = "retail_f4_early_stockout"
    for p in data["products"]:
        for t in range(5):
            data["production_cap"][p][t] = 0
    return data

def inst_14_peak_disruption(data):
    data["name"] = "retail_f4_peak_failure"
    for p in data["products"]:
        for t in range(8, 12):
            data["production_cap"][p][t] = 0
    return data

def inst_15_demand_spike(data):
    data["name"] = "retail_f4_demand_surge"
    for p in data["products"]:
        if len(data["demand_curve"][p]) > 14:
            data["demand_curve"][p][14] *= 4
    return data

def inst_16_quality_hold(data):
    data["name"] = "retail_f4_quality_hold"
    for t in range(10, data["periods"]):
        data["production_cap"]["SKU_Basic"][t] = 0
    return data

def inst_17_impossible_demand(data):
    data["name"] = "retail_f5_impossible_demand"
    for p in data["products"]:
        data["demand_curve"][p] = [
            int(x * 5.0) for x in data["demand_curve"][p]
        ]
    return data

def inst_18_strict_service(data):
    data["name"] = "retail_f5_strict_service_trap"
    for l in data["locations"]:
        data["cold_capacity"][l] *= 0.1
    return data

def inst_19_storage_overflow(data):
    data["name"] = "retail_f5_storage_overflow"
    for l in data["locations"]:
        data["cold_capacity"][l] = 0.5
    return data

# Stress-test composition: cold bottleneck + peak disruption + no substitution.
def inst_20_stress_test(data):
    data = inst_09_cold_tight(data)
    data = inst_14_peak_disruption(data)
    data = inst_05_no_sub(data)
    data["name"] = "retail_f5_ultimate_stress"
    return data

def inst_21_lead_time(data):
    data["name"] = "retail_f6_lead_time"
    data["lead_time"] = {"SKU_Basic": 3, "SKU_Premium": 4, "SKU_ShortLife": 2}
    return data

def inst_22_moq(data):
    data["name"] = "retail_f6_moq_binary"
    data["constraints"]["moq"] = 300
    return data

def inst_23_fixed_cost(data):
    data["name"] = "retail_f6_fixed_order_cost"
    data["costs"]["fixed_order"] = 5000.0
    return data

def inst_24_pack_size(data):
    data["name"] = "retail_f6_pack_size_integer"
    data["constraints"]["pack_size"] = 100
    return data

def inst_25_transshipment(data):
    data["name"] = "retail_f7_transshipment"
    locs = data["locations"]
    edges = []
    for l1 in locs:
        for l2 in locs:
            if l1 != l2:
                edges.append([l1, l2])
    data["network"]["trans_edges"] = edges
    return data

def inst_26_hub_and_spoke(data):
    data["name"] = "retail_f7_hub_and_spoke"
    data["cold_capacity"]["DC1"] = 50000.0
    for l in ["DC2", "DC3", "DC4", "DC5"]:
        data["cold_capacity"][l] = 500.0
    data["network"]["trans_edges"] = [
        ["DC1", "DC2"],
        ["DC1", "DC3"],
        ["DC1", "DC4"],
        ["DC1", "DC5"],
    ]
    return data

def inst_27_budget_constraint(data):
    data["name"] = "retail_f7_budget_limit"
    data["constraints"]["budget_per_period"] = 10000.0
    return data

def inst_28_multi_sourcing(data):
    data["name"] = "retail_f7_multi_sourcing"
    data["lead_time"] = {"SKU_Basic": 5, "SKU_Premium": 0, "SKU_ShortLife": 1}
    data["costs"]["inventory"]["SKU_Basic"] = 0.5 
    data["costs"]["inventory"]["SKU_Premium"] = 10.0 
    return data

def inst_29_reverse_logistics(data):
    data["name"] = "retail_f8_reverse_logistics"
    data["return_rate"] = {"SKU_Basic": 0.2, "SKU_Premium": 0.1, "SKU_ShortLife": 0.05}
    return data

def inst_30_labor_constraint(data):
    data["name"] = "retail_f8_labor_constraint"
    for l in data["locations"]:
        data["labor_cap"][l] = [200.0] * data["periods"]
    data["labor_usage"] = {"SKU_Basic": 0.1, "SKU_Premium": 0.2, "SKU_ShortLife": 0.1}
    return data

def inst_31_omni_ship_from_store(data):
    data["name"] = "retail_f8_ship_from_store"
    data["labor_usage"] = {"SKU_Basic": 0.5, "SKU_Premium": 0.8, "SKU_ShortLife": 0.6}
    for l in data["locations"]:
        data["cold_capacity"][l] *= 5.0
        data["labor_cap"][l] = [500.0] * data["periods"]
    return data

def inst_32_sustainability(data):
    data["name"] = "retail_f8_sustainability"
    data["constraints"]["waste_limit_pct"] = 0.02 
    return data

def inst_33_price_band_tight(data):
    """
    Family 2: price band and margin trade-offs.
    Premium has higher margin and stricter service, basic is cheaper but more elastic.
    """
    data["name"] = "retail_f2_price_band_tight"
    # Adjust purchasing costs to emulate different price bands / margins.
    data["costs"]["purchasing"]["SKU_Premium"] *= 0.8
    data["costs"]["purchasing"]["SKU_Basic"] *= 1.1
    # Premium lost sales become very expensive (service-level commitment).
    data["costs"]["lost_sales"]["SKU_Premium"] *= 2.0
    # Keep one-way substitution from Basic to Premium.
    data["network"]["sub_edges"] = [["SKU_Basic", "SKU_Premium"]]
    return data

def inst_34_promo_budget(data):
    """
    Family 2: promotion window with demand lift and budget cap.
    Last few periods have strong promotions on Basic and ShortLife.
    """
    data["name"] = "retail_f2_promo_budget"
    T = data["periods"]
    promo_horizon = min(4, T)
    for p in ["SKU_Basic", "SKU_ShortLife"]:
        for t in range(T - promo_horizon, T):
            data["demand_curve"][p][t] = int(data["demand_curve"][p][t] * 2.0)
    # Add an open-to-buy style budget constraint to couple periods.
    data["constraints"]["budget_per_period"] = 15000.0
    return data

def inst_35_robust_demand_variance(data):
    """
    Family 4: high variance demand to emulate robust planning.
    Alternating high and low periods; lost sales are very expensive.
    """
    data["name"] = "retail_f4_robust_variance"
    for p in data["products"]:
        curve = np.array(data["demand_curve"][p], dtype=float)
        for idx in range(len(curve)):
            if idx % 2 == 0:
                curve[idx] *= 1.5
            else:
                curve[idx] *= 0.7
        data["demand_curve"][p] = [int(x) for x in curve]
        data["costs"]["lost_sales"][p] *= 2.5
    return data

def inst_36_robust_supply_risk(data):
    """
    Family 4: mild but persistent supply risk plus expensive waste.
    Encourages balancing safety stock against obsolescence.
    """
    data["name"] = "retail_f4_supply_risk"
    T = data["periods"]
    # Reduce production capacity mid-season.
    mid_start = max(3, T // 3)
    mid_end = min(T, mid_start + 4)
    for p in data["products"]:
        for t in range(mid_start, mid_end):
            data["production_cap"][p][t] = int(
                data["production_cap"][p][t] * 0.4
            )
        # Make waste more expensive so that overstocking is not trivial.
        data["costs"]["waste"][p] *= 3.0
    return data

def inst_37_multiechelon_chain(data):
    """
    Family 7: three-echelon Plant–DC–Store network with explicit topology.
    Demand only occurs at stores; Plant and DCs act as upstream nodes.
    """
    data["name"] = "retail_f7_multiechelon_chain"
    periods = data["periods"]
    data["locations"] = ["Plant", "DC1", "DC2", "Store1", "Store2", "Store3"]
    data["cold_capacity"] = {
        "Plant": 8000.0,
        "DC1": 4000.0,
        "DC2": 4000.0,
        "Store1": 600.0,
        "Store2": 600.0,
        "Store3": 600.0,
    }
    data["labor_cap"] = {
        "Plant": [99999.0] * periods,
        "DC1": [500.0] * periods,
        "DC2": [500.0] * periods,
        "Store1": [200.0] * periods,
        "Store2": [200.0] * periods,
        "Store3": [200.0] * periods,
    }
    # Demand is concentrated at stores; upstream nodes have zero retail demand.
    data["demand_share"] = {
        "Plant": 0.0,
        "DC1": 0.0,
        "DC2": 0.0,
        "Store1": 0.3,
        "Store2": 0.4,
        "Store3": 0.3,
    }
    data["network"]["trans_edges"] = [
        ["Plant", "DC1"],
        ["Plant", "DC2"],
        ["DC1", "Store1"],
        ["DC1", "Store2"],
        ["DC2", "Store2"],
        ["DC2", "Store3"],
    ]
    return data

def inst_38_ring_routing(data):
    """
    Family 7: ring-style routing proxy.
    Locations are connected in a cycle, mimicking a truck route.
    """
    data["name"] = "retail_f7_ring_routing"
    locs = data["locations"]
    edges = []
    n = len(locs)
    for i in range(n):
        l1 = locs[i]
        l2 = locs[(i + 1) % n]
        edges.append([l1, l2])
    data["network"]["trans_edges"] = edges
    # Slightly tighten capacity to make the cyclic structure matter.
    for l in data["locations"]:
        data["cold_capacity"][l] *= 0.8
    return data

# ==========================================
# Perturbation
# ==========================================
def perturb_data(data, seed, intensity=0.15):
    rng = np.random.default_rng(seed)
    new_data = copy.deepcopy(data)
    
    # Demand perturbation.
    for p in new_data["products"]:
        original = np.array(new_data["demand_curve"][p], dtype=float)
        noise = rng.uniform(1.0 - intensity, 1.0 + intensity, size=len(original))
        new_data["demand_curve"][p] = [int(x) for x in original * noise]
        
    # Capacity perturbation.
    for l in new_data["locations"]:
        val = float(new_data["cold_capacity"][l])
        new_data["cold_capacity"][l] = val * rng.uniform(
            1.0 - intensity, 1.0 + intensity
        )
        
    return new_data

# ==========================================
# Main
# ==========================================
def main():
    generators = [
        inst_01_base,
        inst_02_high_waste_cost,
        inst_03_high_holding_cost,
        inst_04_long_horizon,
        inst_05_no_sub,
        inst_06_deep_sub,
        inst_07_cannibalization,
        inst_08_short_life,
        inst_09_cold_tight,
        inst_10_heavy_item,
        inst_11_production_bottleneck,
        inst_12_asymmetric,
        inst_13_early_disruption,
        inst_14_peak_disruption,
        inst_15_demand_spike,
        inst_16_quality_hold,
        inst_17_impossible_demand,
        inst_18_strict_service,
        inst_19_storage_overflow,
        inst_20_stress_test,
        inst_21_lead_time,
        inst_22_moq,
        inst_23_fixed_cost,
        inst_24_pack_size,
        inst_25_transshipment,
        inst_26_hub_and_spoke,
        inst_27_budget_constraint,
        inst_28_multi_sourcing,
        inst_29_reverse_logistics,
        inst_30_labor_constraint,
        inst_31_omni_ship_from_store,
        inst_32_sustainability,
        inst_33_price_band_tight,
        inst_34_promo_budget,
        inst_35_robust_demand_variance,
        inst_36_robust_supply_risk,
        inst_37_multiechelon_chain,
        inst_38_ring_routing,
    ]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(
        script_dir, "..", "scenarios", "retail_comprehensive", "data"
    )
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    NUM_VARIATIONS = 5
    total_instances = len(generators) * NUM_VARIATIONS
    print(f"Generating {total_instances} instances into:\n{output_dir}")

    for gen_func in generators:
        base_struct = gen_func(get_base_scenario())
        base_name = base_struct["name"]
        
        for v in range(NUM_VARIATIONS):
            current_data = gen_func(get_base_scenario())
            if v == 0:
                final_name = f"{base_name}_v0"
                final_data = current_data
            else:
                seed = (hash(base_name) + v) % (2**32)
                final_data = perturb_data(current_data, seed=seed)
                final_name = f"{base_name}_v{v}"
            
            final_data["name"] = final_name
            with open(os.path.join(output_dir, f"{final_name}.json"), "w") as f:
                json.dump(final_data, f, indent=2)
            
    print("Generation complete.")

if __name__ == "__main__":
    main()
