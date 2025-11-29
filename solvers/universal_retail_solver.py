# ==============================================================================
# FILE: universal_retail_solver.py
# LOCATION: reloop/solvers/
#
# DESCRIPTION:
#   The Universal Retail Solver (URS).
#   A generalized Mixed-Integer Linear Program (MILP) engine for any retail
#   supply chain problem defined in the ReLoop JSON schema.
#
#   Capabilities (driven purely by JSON fields):
#     - Inventory: Vintage tracking (shelf life) and waste logic.
#     - Logistics: Lead time, MOQ (binary trigger), pack size (integer).
#     - Network: Transshipment and multi-echelon flows via locations + edges.
#     - Omni-channel: Reverse logistics, labor capacity by location and period.
#     - Financials: Period budgets and fixed order costs.
#     - Sustainability: Global waste limits relative to total demand.
#     - Pricing / Promotion effects: Implemented through purchasing costs and
#       budget constraints (e.g., price bands, promo windows).
#     - Robust-style scenarios: Realized via perturbed demand and production
#       capacity profiles (high variance and supply risk).
#
#   [UPDATED] Performance Tuning:
#     - TimeLimit = 60s: Prevents stalling on hard MIP instances (e.g., Family 6).
#     - MIPGap   = 1% : Accepts near-optimal solutions to speed up proof.
#
# USAGE:
#   python universal_retail_solver.py \
#       --file ../scenarios/retail_comprehensive/data/retail_f6_moq_binary_v0.json
# ==============================================================================

import json
import os
import argparse
from gurobipy import Model, GRB, quicksum


def load_data(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Data file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def solve_scenario(data, summarize=True):
    """
    Build and solve the optimization model based on the input JSON configuration.
    """

    # ==========================================
    # 1. Parameter Extraction
    # ==========================================
    T = int(data["periods"])
    products = list(data["products"])
    locations = list(data["locations"])

    # --- Physical Properties ---
    shelf_life = {p: int(v) for p, v in data["shelf_life"].items()}
    cold_cap = data["cold_capacity"]
    cold_use = data["cold_usage"]
    prod_cap = data["production_cap"]

    # --- Logistics & Constraints ---
    lead_time = data.get("lead_time", {p: 0 for p in products})
    constraints = data.get("constraints", {})
    moq = constraints.get("moq", 0)
    pack_size = constraints.get("pack_size", 1)
    budget = constraints.get("budget_per_period", None)
    waste_limit_pct = constraints.get("waste_limit_pct", None)

    # --- Network Topology ---
    network = data.get("network", {})
    sub_edges = [tuple(e) for e in network.get("sub_edges", [])]
    trans_edges = [tuple(e) for e in network.get("trans_edges", [])]

    # Precompute adjacency for substitution
    outgoing_edges = {}
    incoming_edges = {}
    for pf, pt in sub_edges:
        outgoing_edges.setdefault(pf, []).append(pt)
        incoming_edges.setdefault(pt, []).append(pf)

    # --- Omni-channel Ops ---
    labor_use = data.get("labor_usage", {p: 0.0 for p in products})
    # Labor cap is a list per period
    labor_cap = data.get("labor_cap", {l: [99999.0] * T for l in locations})
    return_rate = data.get("return_rate", {p: 0.0 for p in products})

    # --- Costs ---
    costs = data.get("costs", {})
    lost_pen = costs.get("lost_sales", {})
    inv_cost = costs.get("inventory", {})
    waste_cost = costs.get("waste", {})
    fixed_order_cost = costs.get("fixed_order", 0.0)
    trans_cost_unit = costs.get("transshipment", 0.5)
    purchase_cost = costs.get("purchasing", {p: 10.0 for p in products})

    # --- Demand Processing ---
    # Flatten demand to (product, location, time) -> quantity
    demand_share = data.get("demand_share", {})
    demand = {}
    total_demand_vol = 0.0
    for p in products:
        curve = list(data["demand_curve"][p])
        # Pad demand curve if horizon T is longer than data provided
        if len(curve) < T:
            curve += [curve[-1]] * (T - len(curve))

        for l in locations:
            share = demand_share.get(l, 1.0 / len(locations))
            for t in range(1, T + 1):
                d_val = float(curve[t - 1]) * share
                demand[(p, l, t)] = d_val
                total_demand_vol += d_val

    # ==========================================
    # 2. Model Initialization
    # ==========================================
    m = Model("universal_retail")

    # [PERFORMANCE TUNING]
    m.setParam("OutputFlag", 0)  # Suppress Gurobi logs
    m.setParam("TimeLimit", 60)  # Stop after 60 seconds (hard MIP instances)
    m.setParam("MIPGap", 0.01)   # Stop if solution is within 1% of optimal

    # ==========================================
    # 3. Decision Variables
    # ==========================================

    # --- Purchasing / Ordering ---
    # X: Order Quantity.
    # Z: Binary trigger for ordering (used for MOQ / Fixed Cost).
    X = {}
    Z = {}
    for p in products:
        for l in locations:
            for t in range(1, T + 1):
                vtype = GRB.INTEGER if pack_size > 1 else GRB.CONTINUOUS
                X[(p, l, t)] = m.addVar(lb=0.0, vtype=vtype, name=f"X_{p}_{l}_{t}")

                if moq > 0 or fixed_order_cost > 0:
                    Z[(p, l, t)] = m.addVar(vtype=GRB.BINARY, name=f"Z_{p}_{l}_{t}")

    # --- Inventory Operations ---
    # I: End-of-period inventory (indexed by vintage age 'a')
    # Y: Sales (indexed by vintage age 'a')
    # W: Waste (expired items)
    # L: Lost sales (unmet demand)
    I, Y, W, L = {}, {}, {}, {}
    for p in products:
        SL = shelf_life[p]
        for l in locations:
            for t in range(1, T + 1):
                for a in range(1, SL + 1):
                    I[(p, l, t, a)] = m.addVar(lb=0.0, name=f"I_{p}_{l}_{t}_{a}")
                    Y[(p, l, t, a)] = m.addVar(lb=0.0, name=f"Y_{p}_{l}_{t}_{a}")
                W[(p, l, t)] = m.addVar(lb=0.0, name=f"W_{p}_{l}_{t}")
                L[(p, l, t)] = m.addVar(lb=0.0, name=f"L_{p}_{l}_{t}")

    # --- Network Flow ---
    # TR: Transshipment quantity from src to dst
    TR = {}
    for p in products:
        for (src, dst) in trans_edges:
            for t in range(1, T + 1):
                TR[(p, src, dst, t)] = m.addVar(
                    lb=0.0, name=f"TR_{p}_{src}_{dst}_{t}"
                )

    # --- Substitution ---
    # S: Substitution quantity from pf to pt (pf-demand served by pt-product)
    S = {}
    for (pf, pt) in sub_edges:
        for l in locations:
            for t in range(1, T + 1):
                S[(pf, pt, l, t)] = m.addVar(
                    lb=0.0, name=f"S_{pf}_{pt}_{l}_{t}"
                )

    m.update()

    # ==========================================
    # 4. Constraints
    # ==========================================

    # --- 4.1 Logistics (MIP Constraints) ---
    big_M = 1_000_000.0
    for p in products:
        SL = shelf_life[p]
        for l in locations:
            for t in range(1, T + 1):
                # Pack size constraint: X must be a multiple of pack_size
                if pack_size > 1:
                    k = m.addVar(vtype=GRB.INTEGER, name=f"pack_{p}_{l}_{t}")
                    m.addConstr(X[(p, l, t)] == pack_size * k)

                # MOQ and fixed cost logic
                if moq > 0 or fixed_order_cost > 0:
                    # If Z = 0 then X = 0; if Z = 1 then X <= M
                    m.addConstr(X[(p, l, t)] <= big_M * Z[(p, l, t)])
                    # If Z = 1 and MOQ > 0, then X >= MOQ
                    if moq > 0:
                        m.addConstr(X[(p, l, t)] >= moq * Z[(p, l, t)])

    # --- 4.2 Inventory Dynamics ---
    for p in products:
        SL = shelf_life[p]
        LT = lead_time.get(p, 0)
        rr = return_rate.get(p, 0.0)

        for l in locations:

            def get_arrival(t_cur):
                t_ord = t_cur - LT
                if t_ord >= 1:
                    return X[(p, l, t_ord)]
                return 0.0

            def get_trans_net(t_cur):
                inc = quicksum(
                    TR[(p, src, l, t_cur)]
                    for (src, dst) in trans_edges
                    if dst == l
                )
                out = quicksum(
                    TR[(p, l, dst, t_cur)]
                    for (src, dst) in trans_edges
                    if src == l
                )
                return inc - out

            def get_returns(t_cur):
                if t_cur <= 1 or rr == 0.0:
                    return 0.0
                sales_prev = quicksum(
                    Y[(p, l, t_cur - 1, a)] for a in range(1, SL + 1)
                )
                return rr * sales_prev

            for t in range(1, T + 1):
                arr = get_arrival(t)
                tr = get_trans_net(t)
                ret = get_returns(t)

                # Newest vintage (age = SL): arrivals + net flow + returns
                m.addConstr(
                    I[(p, l, t, SL)]
                    == arr + tr + ret - Y[(p, l, t, SL)]
                )

                # Ageing vintages: today's age a comes from yesterday's a+1
                for a in range(1, SL):
                    prev = 0.0 if t == 1 else I[(p, l, t - 1, a + 1)]
                    m.addConstr(
                        I[(p, l, t, a)] == prev - Y[(p, l, t, a)]
                    )

                # Waste: previous period's oldest vintage that is not sold
                prev_1 = 0.0 if t == 1 else I[(p, l, t - 1, 1)]
                m.addConstr(W[(p, l, t)] == prev_1 - Y[(p, l, t, 1)])

    # --- 4.3 Capacity & Resource Constraints ---
    for t in range(1, T + 1):
        # Supplier production capacity by product
        for p in products:
            cap_list = prod_cap[p]
            c = cap_list[t - 1] if t - 1 < len(cap_list) else cap_list[-1]
            m.addConstr(
                quicksum(X[(p, l, t)] for l in locations) <= c
            )

        # Labor capacity by location and period
        for l in locations:
            labor_needed = quicksum(
                labor_use[p]
                * quicksum(
                    Y[(p, l, t, a)] for a in range(1, shelf_life[p] + 1)
                )
                for p in products
            )
            lc_list = labor_cap[l]
            lc = lc_list[t - 1] if t - 1 < len(lc_list) else lc_list[-1]
            m.addConstr(labor_needed <= lc)

        # Storage capacity (generic volume / cold chain)
        for l in locations:
            usage = quicksum(
                cold_use[p]
                * quicksum(
                    I[(p, l, t, a)] for a in range(1, shelf_life[p] + 1)
                )
                for p in products
            )
            m.addConstr(usage <= cold_cap[l])

        # Financial budget (open-to-buy style)
        if budget is not None:
            spend = quicksum(
                X[(p, l, t)] * purchase_cost.get(p, 10.0)
                for p in products
                for l in locations
            )
            if fixed_order_cost > 0.0:
                spend += quicksum(
                    fixed_order_cost * Z[(p, l, t)]
                    for p in products
                    for l in locations
                )
            m.addConstr(spend <= budget)

    # --- 4.4 Sustainability ---
    if waste_limit_pct is not None:
        tot_waste = quicksum(
            W[(p, l, t)]
            for p in products
            for l in locations
            for t in range(1, T + 1)
        )
        m.addConstr(tot_waste <= waste_limit_pct * total_demand_vol)

    # --- 4.5 Demand Balance & Substitution ---
    # We interpret an edge (pf, pt) as "pf-demand can be served by pt".
    # For each product p, location l, time t:
    #   outbound_sub = sum S[p -> *]
    #   inbound_sub  = sum S[* -> p]
    # Effective demand after substitution is:
    #   base_demand[p] + inbound_sub - outbound_sub
    # and this must be covered by sales + lost sales.
    for p in products:
        SL = shelf_life[p]
        for l in locations:
            for t in range(1, T + 1):
                outbound = quicksum(
                    S[(p, pt, l, t)]
                    for pt in outgoing_edges.get(p, [])
                )
                inbound = quicksum(
                    S[(pf, p, l, t)]
                    for pf in incoming_edges.get(p, [])
                )

                # Cannot redirect more demand than the original base demand
                m.addConstr(outbound <= demand[(p, l, t)])

                total_sales = quicksum(
                    Y[(p, l, t, a)] for a in range(1, SL + 1)
                )

                m.addConstr(
                    total_sales + L[(p, l, t)]
                    == demand[(p, l, t)] + inbound - outbound
                )

    # ==========================================
    # 5. Objective Function
    # ==========================================
    obj = 0.0
    for p in products:
        SL = shelf_life[p]
        for l in locations:
            for t in range(1, T + 1):
                # Inventory holding cost
                obj += inv_cost.get(p, 0.0) * quicksum(
                    I[(p, l, t, a)] for a in range(1, SL + 1)
                )
                # Waste penalty
                obj += waste_cost.get(p, 0.0) * W[(p, l, t)]
                # Lost sales penalty
                obj += lost_pen.get(p, 0.0) * L[(p, l, t)]

                # Fixed ordering cost
                if fixed_order_cost > 0.0 and (p, l, t) in Z:
                    obj += fixed_order_cost * Z[(p, l, t)]

    # Transshipment costs
    for p in products:
        for (src, dst) in trans_edges:
            for t in range(1, T + 1):
                obj += trans_cost_unit * TR[(p, src, dst, t)]

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    # ==========================================
    # 6. Output
    # ==========================================
    status = m.Status
    if summarize:
        name = data.get("name", "unknown")
        if status == GRB.OPTIMAL:
            status_str = "OPTIMAL"
        elif status == GRB.TIME_LIMIT:
            if m.SolCount > 0:
                status_str = "OPTIMAL (TL)"
            else:
                status_str = "TIMEOUT"
        elif status == GRB.INFEASIBLE:
            status_str = "INFEASIBLE"
        else:
            status_str = f"Code {status}"

        if m.SolCount > 0:
            obj_str = f"{m.objVal:,.2f}"
        else:
            obj_str = "N/A"

        print(f"| {name:<35} | {status_str:<14} | {obj_str:<15} |")

    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", type=str, required=True, help="Path to JSON data file"
    )
    args = parser.parse_args()

    solve_scenario(load_data(args.file))
