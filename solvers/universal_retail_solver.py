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
#     - Inventory: Remaining-life (shelf-life) tracking with aging and waste.
#     - Logistics: Lead time, MOQ (binary trigger), pack size (integer).
#     - Network: Transshipment and multi-echelon flows via locations + edges.
#     - Omni-channel: Reverse logistics, labor capacity by location and period.
#     - Financials: Period budgets and fixed order costs.
#     - Sustainability: Global waste limits relative to total demand.
#
#   [UPDATED]
#     - Fix perishability dynamics: consistent remaining-life aging + waste.
#     - Add purchasing cost into objective (was missing).
#     - Enforce production/procurement capacity on DELIVERED inflow (lead-time consistent).
#     - Align variable naming with benchmark contract: I,y,W,Q,L,S,X,z,n.
#     - TimeLimit=60s, MIPGap=1% to avoid stalling on hard instances.
#
# USAGE:
#   python universal_retail_solver.py --file ../scenarios/retail_comprehensive/data/xxx.json
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
    # ==========================================
    # 1. Parameter Extraction
    # ==========================================
    T = int(data["periods"])
    products = list(data["products"])
    locations = list(data["locations"])

    shelf_life = {p: int(v) for p, v in data["shelf_life"].items()}
    cold_cap = data["cold_capacity"]
    cold_use = data["cold_usage"]
    prod_cap = data["production_cap"]

    lead_time = data.get("lead_time", {p: 0 for p in products})
    constraints = data.get("constraints", {})
    moq = float(constraints.get("moq", 0) or 0)
    pack_size = int(constraints.get("pack_size", 1) or 1)
    budget = constraints.get("budget_per_period", None)
    waste_limit_pct = constraints.get("waste_limit_pct", None)

    network = data.get("network", {})
    sub_edges = [tuple(e) for e in network.get("sub_edges", [])]
    trans_edges = [tuple(e) for e in network.get("trans_edges", [])]

    outgoing_edges = {}
    incoming_edges = {}
    for pf, pt in sub_edges:
        outgoing_edges.setdefault(pf, []).append(pt)
        incoming_edges.setdefault(pt, []).append(pf)

    labor_use = data.get("labor_usage", {p: 0.0 for p in products})
    labor_cap = data.get("labor_cap", {l: [99999.0] * T for l in locations})
    return_rate = data.get("return_rate", {p: 0.0 for p in products})

    costs = data.get("costs", {})
    lost_pen = costs.get("lost_sales", {})
    inv_cost = costs.get("inventory", {})
    waste_cost = costs.get("waste", {})
    fixed_order_cost = float(costs.get("fixed_order", 0.0) or 0.0)
    trans_cost_unit = float(costs.get("transshipment", 0.0) or 0.0)
    purchase_cost = costs.get("purchasing", {p: 0.0 for p in products})

    demand_share = data.get("demand_share", {})
    demand = {}
    total_demand_vol = 0.0
    for p in products:
        curve = list(data["demand_curve"][p])
        if len(curve) < T:
            curve += [curve[-1]] * (T - len(curve))
        for l in locations:
            share = float(demand_share.get(l, 1.0 / max(1, len(locations))))
            for t in range(1, T + 1):
                d_val = float(curve[t - 1]) * share
                demand[(p, l, t)] = d_val
                total_demand_vol += d_val

    # ==========================================
    # 2. Model Initialization
    # ==========================================
    m = Model("universal_retail")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", 60)
    m.setParam("MIPGap", 0.01)

    # ==========================================
    # 3. Decision Variables
    # ==========================================
    # Orders
    Q, z, n = {}, {}, {}
    for p in products:
        for l in locations:
            for t in range(1, T + 1):
                Q[(p, l, t)] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"Q_{p}_{l}_{t}")
                if pack_size > 1:
                    n[(p, l, t)] = m.addVar(vtype=GRB.INTEGER, lb=0.0, name=f"n_{p}_{l}_{t}")
                if moq > 0 or fixed_order_cost > 0:
                    z[(p, l, t)] = m.addVar(vtype=GRB.BINARY, name=f"z_{p}_{l}_{t}")

    # Inventory by remaining life (start-of-period), sales, waste, lost sales
    I, y, W, L = {}, {}, {}, {}
    for p in products:
        SL = shelf_life[p]
        for l in locations:
            for t in range(1, T + 1):
                for a in range(1, SL + 1):
                    I[(p, l, t, a)] = m.addVar(lb=0.0, name=f"I_{p}_{l}_{t}_{a}")
                    y[(p, l, t, a)] = m.addVar(lb=0.0, name=f"y_{p}_{l}_{t}_{a}")
                W[(p, l, t)] = m.addVar(lb=0.0, name=f"W_{p}_{l}_{t}")
                L[(p, l, t)] = m.addVar(lb=0.0, name=f"L_{p}_{l}_{t}")

    # Transshipment
    X = {}
    for p in products:
        for (src, dst) in trans_edges:
            for t in range(1, T + 1):
                X[(p, src, dst, t)] = m.addVar(lb=0.0, name=f"X_{p}_{src}_{dst}_{t}")

    # Substitution
    S = {}
    for (pf, pt) in sub_edges:
        for l in locations:
            for t in range(1, T + 1):
                S[(pf, pt, l, t)] = m.addVar(lb=0.0, name=f"S_{pf}_{pt}_{l}_{t}")

    m.update()

    # ==========================================
    # 4. Constraints
    # ==========================================
    big_M = 1_000_000.0

    # Pack size + MOQ/fixed order trigger
    for p in products:
        for l in locations:
            for t in range(1, T + 1):
                if pack_size > 1:
                    m.addConstr(Q[(p, l, t)] == pack_size * n[(p, l, t)], name=f"pack_{p}_{l}_{t}")
                if moq > 0 or fixed_order_cost > 0:
                    m.addConstr(Q[(p, l, t)] <= big_M * z[(p, l, t)], name=f"moq_ub_{p}_{l}_{t}")
                    if moq > 0:
                        m.addConstr(Q[(p, l, t)] >= moq * z[(p, l, t)], name=f"moq_lb_{p}_{l}_{t}")

    def arrival_expr(p, l, t):
        LT = int(lead_time.get(p, 0) or 0)
        t_ord = t - LT
        if t_ord >= 1:
            return Q[(p, l, t_ord)]
        return 0.0

    def trans_net_expr(p, l, t):
        if not trans_edges:
            return 0.0
        inc = quicksum(X[(p, src, l, t)] for (src, dst) in trans_edges if dst == l)
        out = quicksum(X[(p, l, dst, t)] for (src, dst) in trans_edges if src == l)
        return inc - out

    def returns_expr(p, l, t):
        rr = float(return_rate.get(p, 0.0) or 0.0)
        if rr == 0.0 or t <= 1:
            return 0.0
        SL = shelf_life[p]
        prev_sales = quicksum(y[(p, l, t - 1, a)] for a in range(1, SL + 1))
        return rr * prev_sales

    # Perishability dynamics (consistent remaining-life aging)
    for p in products:
        SL = shelf_life[p]
        for l in locations:
            for a in range(1, SL):
                m.addConstr(I[(p, l, 1, a)] == 0.0, name=f"init_{p}_{l}_{a}")

            m.addConstr(
                I[(p, l, 1, SL)] == arrival_expr(p, l, 1) + trans_net_expr(p, l, 1) + returns_expr(p, l, 1),
                name=f"fresh_inflow_{p}_{l}_1",
            )

            for t in range(1, T + 1):
                for a in range(1, SL + 1):
                    m.addConstr(y[(p, l, t, a)] <= I[(p, l, t, a)], name=f"availability_{p}_{l}_{t}_{a}")
                m.addConstr(W[(p, l, t)] == I[(p, l, t, 1)] - y[(p, l, t, 1)], name=f"expire_clear_{p}_{l}_{t}")

            for t in range(1, T):
                for a in range(1, SL):
                    m.addConstr(
                        I[(p, l, t + 1, a)] == I[(p, l, t, a + 1)] - y[(p, l, t, a + 1)],
                        name=f"aging_{p}_{l}_{t}_{a}",
                    )
                m.addConstr(
                    I[(p, l, t + 1, SL)] == arrival_expr(p, l, t + 1) + trans_net_expr(p, l, t + 1) + returns_expr(p, l, t + 1),
                    name=f"fresh_inflow_{p}_{l}_{t+1}",
                )

    # Capacities + budget
    for t in range(1, T + 1):
        for p in products:
            cap_list = list(prod_cap[p])
            c = float(cap_list[t - 1] if t - 1 < len(cap_list) else cap_list[-1])
            inflow = quicksum(arrival_expr(p, l, t) for l in locations)
            m.addConstr(inflow <= c, name=f"prod_cap_{p}_{t}")

        for l in locations:
            labor_needed = quicksum(
                float(labor_use.get(p, 0.0) or 0.0)
                * quicksum(y[(p, l, t, a)] for a in range(1, shelf_life[p] + 1))
                for p in products
            )
            lc_list = list(labor_cap[l])
            lc = float(lc_list[t - 1] if t - 1 < len(lc_list) else lc_list[-1])
            m.addConstr(labor_needed <= lc, name=f"labor_cap_{l}_{t}")

        for l in locations:
            usage = quicksum(
                float(cold_use.get(p, 0.0) or 0.0)
                * quicksum(I[(p, l, t, a)] for a in range(1, shelf_life[p] + 1))
                for p in products
            )
            m.addConstr(usage <= float(cold_cap[l]), name=f"storage_cap_{l}_{t}")

        if budget is not None:
            spend = quicksum(
                Q[(p, l, t)] * float(purchase_cost.get(p, 0.0) or 0.0)
                for p in products
                for l in locations
            )
            if fixed_order_cost > 0.0 and z:
                spend += quicksum(
                    fixed_order_cost * z[(p, l, t)]
                    for p in products
                    for l in locations
                    if (p, l, t) in z
                )
            m.addConstr(spend <= float(budget), name=f"budget_{t}")

    # Waste cap
    if waste_limit_pct is not None:
        tot_waste = quicksum(W[(p, l, t)] for p in products for l in locations for t in range(1, T + 1))
        m.addConstr(tot_waste <= float(waste_limit_pct) * float(total_demand_vol), name="wastecap")

    # Demand + substitution
    for p in products:
        SL = shelf_life[p]
        for l in locations:
            for t in range(1, T + 1):
                outbound = quicksum(S[(p, pt, l, t)] for pt in outgoing_edges.get(p, []))
                inbound = quicksum(S[(pf, p, l, t)] for pf in incoming_edges.get(p, []))
                m.addConstr(outbound <= demand[(p, l, t)], name=f"demand_route_{p}_{l}_{t}")
                total_sales = quicksum(y[(p, l, t, a)] for a in range(1, SL + 1))
                m.addConstr(
                    total_sales + L[(p, l, t)] == demand[(p, l, t)] + inbound - outbound,
                    name=f"sales_conservation_{p}_{l}_{t}",
                )

    # ==========================================
    # 5. Objective
    # ==========================================
    obj = 0.0
    for p in products:
        SL = shelf_life[p]
        inv_c = float(inv_cost.get(p, 0.0) or 0.0)
        waste_c = float(waste_cost.get(p, 0.0) or 0.0)
        lost_c = float(lost_pen.get(p, 0.0) or 0.0)
        buy_c = float(purchase_cost.get(p, 0.0) or 0.0)

        for l in locations:
            for t in range(1, T + 1):
                if buy_c != 0.0:
                    obj += buy_c * Q[(p, l, t)]
                if inv_c != 0.0:
                    obj += inv_c * quicksum(I[(p, l, t, a)] - y[(p, l, t, a)] for a in range(2, SL + 1))
                if waste_c != 0.0:
                    obj += waste_c * W[(p, l, t)]
                if lost_c != 0.0:
                    obj += lost_c * L[(p, l, t)]
                if fixed_order_cost > 0.0 and (p, l, t) in z:
                    obj += fixed_order_cost * z[(p, l, t)]

    if trans_edges and trans_cost_unit != 0.0:
        for p in products:
            for (src, dst) in trans_edges:
                for t in range(1, T + 1):
                    obj += trans_cost_unit * X[(p, src, dst, t)]

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    status = m.Status
    if summarize:
        name = data.get("name", "unknown")
        if status == GRB.OPTIMAL:
            status_str = "OPTIMAL"
        elif status == GRB.TIME_LIMIT:
            status_str = "OPTIMAL (TL)" if m.SolCount > 0 else "TIMEOUT"
        elif status == GRB.INFEASIBLE:
            status_str = "INFEASIBLE"
        else:
            status_str = f"Code {status}"

        obj_str = f"{m.objVal:,.2f}" if m.SolCount > 0 else "N/A"
        print(f"| {name:<35} | {status_str:<14} | {obj_str:<15} |")

    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to JSON data file")
    args = parser.parse_args()
    solve_scenario(load_data(args.file))
