import gurobipy as gp
from gurobipy import GRB

# Assume 'data' dict is pre-loaded in the environment

# Basic sets
P = data['products']
L = data['locations']
T = range(1, data['periods'] + 1)

# Shelf life per product
SL = {p: data['shelf_life'][p] for p in P}
R = {p: range(1, SL[p] + 1) for p in P}

# Network sets
network = data.get('network', {})
sub_edges_list = network.get('sub_edges', [])
trans_edges_list = network.get('trans_edges', [])
E_sub = [tuple(e) for e in sub_edges_list]
E_trans = [tuple(e) for e in trans_edges_list]

# Parameters
demand_share = data['demand_share']
production_cap = {p: {t: data['production_cap'][p][t-1] for t in T} for p in P}

# Demand d_p_l_t = data['demand_curve'][p][t-1] * data['demand_share'][l]
demand = {
    (p, l, t): data['demand_curve'][p][t-1] * demand_share[l]
    for p in P for l in L for t in T
}

cold_capacity = data['cold_capacity']
cold_usage = data['cold_usage']

cost_purch = data['costs']['purchasing']
cost_inv = data['costs']['inventory']
cost_waste = data['costs']['waste']
cost_lost = data['costs']['lost_sales']
cost_fixed = data['costs']['fixed_order']
cost_trans = data['costs']['transshipment']

constraints_cfg = data['constraints']
MOQ = constraints_cfg['moq']
pack_size = constraints_cfg['pack_size']
budget_per_period = constraints_cfg.get('budget_per_period', None)
waste_limit_pct = constraints_cfg.get('waste_limit_pct', None)

# Big-M parameter M_p_t = cap_p_t
M = production_cap

# Waste limit alpha_p (if scalar, applies to all p)
alpha_raw = waste_limit_pct
if alpha_raw is None:
    alpha = {p: None for p in P}
else:
    # If scalar, same for all products
    alpha = {p: alpha_raw for p in P}

# Model
m = gp.Model()
m.Params.OutputFlag = 0
m.Params.Threads = 1
m.Params.Seed = 0
m.Params.TimeLimit = 120

# Decision variables
Q = m.addVars(P, T, lb=0, vtype=GRB.CONTINUOUS, name="Q")
I = {
    (p, l, t, r): m.addVar(lb=0, vtype=GRB.CONTINUOUS,
                           name=f"I[{p},{l},{t},{r}]")
    for p in P for l in L for t in T for r in R[p]
}
sales = {
    (p, l, t, r): m.addVar(lb=0, vtype=GRB.CONTINUOUS,
                           name=f"sales[{p},{l},{t},{r}]")
    for p in P for l in L for t in T for r in R[p]
}
W = m.addVars(P, L, T, lb=0, vtype=GRB.CONTINUOUS, name="W")
L_var = m.addVars(P, L, T, lb=0, vtype=GRB.CONTINUOUS, name="L")
sub = m.addVars(
    [(p_from, p_to, l, t)
     for (p_from, p_to) in E_sub for l in L for t in T],
    lb=0, vtype=GRB.CONTINUOUS, name="sub"
)
trans = m.addVars(
    [(p, l_from, l_to, t)
     for p in P for (l_from, l_to) in E_trans for t in T],
    lb=0, vtype=GRB.CONTINUOUS, name="trans"
)
y = m.addVars(P, T, vtype=GRB.BINARY, name="y")

# Pack-size integer variable z[p,t] if pack_size is enforced (>0)
if pack_size is not None and pack_size > 0:
    z = m.addVars(P, T, lb=0, vtype=GRB.INTEGER, name="z")
else:
    z = None

# Objective components

# purchasing_cost: ∑_{p∈P} ∑_{t∈T} c_purch_p · Q[p,t]
purchasing_cost = gp.quicksum(cost_purch[p] * Q[p, t] for p in P for t in T)

# inventory_holding_cost:
# ∑_{p∈P} ∑_{l∈L} ∑_{t∈T} c_inv_p · (∑_{r ∈ R_p, r ≥ 2} (I[p,l,t,r] − sales[p,l,t,r]))
inventory_holding_cost = gp.quicksum(
    cost_inv[p] * gp.quicksum(
        I[p, l, t, r] - sales[p, l, t, r]
        for r in R[p] if r >= 2
    )
    for p in P for l in L for t in T
)

# waste_cost: ∑_{p∈P} ∑_{l∈L} ∑_{t∈T} c_waste_p · W[p,l,t]
waste_cost = gp.quicksum(
    cost_waste[p] * W[p, l, t] for p in P for l in L for t in T
)

# lost_sales_cost: ∑_{p∈P} ∑_{l∈L} ∑_{t∈T} c_lost_p · L[p,l,t]
lost_sales_cost = gp.quicksum(
    cost_lost[p] * L_var[p, l, t] for p in P for l in L for t in T
)

# fixed_order_cost: ∑_{p∈P} ∑_{t∈T} c_fixed · y[p,t]
fixed_order_cost = gp.quicksum(
    cost_fixed * y[p, t] for p in P for t in T
)

# transshipment_cost:
# ∑_{p∈P} ∑_{t∈T} ∑_{(l_from,l_to)∈E_trans} c_trans · trans[p,l_from,l_to,t]
transshipment_cost = gp.quicksum(
    cost_trans * trans[p, l_from, l_to, t]
    for p in P for (l_from, l_to) in E_trans for t in T
)

m.setObjective(
    purchasing_cost
    + inventory_holding_cost
    + waste_cost
    + lost_sales_cost
    + fixed_order_cost
    + transshipment_cost,
    GRB.MINIMIZE
)

# Constraints

# fresh_inflow_by_remaining_life:
# I[p,l,t,SL_p] = Q[p,t] · demand_share[l]
for p in P:
    SL_p = SL[p]
    for l in L:
        share_l = demand_share[l]
        for t in T:
            m.addConstr(
                I[p, l, t, SL_p] == Q[p, t] * share_l,
                name=f"fresh_inflow_by_remaining_life[{p},{l},{t}]"
            )

# aging_of_inventory:
# I[p,l,t+1,r] = I[p,l,t,r+1] − sales[p,l,t,r+1]
# ∀ t with t+1 ∈ T, ∀ r ∈ {1, …, SL_p−1}
for p in P:
    SL_p = SL[p]
    for l in L:
        for t in T:
            if t + 1 in T:
                for r in range(1, SL_p):
                    m.addConstr(
                        I[p, l, t + 1, r] == I[p, l, t, r + 1] - sales[p, l, t, r + 1],
                        name=f"aging_of_inventory[{p},{l},{t},{r}]"
                    )

# waste_definition:
# W[p,l,t] = I[p,l,t,1] − sales[p,l,t,1]
for p in P:
    for l in L:
        for t in T:
            m.addConstr(
                W[p, l, t] == I[p, l, t, 1] - sales[p, l, t, 1],
                name=f"waste_definition[{p},{l},{t}]"
            )

# sales_limited_by_inventory:
# sales[p,l,t,r] ≤ I[p,l,t,r]
for p in P:
    for l in L:
        for t in T:
            for r in R[p]:
                m.addConstr(
                    sales[p, l, t, r] <= I[p, l, t, r],
                    name=f"sales_limited_by_inventory[{p},{l},{t},{r}]"
                )

# nonnegativity_flows_stocks: already enforced by variable lb >= 0, so no extra constraints needed

# production_capacity:
# Q[p,t] ≤ cap_p_t
for p in P:
    for t in T:
        m.addConstr(
            Q[p, t] <= production_cap[p][t],
            name=f"production_capacity[{p},{t}]"
        )

# storage_capacity:
# ∑_{p∈P} u_p · (∑_{r∈R_p} (I[p,l,t,r] − sales[p,l,t,r])) ≤ K_l
for l in L:
    K_l = cold_capacity[l]
    m.addConstrs(
        (
            gp.quicksum(
                cold_usage[p] * gp.quicksum(
                    I[p, l, t, r] - sales[p, l, t, r]
                    for r in R[p]
                )
                for p in P
            ) <= K_l
        )
        for t in T
    )

# demand_fulfillment_with_substitution_p_from:
# ∑_{r∈R_p} sales[p_from,l,t,r] +
# ∑_{p_to : (p_from,p_to)∈E_sub} sub[p_from,p_to,l,t] +
# L[p_from,l,t] = d_p_from_l_t
for p_from in P:
    for l in L:
        for t in T:
            m.addConstr(
                gp.quicksum(sales[p_from, l, t, r] for r in R[p_from]) +
                gp.quicksum(
                    sub[p_from, p_to, l, t]
                    for (pf, p_to) in E_sub if pf == p_from
                ) +
                L_var[p_from, l, t] ==
                demand[(p_from, l, t)],
                name=f"demand_fulfillment_p_from[{p_from},{l},{t}]"
            )

# demand_fulfillment_with_substitution_p_to:
# ∑_{r∈R_p} sales[p_to,l,t,r] −
# ∑_{p_from : (p_from,p_to)∈E_sub} sub[p_from,p_to,l,t] +
# L[p_to,l,t] = d_p_to_l_t
for p_to in P:
    for l in L:
        for t in T:
            m.addConstr(
                gp.quicksum(sales[p_to, l, t, r] for r in R[p_to]) -
                gp.quicksum(
                    sub[p_from, p_to, l, t]
                    for (p_from, pt) in E_sub if pt == p_to
                ) +
                L_var[p_to, l, t] ==
                demand[(p_to, l, t)],
                name=f"demand_fulfillment_p_to[{p_to},{l},{t}]"
            )

# substitution_limited_by_sales_capacity:
# ∑_{p_from : (p_from,p_to)∈E_sub} sub[p_from,p_to,l,t] ≤ ∑_{r∈R_p} sales[p_to,l,t,r]
for p_to in P:
    for l in L:
        for t in T:
            m.addConstr(
                gp.quicksum(
                    sub[p_from, p_to, l, t]
                    for (p_from, pt) in E_sub if pt == p_to
                ) <=
                gp.quicksum(sales[p_to, l, t, r] for r in R[p_to]),
                name=f"substitution_limited_by_sales_capacity[{p_to},{l},{t}]"
            )

# no_transshipment:
# trans[p,l_from,l_to,t] = 0
for p in P:
    for (l_from, l_to) in E_trans:
        for t in T:
            m.addConstr(
                trans[p, l_from, l_to, t] == 0,
                name=f"no_transshipment[{p},{l_from},{l_to},{t}]"
            )

# fixed_order_linkage:
# 0 ≤ Q[p,t] ≤ M_p_t · y[p,t]
for p in P:
    for t in T:
        m.addConstr(
            Q[p, t] <= M[p][t] * y[p, t],
            name=f"fixed_order_linkage_ub[{p},{t}]"
        )
        # Lower bound 0 is already ensured by Q lb, but we keep the structure
        # "0 ≤ Q[p,t]" is inherent

# minimum_order_quantity:
# Q[p,t] ≥ MOQ · y[p,t]
for p in P:
    for t in T:
        m.addConstr(
            Q[p, t] >= MOQ * y[p, t],
            name=f"minimum_order_quantity[{p},{t}]"
        )

# pack_size_order_multiple:
# Q[p,t] = pack · z[p,t]
if z is not None:
    for p in P:
        for t in T:
            m.addConstr(
                Q[p, t] == pack_size * z[p, t],
                name=f"pack_size_order_multiple[{p},{t}]"
            )

# budget_per_period:
# ∑_{p∈P} c_purch_p · Q[p,t] ≤ B_t
if budget_per_period is not None:
    for t in T:
        m.addConstr(
            gp.quicksum(cost_purch[p] * Q[p, t] for p in P) <= budget_per_period,
            name=f"budget_per_period[{t}]"
        )

# waste_limit_fraction_of_demand:
# ∑_{l∈L} ∑_{t∈T} W[p,l,t] ≤ alpha_p · ∑_{l∈L} ∑_{t∈T} d_p_l_t
for p in P:
    if alpha[p] is not None:
        total_demand_p = gp.quicksum(
            demand[(p, l, t)] for l in L for t in T
        )
        m.addConstr(
            gp.quicksum(W[p, l, t] for l in L for t in T) <= alpha[p] * total_demand_p,
            name=f"waste_limit_fraction_of_demand[{p}]"
        )

# boundary_conditions: Initial inventory at t = 1 set to 0 (if not provided)
# I[p,l,1,r] given exogenously or set to 0
# Here we set to 0
for p in P:
    for l in L:
        for r in R[p]:
            m.addConstr(
                I[p, l, 1, r] == 0,
                name=f"initial_inventory_zero[{p},{l},{r}]"
            )

# Aging equation at last period already handled by only adding for t+1 in T
# Terminal inventory handling: no extra constraints

# Solve
m.optimize()

# Output
print(f"status: {m.Status}")
if m.Status == GRB.OPTIMAL:
    print(f"objective: {m.ObjVal}")