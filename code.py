import gurobipy as gp
from gurobipy import GRB

def ts(param, t, default=0):
    if param is None:
        return default
    if isinstance(param, (int, float)):
        return param
    if isinstance(param, list):
        if len(param) == 0:
            return default
        return param[min(t-1, len(param)-1)]
    return default

T = int(data["periods"])
products = list(data["products"])
locations = list(data["locations"])
sub_edges = data["network"]["sub_edges"]
trans_edges = data["network"]["trans_edges"]
demand_share = data["demand_share"]
shelf_life = {p: data["shelf_life"][p] for p in products}
lead_time = {p: data.get("lead_time", {}).get(p, 0) for p in products}
costs_inventory = data["costs"]["inventory"]
costs_waste = data["costs"]["waste"]
costs_lost_sales = data["costs"]["lost_sales"]
costs_purchasing = data["costs"]["purchasing"]

# Build edge mappings
outgoing_edges = {p: [] for p in products}
incoming_edges = {p: [] for p in products}
for p_from, p_to in sub_edges:
    outgoing_edges[p_to].append(p_from)
    incoming_edges[p_from].append(p_to)

# Create model
m = gp.Model()
m.Params.OutputFlag = 0
m.Params.Threads = 1
m.Params.Seed = 0

# Create variables
I = m.addVars(products, locations, range(1, T+1), range(1, max(shelf_life.values())+1), name="I")
y = m.addVars(products, locations, range(1, T+1), range(1, max(shelf_life.values())+1), name="y")
W = m.addVars(products, locations, range(1, T+1), name="W")
Q = m.addVars(products, locations, range(1, T+1), name="Q")
L = m.addVars(products, locations, range(1, T+1), name="L")
if sub_edges:
    S = m.addVars([(pf, pt) for pf, pt in sub_edges], locations, range(1, T+1), name="S")

# Set objective
m.setObjective(gp.quicksum(costs_inventory[p] * gp.quicksum(I[p, l, t, a] for a in range(2, shelf_life[p]+1)) for p in products for l in locations for t in range(1, T+1)) +
              gp.quicksum(costs_waste[p] * W[p, l, t] for p in products for l in locations for t in range(1, T+1)) +
              gp.quicksum(costs_lost_sales[p] * L[p, l, t] for p in products for l in locations for t in range(1, T+1)) +
              gp.quicksum(costs_purchasing[p] * Q[p, l, t] for p in products for l in locations for t in range(1, T+1)), GRB.MINIMIZE)

# Add constraints
for p in products:
    for l in locations:
        for t in range(1, T+1):
            demand_plt = data["demand_curve"][p][t-1] * demand_share[l]
            
            # Substitution flows
            if sub_edges:
                S_in = gp.quicksum(S[pf, p, l, t] for pf in incoming_edges[p]) if incoming_edges[p] else 0
                S_out = gp.quicksum(S[p, pt, l, t] for pt in outgoing_edges[p]) if outgoing_edges[p] else 0
            else:
                S_in = 0
                S_out = 0
            
            # demand_route: S_out <= demand
            if sub_edges and outgoing_edges[p]:
                m.addConstr(S_out <= demand_plt, name=f"demand_route_{p}_{l}_{t}")
            
            # sales_conservation: y + L = demand + S_in - S_out
            total_sales = gp.quicksum(y[p, l, t, a] for a in range(1, shelf_life[p]+1))
            m.addConstr(total_sales + L[p, l, t] == demand_plt + S_in - S_out, 
                       name=f"sales_conservation_{p}_{l}_{t}")
            
            # availability: y <= I
            for a in range(1, shelf_life[p]+1):
                m.addConstr(y[p, l, t, a] <= I[p, l, t, a], name=f"availability_{p}_{l}_{t}_{a}")
            
            # aging: I[t+1, a] = I[t, a+1] - y[t, a+1]
            for a in range(1, shelf_life[p]):
                m.addConstr(I[p, l, t+1, a] == I[p, l, t, a+1] - y[p, l, t, a+1], name=f"aging_{p}_{l}_{t}_{a}")
            
            # expire_clear: W[t] = I[t, 1] - y[t, 1]
            m.addConstr(W[p, l, t] == I[p, l, t, 1] - y[p, l, t, 1], name=f"expire_clear_{p}_{l}_{t}")
            
            # fresh_inflow: I[t, shelf_life] = Q[t-lead_time] + returns
            if t > lead_time[p]:
                m.addConstr(I[p, l, t, shelf_life[p]] == Q[p, l, t - lead_time[p]], name=f"fresh_inflow_{p}_{l}_{t}")
            else:
                m.addConstr(I[p, l, t, shelf_life[p]] == 0, name=f"fresh_inflow_{p}_{l}_{t}")

# Initialization at t=1
for p in products:
    for l in locations:
        for a in range(1, shelf_life[p]):
            m.addConstr(I[p, l, 1, a] == 0, name=f"init_nonfresh_{p}_{l}_{a}")
        if lead_time[p] == 0:
            m.addConstr(I[p, l, 1, shelf_life[p]] == Q[p, l, 1], name=f"init_fresh_{p}_{l}")
        else:
            m.addConstr(I[p, l, 1, shelf_life[p]] == 0, name=f"init_fresh_{p}_{l}")

# Solve and print
m.optimize()
print(f"status: {m.Status}")
if m.Status == GRB.OPTIMAL:
    print(f"objective: {m.ObjVal}")