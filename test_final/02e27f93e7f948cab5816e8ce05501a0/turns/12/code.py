import gurobipy as gp

products = data['products']
locations = data['locations']
T = data['periods']
shelf_life = data['shelf_life']
lead_time = data['lead_time']
demand_curve = data['demand_curve']
demand_share = data['demand_share']
production_cap = data['production_cap']
cold_capacity = data['cold_capacity']
cold_usage = data['cold_usage']
sub_edges = data['network']['sub_edges']
trans_edges = data['network']['trans_edges']
costs = data['costs']

m = gp.Model()

I = {}
for p in products:
    for l in locations:
        for t in range(1, T + 1):
            for a in range(1, shelf_life[p] + 1):
                I[p, l, t, a] = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=f"I_{p}_{l}_{t}_{a}")

y = {}
for p in products:
    for l in locations:
        for t in range(1, T + 1):
            for a in range(1, shelf_life[p] + 1):
                y[p, l, t, a] = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=f"y_{p}_{l}_{t}_{a}")

W = {}
for p in products:
    for l in locations:
        for t in range(1, T + 1):
            W[p, l, t] = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=f"W_{p}_{l}_{t}")

Q = {}
for p in products:
    for l in locations:
        for t in range(1, T + 1):
            Q[p, l, t] = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=f"Q_{p}_{l}_{t}")

L = {}
for p in products:
    for l in locations:
        for t in range(1, T + 1):
            L[p, l, t] = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=f"L_{p}_{l}_{t}")

S = {}
if sub_edges:
    for edge in sub_edges:
        p_from, p_to = edge
        for l in locations:
            for t in range(1, T + 1):
                S[p_from, p_to, l, t] = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=f"S_{p_from}_{p_to}_{l}_{t}")

for p in products:
    for l in locations:
        for a in range(1, shelf_life[p]):
            m.addConstr(I[p, l, 1, a] == 0, name=f"initialization_{p}_{l}_a{a}")

for p in products:
    for l in locations:
        for t in range(1, T + 1):
            if t > lead_time[p]:
                m.addConstr(I[p, l, t, shelf_life[p]] == Q[p, l, t - lead_time[p]], name=f"fresh_inflow_{p}_{l}_{t}")
            else:
                m.addConstr(I[p, l, t, shelf_life[p]] == 0, name=f"fresh_inflow_{p}_{l}_{t}")

for p in products:
    for l in locations:
        for t in range(1, T):
            for a in range(1, shelf_life[p]):
                m.addConstr(I[p, l, t + 1, a] == I[p, l, t, a + 1] - y[p, l, t, a + 1], name=f"aging_{p}_{l}_{t}_{a}")

for p in products:
    for l in locations:
        for t in range(1, T + 1):
            m.addConstr(W[p, l, t] == I[p, l, t, 1] - y[p, l, t, 1], name=f"expire_clear_{p}_{l}_{t}")

for p in products:
    for l in locations:
        for t in range(1, T + 1):
            for a in range(1, shelf_life[p] + 1):
                m.addConstr(y[p, l, t, a] <= I[p, l, t, a], name=f"availability_{p}_{l}_{t}_{a}")

for p in products:
    for l in locations:
        for t in range(1, T + 1):
            demand = demand_curve[p][t - 1] * demand_share[l]
            sales_sum = gp.quicksum(y[p, l, t, a] for a in range(1, shelf_life[p] + 1))
            inbound = gp.quicksum(S[p_from, p, l, t] for p_from, p_to in sub_edges if p_to == p) if sub_edges else 0
            outbound = gp.quicksum(S[p, p_to, l, t] for p_from, p_to in sub_edges if p_from == p) if sub_edges else 0
            m.addConstr(sales_sum + L[p, l, t] == demand + inbound - outbound, name=f"sales_conservation_{p}_{l}_{t}")

if sub_edges:
    for p in products:
        outgoing_edges = [(p_from, p_to) for p_from, p_to in sub_edges if p_from == p]
        if outgoing_edges:
            for l in locations:
                for t in range(1, T + 1):
                    demand = demand_curve[p][t - 1] * demand_share[l]
                    outbound_sum = gp.quicksum(S[p, p_to, l, t] for p_from, p_to in outgoing_edges)
                    m.addConstr(outbound_sum <= demand, name=f"demand_route_{p}_{l}_{t}")

for l in locations:
    for t in range(1, T + 1):
        storage_sum = gp.quicksum(cold_usage[p] * gp.quicksum(I[p, l, t, a] for a in range(1, shelf_life[p] + 1)) for p in products)
        m.addConstr(storage_sum <= cold_capacity[l], name=f"storage_cap_{l}_{t}")

for p in products:
    for t in range(1, T + 1):
        production_sum = gp.quicksum(Q[p, l, t] for l in locations)
        m.addConstr(production_sum <= production_cap[p][t - 1], name=f"prod_cap_{p}_{t}")

obj = 0

for p in products:
    for l in locations:
        for t in range(1, T + 1):
            obj += costs['purchasing'][p] * Q[p, l, t]

for p in products:
    for l in locations:
        for t in range(1, T + 1):
            for a in range(2, shelf_life[p] + 1):
                obj += costs['inventory'][p] * (I[p, l, t, a] - y[p, l, t, a])

for p in products:
    for l in locations:
        for t in range(1, T + 1):
            obj += costs['waste'][p] * W[p, l, t]

for p in products:
    for l in locations:
        for t in range(1, T + 1):
            obj += costs['lost_sales'][p] * L[p, l, t]

m.setObjective(obj, gp.GRB.MINIMIZE)

m.setParam('OutputFlag', 0)
m.setParam('Threads', 1)
m.setParam('Seed', 0)

m.optimize()

print(f"Status: {m.status}")
if m.status == gp.GRB.OPTIMAL:
    print(f"Objective: {m.objVal}")