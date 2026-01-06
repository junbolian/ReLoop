import gurobipy as gp

m = gp.Model()

products = data['products']
locations = data['locations']
T = data['periods']

I = {}
y = {}
W = {}
Q = {}
L = {}
S = {}

for p in products:
    for l in locations:
        for t in range(1, T + 1):
            for a in range(1, data['shelf_life'][p] + 1):
                I[p,l,t,a] = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=f"I_{p}_{l}_{t}_{a}")
                y[p,l,t,a] = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=f"y_{p}_{l}_{t}_{a}")
            W[p,l,t] = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=f"W_{p}_{l}_{t}")
            Q[p,l,t] = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=f"Q_{p}_{l}_{t}")
            L[p,l,t] = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=f"L_{p}_{l}_{t}")

if data['network']['sub_edges']:
    for edge in data['network']['sub_edges']:
        p_from, p_to = edge
        for l in locations:
            for t in range(1, T + 1):
                S[p_from,p_to,l,t] = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=f"S_{p_from}_{p_to}_{l}_{t}")

for p in products:
    for l in locations:
        for a in range(1, data['shelf_life'][p]):
            m.addConstr(I[p,l,1,a] == 0, name=f"initialization_{p}_{l}_a{a}")

for p in products:
    for l in locations:
        for t in range(1, T + 1):
            if t > data['lead_time'][p]:
                m.addConstr(I[p,l,t,data['shelf_life'][p]] == Q[p,l,t-data['lead_time'][p]], name=f"fresh_inflow_{p}_{l}_{t}")
            else:
                m.addConstr(I[p,l,t,data['shelf_life'][p]] == 0, name=f"fresh_inflow_{p}_{l}_{t}")

for p in products:
    for l in locations:
        for t in range(1, T):
            for a in range(1, data['shelf_life'][p]):
                m.addConstr(I[p,l,t+1,a] == I[p,l,t,a+1] - y[p,l,t,a+1], name=f"aging_{p}_{l}_{t}_{a}")

for p in products:
    for l in locations:
        for t in range(1, T + 1):
            m.addConstr(W[p,l,t] == I[p,l,t,1] - y[p,l,t,1], name=f"expire_clear_{p}_{l}_{t}")

for p in products:
    for l in locations:
        for t in range(1, T + 1):
            for a in range(1, data['shelf_life'][p] + 1):
                m.addConstr(y[p,l,t,a] <= I[p,l,t,a], name=f"availability_{p}_{l}_{t}_{a}")

for p in products:
    for l in locations:
        for t in range(1, T + 1):
            demand = data['demand_curve'][p][t-1] * data['demand_share'][l]
            sales_sum = gp.quicksum(y[p,l,t,a] for a in range(1, data['shelf_life'][p] + 1))
            
            inbound = 0
            outbound = 0
            if data['network']['sub_edges']:
                for edge in data['network']['sub_edges']:
                    if edge[1] == p:
                        inbound += S[edge[0],edge[1],l,t]
                    if edge[0] == p:
                        outbound += S[edge[0],edge[1],l,t]
            
            m.addConstr(sales_sum + L[p,l,t] == demand + inbound - outbound, name=f"sales_conservation_{p}_{l}_{t}")

if data['network']['sub_edges']:
    for p in products:
        outgoing_edges = [edge for edge in data['network']['sub_edges'] if edge[0] == p]
        if outgoing_edges:
            for l in locations:
                for t in range(1, T + 1):
                    demand = data['demand_curve'][p][t-1] * data['demand_share'][l]
                    outbound_sum = gp.quicksum(S[p,edge[1],l,t] for edge in outgoing_edges)
                    m.addConstr(outbound_sum <= demand, name=f"demand_route_{p}_{l}_{t}")

for l in locations:
    for t in range(1, T + 1):
        inventory_sum = gp.quicksum(data['cold_usage'][p] * gp.quicksum(I[p,l,t,a] for a in range(1, data['shelf_life'][p] + 1)) for p in products)
        m.addConstr(inventory_sum <= data['cold_capacity'][l], name=f"storage_cap_{l}_{t}")

for p in products:
    for t in range(1, T + 1):
        production_sum = gp.quicksum(Q[p,l,t] for l in locations)
        m.addConstr(production_sum <= data['production_cap'][p][t-1], name=f"prod_cap_{p}_{t}")

if 'labor_cap' in data and 'labor_usage' in data:
    for l in locations:
        for t in range(1, T + 1):
            labor_sum = gp.quicksum(data['labor_usage'][p] * gp.quicksum(y[p,l,t,a] for a in range(1, data['shelf_life'][p] + 1)) for p in products)
            m.addConstr(labor_sum <= data['labor_cap'][l][t-1], name=f"labor_cap_{l}_{t}")

obj = 0

for p in products:
    for l in locations:
        for t in range(1, T + 1):
            obj += data['costs']['purchasing'][p] * Q[p,l,t]

for p in products:
    for l in locations:
        for t in range(1, T + 1):
            for a in range(2, data['shelf_life'][p] + 1):
                obj += data['costs']['inventory'][p] * (I[p,l,t,a] - y[p,l,t,a])

for p in products:
    for l in locations:
        for t in range(1, T + 1):
            obj += data['costs']['waste'][p] * W[p,l,t]

for p in products:
    for l in locations:
        for t in range(1, T + 1):
            obj += data['costs']['lost_sales'][p] * L[p,l,t]

m.setObjective(obj, gp.GRB.MINIMIZE)

m.setParam('OutputFlag', 0)
m.setParam('Threads', 1)
m.setParam('Seed', 0)

m.optimize()

print(f"Status: {m.status}")
if m.status == gp.GRB.OPTIMAL:
    print(f"Objective: {m.objVal}")