import gurobipy as gp
from gurobipy import GRB

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
                I[p,l,t,a] = m.addVar(lb=0, name=f"I_{p}_{l}_{t}_{a}")
                y[p,l,t,a] = m.addVar(lb=0, name=f"y_{p}_{l}_{t}_{a}")
            W[p,l,t] = m.addVar(lb=0, name=f"W_{p}_{l}_{t}")
            Q[p,l,t] = m.addVar(lb=0, name=f"Q_{p}_{l}_{t}")
            L[p,l,t] = m.addVar(lb=0, name=f"L_{p}_{l}_{t}")

if data['network']['sub_edges']:
    for edge in data['network']['sub_edges']:
        p_from, p_to = edge
        for l in locations:
            for t in range(1, T + 1):
                S[p_from,p_to,l,t] = m.addVar(lb=0, name=f"S_{p_from}_{p_to}_{l}_{t}")

obj = 0
for p in products:
    for l in locations:
        for t in range(1, T + 1):
            obj += data['costs']['purchasing'][p] * Q[p,l,t]
            obj += data['costs']['waste'][p] * W[p,l,t]
            obj += data['costs']['lost_sales'][p] * L[p,l,t]
            for a in range(2, data['shelf_life'][p] + 1):
                obj += data['costs']['inventory'][p] * (I[p,l,t,a] - y[p,l,t,a])

m.setObjective(obj, GRB.MINIMIZE)

for p in products:
    for l in locations:
        for a in range(1, data['shelf_life'][p]):
            m.addConstr(I[p,l,1,a] == 0, name=f"initialization_{p}_{l}_a{a}")

for p in products:
    for l in locations:
        for t in range(1, T + 1):
            if t > data['lead_time'][p]:
                m.addConstr(I[p,l,t,data['shelf_life'][p]] == Q[p,l,t-data['lead_time'][p]], 
                           name=f"fresh_inflow_{p}_{l}_{t}")
            else:
                m.addConstr(I[p,l,t,data['shelf_life'][p]] == 0, 
                           name=f"fresh_inflow_{p}_{l}_{t}")

for p in products:
    for l in locations:
        for t in range(1, T):
            for a in range(1, data['shelf_life'][p]):
                m.addConstr(I[p,l,t+1,a] == I[p,l,t,a+1] - y[p,l,t,a+1], 
                           name=f"aging_{p}_{l}_{t}_{a}")

for p in products:
    for l in locations:
        for t in range(1, T + 1):
            m.addConstr(W[p,l,t] == I[p,l,t,1] - y[p,l,t,1], 
                       name=f"expire_clear_{p}_{l}_{t}")

for p in products:
    for l in locations:
        for t in range(1, T + 1):
            for a in range(1, data['shelf_life'][p] + 1):
                m.addConstr(y[p,l,t,a] <= I[p,l,t,a], 
                           name=f"availability_{p}_{l}_{t}_{a}")

for p in products:
    for l in locations:
        for t in range(1, T + 1):
            demand = data['demand_curve'][p][t-1] * data['demand_share'][l]
            sales_sum = sum(y[p,l,t,a] for a in range(1, data['shelf_life'][p] + 1))
            
            inbound = 0
            outbound = 0
            
            if data['network']['sub_edges']:
                for edge in data['network']['sub_edges']:
                    if edge[1] == p:
                        inbound += S[edge[0],edge[1],l,t]
                    if edge[0] == p:
                        outbound += S[edge[0],edge[1],l,t]
            
            m.addConstr(sales_sum + L[p,l,t] == demand + inbound - outbound,
                       name=f"sales_conservation_{p}_{l}_{t}")

if data['network']['sub_edges']:
    for p in products:
        outgoing_edges = [edge for edge in data['network']['sub_edges'] if edge[0] == p]
        if outgoing_edges:
            for l in locations:
                for t in range(1, T + 1):
                    demand = data['demand_curve'][p][t-1] * data['demand_share'][l]
                    outbound_sum = sum(S[p,edge[1],l,t] for edge in outgoing_edges)
                    m.addConstr(outbound_sum <= demand, 
                               name=f"demand_route_{p}_{l}_{t}")

for l in locations:
    for t in range(1, T + 1):
        storage_sum = sum(data['cold_usage'][p] * sum(I[p,l,t,a] for a in range(1, data['shelf_life'][p] + 1)) 
                         for p in products)
        m.addConstr(storage_sum <= data['cold_capacity'][l], 
                   name=f"storage_cap_{l}_{t}")

for p in products:
    for t in range(1, T + 1):
        prod_sum = sum(Q[p,l,t] for l in locations)
        m.addConstr(prod_sum <= data['production_cap'][p][t-1], 
                   name=f"prod_cap_{p}_{t}")

m.setParam('OutputFlag', 0)
m.setParam('Threads', 1)
m.setParam('Seed', 0)
m.optimize()

print(f"Status: {m.status}")
if m.status == GRB.OPTIMAL:
    print(f"Objective: {m.objVal}")