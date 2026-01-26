import json
import sys
sys.path.insert(0, 'e:/reloop')
from solvers.universal_retail_solver import solve_scenario

with open('e:/reloop/scenarios/data/retail_f1_base_v0.json') as f:
    data = json.load(f)

obj, status, _ = solve_scenario(data, summarize=False)
print(f'Ground Truth Optimal: {obj:.2f}')
print(f'Status: {status}')
