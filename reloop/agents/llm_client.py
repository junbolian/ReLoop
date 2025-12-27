import os
import time
import json
import requests
from typing import List, Dict, Any

from .agent_types import LLMMessage, LLMResponse


def _approx_tokens(text: str) -> int:
    # Very rough heuristic; keeps accounting simple without extra deps.
    return max(1, len(text) // 4)


class LLMClient:
    """Abstract LLM client."""

    def complete(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Deterministic mock client that returns a minimal gurobipy script."""

    def complete(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        script = _mock_solver_script()
        tokens_in = sum(_approx_tokens(m.content) for m in messages)
        tokens_out = _approx_tokens(script)
        return LLMResponse(content=script, tokens_in=tokens_in, tokens_out=tokens_out)


class OpenAIClient(LLMClient):
    """OpenAI-compatible chat completions."""

    def __init__(self):
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = os.environ.get("OPENAI_MODEL", "")
        # Allow DashScope-style key as fallback for Qwen compatible endpoint.
        self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
            "DASHSCOPE_API_KEY"
        )
        self.timeout = float(os.environ.get("OPENAI_TIMEOUT_S", "60"))
        self.max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "2"))

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        last_exc = None
        for _ in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    url, headers=headers, json=payload, timeout=self.timeout
                )
                if resp.status_code != 200:
                    last_exc = RuntimeError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                    continue
                return resp.json()
            except Exception as exc:
                last_exc = exc
                time.sleep(1.0)
        raise last_exc

    def complete(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set it for real runs or use the mock client."
            )
        if not self.model:
            raise RuntimeError(
                "OPENAI_MODEL is not set. Example: export OPENAI_MODEL=gpt-4o-mini"
            )

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": kwargs.get("temperature", 0.2),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        raw = self._post(payload)
        choice = raw["choices"][0]["message"]["content"]
        usage = raw.get("usage", {})
        return LLMResponse(
            content=choice,
            tokens_in=usage.get("prompt_tokens", _approx_tokens(json.dumps(payload))),
            tokens_out=usage.get("completion_tokens", _approx_tokens(choice)),
            raw=raw,
        )


def build_llm_client(mode: str) -> LLMClient:
    if mode == "mock":
        return MockLLMClient()
    return OpenAIClient()


def _mock_solver_script() -> str:
    """Tiny gurobipy script designed to satisfy naming/semantic checks quickly."""
    return """import gurobipy as gp
from gurobipy import GRB

T = int(data.get("periods", 1))
products = list(data.get("products", []))
locations = list(data.get("locations", []))
network = data.get("network", {})
sub_edges = [tuple(e) for e in network.get("sub_edges", [])]
trans_edges = [tuple(e) for e in network.get("trans_edges", [])]
constraints = data.get("constraints", {})
moq = constraints.get("moq", 0)
pack_size = constraints.get("pack_size", 1)
budget = constraints.get("budget_per_period")
wastecap = constraints.get("waste_limit_pct")

m = gp.Model("mock_solver")
m.Params.OutputFlag = 0

# Decision variables
I = {}
y = {}
W = {}
Q = {}
L = {}
S = {}
X = {}
z = {}
n = {}
for p in products:
    sl = int(data.get("shelf_life", {}).get(p, 1))
    for l in locations:
        for t in range(1, T + 1):
            for a in range(1, sl + 1):
                I[(p,l,t,a)] = m.addVar(lb=0, name=f"I_{p}_{l}_{t}_{a}")
                y[(p,l,t,a)] = m.addVar(lb=0, name=f"y_{p}_{l}_{t}_{a}")
            W[(p,l,t)] = m.addVar(lb=0, name=f"W_{p}_{l}_{t}")
            L[(p,l,t)] = m.addVar(lb=0, name=f"L_{p}_{l}_{t}")
            Q[(p,l,t)] = m.addVar(lb=0, name=f"Q_{p}_{l}_{t}")
            if moq or constraints.get("fixed_order", 0) or constraints.get("fixed_order_cost", 0):
                z[(p,l,t)] = m.addVar(vtype=GRB.BINARY, name=f"z_{p}_{l}_{t}")
            if pack_size and pack_size > 1:
                n[(p,l,t)] = m.addVar(vtype=GRB.INTEGER, name=f"n_{p}_{l}_{t}")
    for (src,dst) in trans_edges:
        for t in range(1, T + 1):
            X[(p,src,dst,t)] = m.addVar(lb=0, name=f"X_{p}_{src}_{dst}_{t}")
    for (pf,pt) in sub_edges:
        for l in locations:
            for t in range(1, T + 1):
                S[(pf,pt,l,t)] = m.addVar(lb=0, name=f"S_{pf}_{pt}_{l}_{t}")

m.update()

# Simple constraints with naming prefixes for semantic checks.
for p in products:
    sl = int(data.get("shelf_life", {}).get(p, 1))
    for l in locations:
        for t in range(1, T + 1):
            if sl > 1:
                m.addConstr(gp.quicksum(I[(p,l,t,a)] for a in range(1, sl+1)) >= 0, name=f"aging_{p}_{l}_{t}")
                m.addConstr(W[(p,l,t)] >= 0, name=f"expire_clear_{p}_{l}_{t}")
            m.addConstr(gp.quicksum(y[(p,l,t,a)] for a in range(1, sl+1)) + L[(p,l,t)] >= 0, name=f"sales_conservation_{p}_{l}_{t}")
            m.addConstr(Q[(p,l,t)] >= 0, name=f"availability_{p}_{l}_{t}")

for (pf,pt) in sub_edges:
    for l in locations:
        for t in range(1, T + 1):
            m.addConstr(S[(pf,pt,l,t)] >= 0, name=f"demand_route_{pf}_{pt}_{l}_{t}")

if trans_edges:
    for p in products:
        for (src,dst) in trans_edges:
            for t in range(1, T + 1):
                m.addConstr(X[(p,src,dst,t)] >= 0, name=f"transshipment_{p}_{src}_{dst}_{t}")

for p in products:
    for t in range(1, T + 1):
        prod_cap = data.get("production_cap", {}).get(p, [])
        cap_val = prod_cap[t-1] if t-1 < len(prod_cap) else None
        if cap_val is not None:
            m.addConstr(gp.quicksum(Q[(p,l,t)] for l in locations) <= cap_val, name=f"prod_cap_{p}_{t}")

for l in locations:
    for t in range(1, T + 1):
        usage = []
        cold_usage = data.get("cold_usage", {})
        cap = data.get("cold_capacity", {}).get(l)
        if cap is None:
            continue
        for p in products:
            sl = int(data.get("shelf_life", {}).get(p, 1))
            usage.append(cold_usage.get(p, 0) * gp.quicksum(I[(p,l,t,a)] for a in range(1, sl+1)))
        if usage:
            m.addConstr(gp.quicksum(usage) <= cap, name=f"storage_cap_{l}_{t}")

if budget is not None:
    for t in range(1, T + 1):
        m.addConstr(gp.quicksum(Q[(p,l,t)] for p in products for l in locations) <= budget, name=f"budget_{t}")

if wastecap is not None:
    m.addConstr(gp.quicksum(W.values()) <= wastecap * max(1.0, float(T)), name="wastecap_global")

for (p,l,t), var in list(Q.items()):
    if moq:
        if (p,l,t) in z:
            m.addConstr(var >= moq * z[(p,l,t)], name=f"moq_lb_{p}_{l}_{t}")
            m.addConstr(var <= 1e6 * z[(p,l,t)], name=f"moq_ub_{p}_{l}_{t}")
    if pack_size and pack_size > 1 and (p,l,t) in n:
        m.addConstr(var == pack_size * n[(p,l,t)], name=f"pack_{p}_{l}_{t}")

objective = gp.quicksum(var for var in Q.values()) * 0.0
m.setObjective(objective, GRB.MINIMIZE)
m.optimize()

print(f"status={m.Status}")
if m.SolCount > 0:
    print(f"obj={m.objVal}")
"""
