from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI


@dataclass
class LLMResponse:
    content: str
    raw: Any
    usage: Dict[str, Any]


class LLMClient:
    """Abstract adapter used by the agent stack."""

    def complete(self, messages: List[BaseMessage], **kwargs) -> LLMResponse:  # pragma: no cover - interface only
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Deterministic mock that returns schema-valid placeholders per step for offline runs."""

    def complete(self, messages: List[BaseMessage], **kwargs) -> LLMResponse:
        step = None
        for msg in messages:
            text = getattr(msg, "content", "")
            if "STEP 0" in text:
                step = "step0"
            elif "STEP 1" in text:
                step = "step1"
            elif "STEP 2" in text:
                step = "step2"
            elif "STEP 3" in text:
                step = "step3"
            elif "STEP 5" in text:
                step = "step5"
            elif "STEP 6" in text:
                step = "step6"
        content = ""
        if step == "step0":
            content = json.dumps(
                {
                    "optimize": "minimize total cost",
                    "controls": ["orders", "inventory"],
                    "hard_constraints": ["capacity", "flow balance"],
                    "soft_violations": [
                        {"name": "lost sales", "penalty_source": "costs.lost_sales"}
                    ],
                    "contract_summary": "capacity hard, lost sales penalized",
                }
            )
        elif step == "step1":
            content = json.dumps(
                [
                    {
                        "id": "S1",
                        "text": "Demand must be met or penalized.",
                        "tag": "RULE_CONSTRAINT_SOURCE",
                        "extracted": {
                            "sets": [],
                            "params": ["demand"],
                            "decisions": ["L"],
                            "rule_hints": ["lost sales allowed"],
                        },
                    }
                ]
            )
        elif step == "step2":
            content = json.dumps(
                {
                    "sets": [
                        {"name": "P", "description": "products", "source": "data.products"},
                        {"name": "L", "description": "locations", "source": "data.locations"},
                        {"name": "T", "description": "periods index 0..T-1", "source": "data.periods"},
                        {
                            "name": "A",
                            "description": "remaining life index per product if shelf_life active",
                            "source": "data.shelf_life",
                        },
                    ],
                    "decisions": [
                        {
                            "name": "I",
                            "type": "continuous",
                            "domain": ">=0",
                            "indices": ["p", "l", "t", "a"],
                            "meaning": "inventory by remaining life",
                            "active_if": "shelf_life",
                        },
                        {
                            "name": "y",
                            "type": "continuous",
                            "domain": ">=0",
                            "indices": ["p", "l", "t", "a"],
                            "meaning": "sales/consumption by remaining life",
                            "active_if": "shelf_life",
                        },
                        {
                            "name": "W",
                            "type": "continuous",
                            "domain": ">=0",
                            "indices": ["p", "l", "t"],
                            "meaning": "waste/disposal",
                            "active_if": "always",
                        },
                        {
                            "name": "Q",
                            "type": "continuous",
                            "domain": ">=0",
                            "indices": ["p", "l", "t"],
                            "meaning": "orders/production allocated to location",
                            "active_if": "always",
                        },
                        {
                            "name": "L",
                            "type": "continuous",
                            "domain": ">=0",
                            "indices": ["p", "l", "t"],
                            "meaning": "lost sales",
                            "active_if": "lost_sales_allowed",
                        },
                        {
                            "name": "d",
                            "type": "continuous",
                            "domain": ">=0",
                            "indices": ["p", "l", "t"],
                            "meaning": "direct demand served by same product",
                            "active_if": "always",
                        },
                        {
                            "name": "S",
                            "type": "continuous",
                            "domain": ">=0",
                            "indices": ["p_from", "p_to", "l", "t"],
                            "meaning": "substitution flow serving p_to using p_from",
                            "active_if": "sub_edges_nonempty",
                        },
                        {
                            "name": "X",
                            "type": "continuous",
                            "domain": ">=0",
                            "indices": ["l_from", "l_to", "p", "t"],
                            "meaning": "transshipment flow",
                            "active_if": "trans_edges_nonempty",
                        },
                        {
                            "name": "z",
                            "type": "binary",
                            "domain": "{0,1}",
                            "indices": ["p", "l", "t"],
                            "meaning": "order trigger for fixed cost / MOQ",
                            "active_if": "moq_or_fixed_order",
                        },
                        {
                            "name": "n",
                            "type": "integer",
                            "domain": ">=0",
                            "indices": ["p", "l", "t"],
                            "meaning": "pack multiple",
                            "active_if": "pack_size",
                        },
                    ],
                    "objective_terms": [],
                    "constraint_families": [],
                    "edge_cases": [{"case": "t0_init", "handling": "assume zero if absent"}],
                    "open_questions": [
                        {
                            "question": "Define waste_limit_pct base; default demand if unspecified."
                        }
                    ],
                }
            )
        elif step == "step3":
            content = json.dumps(
                [
                    {
                        "prefix": "demand_route",
                        "template_type": "NETWORK",
                        "applies_when": "demand present",
                        "indices": ["p", "l", "t"],
                        "equations": [
                            {
                                "name_suffix": "p_l_t",
                                "sense": "=",
                                "lhs": "d[p,l,t] + S[p_from,p,l,t] + L[p,l,t]",
                                "rhs": "demand[p,l,t]",
                            }
                        ],
                        "notes": ["placeholder template"],
                    },
                    {
                        "prefix": "availability",
                        "template_type": "BALANCE",
                        "applies_when": "substitution",
                        "indices": ["p", "l", "t", "a"],
                        "equations": [
                            {
                                "name_suffix": "p_l_t_a",
                                "sense": ">=",
                                "lhs": "I[p,l,t,a]",
                                "rhs": "y[p,l,t,a] + S[p,l_to,l,t]",
                            }
                        ],
                        "notes": ["links substitution to supplying product consumption"],
                    },
                ]
            )
        elif step == "step5":
            content = _mock_script()
        elif step == "step6":
            content = json.dumps(
                {
                    "target": "CODEGEN",
                    "diagnosis": {
                        "category": "INFEASIBLE",
                        "most_likely_causes": ["mock"],
                        "evidence": [],
                        "affected_constraint_prefixes": [],
                    },
                    "repairs": [
                        {
                            "change_type": "ADD",
                            "where": "codegen_instructions",
                            "description": "placeholder",
                            "acceptance_test": "mock passes",
                        }
                    ],
                    "next_step_prompting": {"extra_instructions_to_inject": []},
                }
            )
        else:
            # Fallback echo
            last_human = ""
            for msg in reversed(messages):
                if getattr(msg, "type", getattr(msg, "role", "")) == "human":
                    last_human = msg.content
                    break
            content = last_human
        return LLMResponse(content=content, raw={"mock": True, "step": step}, usage={"tokens": 0})


class OpenAILLMClient(LLMClient):
    """OpenAI-compatible LangChain chat model adapter."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ):
        llm_model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        extra = {}
        if base_url:
            extra["base_url"] = base_url
        self.client = ChatOpenAI(
            model=llm_model, temperature=temperature, max_tokens=max_tokens, api_key=api_key, **extra
        )

    def complete(self, messages: List[BaseMessage], **kwargs) -> LLMResponse:
        response = self.client.invoke(messages, **kwargs)
        text = getattr(response, "content", "") or str(response)
        usage = getattr(response, "response_metadata", {})
        return LLMResponse(content=text, raw=response, usage=usage)


def build_llm_client(mode: str = "openai", **kwargs) -> LLMClient:
    if mode == "mock":
        return MockLLMClient()
    return OpenAILLMClient(**kwargs)


def _mock_script() -> str:
    return """import sys
try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    class _DummyGRB:
        OPTIMAL=2
        INFEASIBLE=3
        INF_OR_UNBD=4
        UNBOUNDED=5
    GRB=_DummyGRB()
    class _DummyModel:
        def __init__(self):
            self.Params=type("Params", (), {})()
            self.Params.OutputFlag=0
            self.Params.Threads=1
            self.Params.Seed=0
            self.Status=GRB.OPTIMAL
            self.objVal=0
            self.objBound=0
            self.MIPGap=0
        def optimize(self): pass
    gp=type("gp", (), {"Model": _DummyModel, "quicksum": lambda x: 0})

m = gp.Model()
m.Params.OutputFlag = 0
m.Params.Threads = 1
m.Params.Seed = 0

# minimal variables/constraints to satisfy naming contract
I = {}
y = {}
W = {}
Q = {}
L = {}
d = {}
S = {}
X = {}
z = {}
n = {}

m.optimize()
print(f"status={m.Status}")
if hasattr(m, 'objVal'):
    if hasattr(GRB, 'OPTIMAL') and m.Status == getattr(GRB, 'OPTIMAL', None):
        print(f"obj={getattr(m, 'objVal', 0)}")
    else:
        if hasattr(m, 'objVal'):
            print(f"obj={m.objVal}")
        if hasattr(m, 'objBound'):
            print(f"objbound={getattr(m, 'objBound', '')}")
        if hasattr(m, 'MIPGap'):
            print(f"mipgap={getattr(m, 'MIPGap', '')}")
"""
