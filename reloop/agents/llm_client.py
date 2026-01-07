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
    """
    Deterministic mock that returns schema-valid placeholders per step for offline runs.
    
    Updated for new 5-step pipeline:
    - step0: Contract
    - step1: Spec Sheet (was step2)
    - step2: Constraint Templates (was step3)
    - step3: Sanity Check (was step4)
    - step4: Codegen (was step5)
    - step5: Repair Brief (was step6)
    """

    def complete(self, messages: List[BaseMessage], **kwargs) -> LLMResponse:
        step = None
        for msg in messages:
            text = getattr(msg, "content", "")
            if "STEP 0" in text or "step0" in text.lower():
                step = "step0"
            elif "STEP 1" in text or "step1" in text.lower() or "SPEC SHEET" in text.upper():
                step = "step1"
            elif "STEP 2" in text or "step2" in text.lower() or "CONSTRAINT TEMPLATE" in text.upper():
                step = "step2"
            elif "STEP 3" in text or "step3" in text.lower() or "SANITY" in text.upper():
                step = "step3"
            elif "STEP 4" in text or "step4" in text.lower() or "CODEGEN" in text.upper():
                step = "step4"
            elif "STEP 5" in text or "step5" in text.lower() or "REPAIR" in text.upper():
                step = "step5"
        
        content = ""
        
        if step == "step0":
            # Contract extraction
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
            # Spec Sheet (was step2)
            content = json.dumps(
                {
                    "sets": [
                        {"name": "P", "description": "products", "source": "data.products"},
                        {"name": "L", "description": "locations", "source": "data.locations"},
                        {"name": "T", "description": "periods index 1..T", "source": "range(1, data['periods']+1)"},
                        {"name": "A", "description": "remaining life index", "source": "range(1, max(shelf_life)+1)"},
                    ],
                    "decisions": [
                        {
                            "name": "I",
                            "type": "continuous",
                            "domain": ">=0",
                            "indices": ["p", "l", "t", "a"],
                            "meaning": "inventory by remaining life",
                            "active_if": "shelf_life active",
                        },
                        {
                            "name": "y",
                            "type": "continuous",
                            "domain": ">=0",
                            "indices": ["p", "l", "t", "a"],
                            "meaning": "sales by remaining life",
                            "active_if": "shelf_life active",
                        },
                        {
                            "name": "W",
                            "type": "continuous",
                            "domain": ">=0",
                            "indices": ["p", "l", "t"],
                            "meaning": "waste",
                            "active_if": "always",
                        },
                        {
                            "name": "Q",
                            "type": "continuous",
                            "domain": ">=0",
                            "indices": ["p", "l", "t"],
                            "meaning": "orders/production",
                            "active_if": "always",
                        },
                        {
                            "name": "L",
                            "type": "continuous",
                            "domain": ">=0",
                            "indices": ["p", "l", "t"],
                            "meaning": "lost sales",
                            "active_if": "always",
                        },
                        {
                            "name": "S",
                            "type": "continuous",
                            "domain": ">=0",
                            "indices": ["p_from", "p_to", "l", "t"],
                            "meaning": "substitution flow",
                            "active_if": "sub_edges nonempty",
                        },
                    ],
                    "objective_terms": [
                        {"name": "holding", "expression": "sum I * c_inv", "source": "costs.inventory", "active_if": "always"},
                        {"name": "waste", "expression": "sum W * c_waste", "source": "costs.waste", "active_if": "always"},
                        {"name": "lost_sales", "expression": "sum L * c_lost", "source": "costs.lost_sales", "active_if": "always"},
                    ],
                    "constraint_families": [
                        {"prefix": "demand_route", "meaning": "route demand", "indices": ["p", "l", "t"], "sense": "==", "active_if": "always"},
                        {"prefix": "availability", "meaning": "sales <= inventory", "indices": ["p", "l", "t", "a"], "sense": "<=", "active_if": "always"},
                    ],
                    "edge_cases": [{"case": "t=1 init", "handling": "non-fresh buckets = 0"}],
                    "open_questions": [],
                }
            )
        
        elif step == "step2":
            # Constraint Templates (was step3)
            content = json.dumps(
                [
                    {
                        "prefix": "demand_route",
                        "template_type": "SUBSTITUTION",
                        "applies_when": "sub_edges non-empty",
                        "indices": ["p", "l", "t"],
                        "equations": [
                            {
                                "name_suffix": "",
                                "sense": "<=",
                                "lhs": "sum S_out[p]",
                                "rhs": "demand[p,l,t]",
                            }
                        ],
                        "notes": ["Outbound substitution <= own demand"],
                    },
                    {
                        "prefix": "sales_conservation",
                        "template_type": "BALANCE",
                        "applies_when": "sub_edges non-empty",
                        "indices": ["p", "l", "t"],
                        "equations": [
                            {
                                "name_suffix": "",
                                "sense": "==",
                                "lhs": "y[p] + L[p]",
                                "rhs": "demand[p] + S_in[p] - S_out[p]",
                            }
                        ],
                        "notes": ["Sales balance with substitution"],
                    },
                    {
                        "prefix": "availability",
                        "template_type": "CAPACITY",
                        "applies_when": "always",
                        "indices": ["p", "l", "t", "a"],
                        "equations": [
                            {
                                "name_suffix": "",
                                "sense": "<=",
                                "lhs": "y[p,l,t,a]",
                                "rhs": "I[p,l,t,a]",
                            }
                        ],
                        "notes": ["Sales limited by inventory"],
                    },
                ]
            )
        
        elif step == "step4":
            # Codegen (was step5)
            content = _mock_script()
        
        elif step == "step5":
            # Repair Brief (was step6)
            content = json.dumps(
                {
                    "target": "CODEGEN",
                    "diagnosis": {
                        "category": "PROBE_FAILURE",
                        "most_likely_causes": ["substitution constraint missing"],
                        "evidence": ["demand_route_constraint probe failed"],
                        "affected_constraint_prefixes": ["demand_route"],
                        "failed_probes": ["demand_route_constraint"],
                    },
                    "repairs": [
                        {
                            "change_type": "ADD",
                            "where": "codegen_instructions",
                            "description": "Add S_out <= demand constraint",
                            "acceptance_test": "demand_route_constraint probe passes",
                        }
                    ],
                    "next_step_prompting": {"extra_instructions_to_inject": ["Ensure S_out <= demand is enforced"]},
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


class LocalLLMClient(LLMClient):
    """
    Local LLM client using HuggingFace Transformers.
    Supports multi-GPU inference via device_map="auto".
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Local models require `torch`, `transformers` and `accelerate`. "
                "Please install them via pip."
            )

        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens or 4096
        
        print(f"Loading local model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
        print("Model loaded successfully.")

    def complete(self, messages: List[BaseMessage], **kwargs) -> LLMResponse:
        # Convert LangChain messages to chat format
        chat = []
        for msg in messages:
            role = "user"
            msg_type = getattr(msg, "type", getattr(msg, "role", ""))
            if msg_type == "ai":
                role = "assistant"
            elif msg_type == "system":
                role = "system"
            elif msg_type == "human":
                role = "user"
            
            chat.append({"role": role, "content": getattr(msg, "content", "")})

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Setting enable_thinking=False disables thinking mode
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        response_ids = self.model.generate(
            **inputs, 
            max_new_tokens=self.max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False  # Keep deterministic for now, can be changed if needed
        )[0][len(inputs.input_ids[0]):]

        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Usage stats (approximate)
        input_tokens = len(inputs.input_ids[0])
        output_tokens = len(response_ids)
        
        # Usage stats (approximate)
        input_tokens = len(inputs.input_ids[0])
        output_tokens = len(response_ids)
        
        return LLMResponse(
            content=response_text,
            raw=None, 
            usage={
                "prompt_tokens": input_tokens, 
                "completion_tokens": output_tokens, 
                "total_tokens": input_tokens + output_tokens
            }
        )


def build_llm_client(mode: str = "openai", **kwargs) -> LLMClient:
    if mode == "mock":
        return MockLLMClient()
    if mode == "local":
        model_path = kwargs.pop("model", None)
        if not model_path:
            raise ValueError("Must provide `model` path when using mode='local'")
        return LocalLLMClient(model_path=model_path, **kwargs)
    return OpenAILLMClient(**kwargs)


def _mock_script() -> str:
    """Generate a minimal valid script for mock testing."""
    return """import gurobipy as gp
from gurobipy import GRB

# Extract data
T = int(data["periods"])
products = list(data["products"])
locations = list(data["locations"])
shelf_life = {p: int(data["shelf_life"][p]) for p in products}

demand_curve = data["demand_curve"]
demand_share = data["demand_share"]
production_cap = data["production_cap"]
costs = data["costs"]

# Create model
m = gp.Model("retail_mock")
m.setParam("OutputFlag", 0)
m.setParam("Threads", 1)
m.setParam("Seed", 0)

# Variables
I = {}
y = {}
W = {}
Q = {}
L = {}
S = {}

for p in products:
    A_p = shelf_life[p]
    for l in locations:
        for t in range(1, T + 2):
            for a in range(1, A_p + 1):
                I[p, l, t, a] = m.addVar(lb=0, name=f"I_{p}_{l}_{t}_{a}")
                y[p, l, t, a] = m.addVar(lb=0, name=f"y_{p}_{l}_{t}_{a}")

for p in products:
    for l in locations:
        for t in range(1, T + 1):
            W[p, l, t] = m.addVar(lb=0, name=f"W_{p}_{l}_{t}")
            L[p, l, t] = m.addVar(lb=0, name=f"L_{p}_{l}_{t}")
            Q[p, l, t] = m.addVar(lb=0, name=f"Q_{p}_{l}_{t}")

# Minimal constraints and objective
m.setObjective(0, GRB.MINIMIZE)
m.optimize()

print(f"status: {m.Status}")
if m.Status == GRB.OPTIMAL:
    print(f"objective: {m.ObjVal}")
else:
    print(f"ObjBound: {getattr(m, 'ObjBound', 'N/A')}")
    print(f"MIPGap: {getattr(m, 'MIPGap', 'N/A')}")
"""