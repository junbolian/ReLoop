import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except Exception:
    # Fallback for older langchain versions
    from langchain.agents.agent import AgentExecutor  # type: ignore
    from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from textwrap import dedent

from .prompt_loader import load_prompt_for_scenario
from .contract_checker import check_contract
from .script_executor import execute_script
from .semantic_check import semantic_check
from .agent_types import ExecutionResult, SemanticReport
from .logging_artifacts import ensure_dir, save_text, append_jsonl


def _build_reference_script(data: Dict[str, Any]) -> str:
    """Deterministic, schema-safe Gurobi starter script aligned with URS."""
    periods = int(data.get("periods", 0) or 0)
    products = list(data.get("products", []))
    locations = list(data.get("locations", []))
    template = """
        import gurobipy as gp
        from gurobipy import GRB

        data = globals()["data"]

        T = int(__PERIODS__)
        products = __PRODUCTS__
        locations = __LOCATIONS__

        shelf_life = {p: int(v) for p, v in data["shelf_life"].items()}
        cold_cap = data["cold_capacity"]
        cold_use = data["cold_usage"]
        prod_cap = data["production_cap"]

        lead_time = data.get("lead_time", {p: 0 for p in products})
        constraints = data.get("constraints", {})
        moq = constraints.get("moq", 0)
        pack_size = constraints.get("pack_size", 1)
        budget = constraints.get("budget_per_period", None)
        waste_limit_pct = constraints.get("waste_limit_pct", None)

        network = data.get("network", {})
        sub_edges = []
        for edge in network.get("sub_edges", []):
            if isinstance(edge, dict) and "from" in edge and "to" in edge:
                sub_edges.append((edge["from"], edge["to"]))
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                sub_edges.append((edge[0], edge[1]))
        trans_edges = []
        for edge in network.get("trans_edges", []):
            if isinstance(edge, dict) and "from" in edge and "to" in edge:
                trans_edges.append((edge["from"], edge["to"]))
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                trans_edges.append((edge[0], edge[1]))

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
        fixed_order_cost = costs.get("fixed_order", 0.0)
        trans_cost_unit = costs.get("transshipment", 0.5)
        purchase_cost = costs.get("purchasing", {p: 10.0 for p in products})

        demand_share = data.get("demand_share", {})
        demand = {}
        total_demand_vol = 0.0
        for p in products:
            curve = list(data["demand_curve"][p])
            if len(curve) < T:
                curve += [curve[-1]] * (T - len(curve))
            for l in locations:
                share = demand_share.get(l, 1.0 / max(1, len(locations)))
                for t in range(1, T + 1):
                    d_val = float(curve[t - 1]) * share
                    demand[(p, l, t)] = d_val
                    total_demand_vol += d_val

        m = gp.Model("universal_retail")
        m.setParam("OutputFlag", 0)
        m.setParam("TimeLimit", __TIME_LIMIT__)
        m.setParam("MIPGap", 0.01)

        X = {}
        Z = {}
        for p in products:
            for l in locations:
                for t in range(1, T + 1):
                    vtype = GRB.INTEGER if pack_size > 1 else GRB.CONTINUOUS
                    X[(p, l, t)] = m.addVar(lb=0.0, vtype=vtype, name=f"X_{p}_{l}_{t}")
                    if moq > 0 or fixed_order_cost > 0:
                        Z[(p, l, t)] = m.addVar(vtype=GRB.BINARY, name=f"Z_{p}_{l}_{t}")

        I = {}
        Y = {}
        W = {}
        L = {}
        for p in products:
            SL = shelf_life[p]
            for l in locations:
                for t in range(1, T + 1):
                    for a in range(1, SL + 1):
                        I[(p, l, t, a)] = m.addVar(lb=0.0, name=f"I_{p}_{l}_{t}_{a}")
                        Y[(p, l, t, a)] = m.addVar(lb=0.0, name=f"Y_{p}_{l}_{t}_{a}")
                    W[(p, l, t)] = m.addVar(lb=0.0, name=f"W_{p}_{l}_{t}")
                    L[(p, l, t)] = m.addVar(lb=0.0, name=f"L_{p}_{l}_{t}")

        TR = {}
        for p in products:
            for (src, dst) in trans_edges:
                for t in range(1, T + 1):
                    TR[(p, src, dst, t)] = m.addVar(
                        lb=0.0, name=f"TR_{p}_{src}_{dst}_{t}"
                    )

        S = {}
        for (pf, pt) in sub_edges:
            for l in locations:
                for t in range(1, T + 1):
                    S[(pf, pt, l, t)] = m.addVar(
                        lb=0.0, name=f"S_{pf}_{pt}_{l}_{t}"
                    )

        m.update()

        big_M = 1_000_000.0
        for p in products:
            for l in locations:
                for t in range(1, T + 1):
                    if pack_size > 1:
                        k = m.addVar(vtype=GRB.INTEGER, name=f"pack_{p}_{l}_{t}")
                        m.addConstr(
                            X[(p, l, t)] == pack_size * k,
                            name=f"pack_{p}_{l}_{t}",
                        )
                    if moq > 0 or fixed_order_cost > 0:
                        m.addConstr(
                            X[(p, l, t)] <= big_M * Z[(p, l, t)],
                            name=f"moq_ub_{p}_{l}_{t}",
                        )
                        if moq > 0:
                            m.addConstr(
                                X[(p, l, t)] >= moq * Z[(p, l, t)],
                                name=f"moq_lb_{p}_{l}_{t}",
                            )

        for p in products:
            SL = shelf_life[p]
            LT = lead_time.get(p, 0)
            rr = return_rate.get(p, 0.0)
            for l in locations:

                def get_arrival(t_cur):
                    t_ord = t_cur - LT
                    if t_ord >= 1:
                        return X[(p, l, t_ord)]
                    return 0.0

                def get_trans_net(t_cur):
                    inc = gp.quicksum(
                        TR[(p, src, l, t_cur)]
                        for (src, dst) in trans_edges
                        if dst == l
                    )
                    out = gp.quicksum(
                        TR[(p, l, dst, t_cur)]
                        for (src, dst) in trans_edges
                        if src == l
                    )
                    return inc - out

                def get_returns(t_cur):
                    if t_cur <= 1 or rr == 0.0:
                        return 0.0
                    sales_prev = gp.quicksum(
                        Y[(p, l, t_cur - 1, a)] for a in range(1, SL + 1)
                    )
                    return rr * sales_prev

                for t in range(1, T + 1):
                    arr = get_arrival(t)
                    tr = get_trans_net(t)
                    ret = get_returns(t)

                    m.addConstr(
                        I[(p, l, t, SL)] == arr + tr + ret - Y[(p, l, t, SL)],
                        name=f"aging_{p}_{l}_{t}_{SL}",
                    )

                    for a in range(1, SL):
                        prev = 0.0 if t == 1 else I[(p, l, t - 1, a + 1)]
                        m.addConstr(
                            I[(p, l, t, a)] == prev - Y[(p, l, t, a)],
                            name=f"aging_{p}_{l}_{t}_{a}",
                        )

                    prev_1 = 0.0 if t == 1 else I[(p, l, t - 1, 1)]
                    m.addConstr(
                        W[(p, l, t)] == prev_1 - Y[(p, l, t, 1)],
                        name=f"expire_clear_{p}_{l}_{t}",
                    )

        for t in range(1, T + 1):
            for p in products:
                cap_list = prod_cap[p]
                c = cap_list[t - 1] if t - 1 < len(cap_list) else cap_list[-1]
                m.addConstr(
                    gp.quicksum(X[(p, l, t)] for l in locations) <= c,
                    name=f"prod_cap_{p}_{t}",
                )

            for l in locations:
                labor_needed = gp.quicksum(
                    labor_use.get(p, 0.0)
                    * gp.quicksum(
                        Y[(p, l, t, a)] for a in range(1, shelf_life[p] + 1)
                    )
                    for p in products
                )
                lc_list = labor_cap[l]
                lc = lc_list[t - 1] if t - 1 < len(lc_list) else lc_list[-1]
                m.addConstr(labor_needed <= lc, name=f"labor_cap_{l}_{t}")

            for l in locations:
                usage = gp.quicksum(
                    cold_use[p]
                    * gp.quicksum(
                        I[(p, l, t, a)] for a in range(1, shelf_life[p] + 1)
                    )
                    for p in products
                )
                m.addConstr(usage <= cold_cap[l], name=f"storage_cap_{l}_{t}")

            if budget is not None:
                spend = gp.quicksum(
                    X[(p, l, t)]
                    * (purchase_cost.get(p, 10.0) if isinstance(purchase_cost, dict) else purchase_cost)
                    for p in products
                    for l in locations
                )
                if fixed_order_cost > 0.0:
                    spend += gp.quicksum(
                        fixed_order_cost * Z[(p, l, t)]
                        for p in products
                        for l in locations
                    )
                m.addConstr(spend <= budget, name=f"budget_{t}")

        if waste_limit_pct is not None:
            tot_waste = gp.quicksum(
                W[(p, l, t)]
                for p in products
                for l in locations
                for t in range(1, T + 1)
            )
            m.addConstr(tot_waste <= waste_limit_pct * total_demand_vol, name="wastecap")

        for p in products:
            SL = shelf_life[p]
            for l in locations:
                for t in range(1, T + 1):
                    outbound = gp.quicksum(
                        S[(p, pt, l, t)] for pt in outgoing_edges.get(p, [])
                    )
                    inbound = gp.quicksum(
                        S[(pf, p, l, t)] for pf in incoming_edges.get(p, [])
                    )
                    m.addConstr(
                        outbound <= demand[(p, l, t)],
                        name=f"demand_route_{p}_{l}_{t}",
                    )
                    total_sales = gp.quicksum(
                        Y[(p, l, t, a)] for a in range(1, SL + 1)
                    )
                    m.addConstr(
                        total_sales + L[(p, l, t)]
                        == demand[(p, l, t)] + inbound - outbound,
                        name=f"sales_conservation_{p}_{l}_{t}",
                    )

        obj = 0.0
        for p in products:
            SL = shelf_life[p]
            for l in locations:
                for t in range(1, T + 1):
                    obj += inv_cost.get(p, 0.0) * gp.quicksum(
                        I[(p, l, t, a)] for a in range(1, SL + 1)
                    )
                    obj += waste_cost.get(p, 0.0) * W[(p, l, t)]
                    obj += lost_pen.get(p, 0.0) * L[(p, l, t)]
                    if fixed_order_cost > 0.0 and (p, l, t) in Z:
                        obj += fixed_order_cost * Z[(p, l, t)]

        for p in products:
            for (src, dst) in trans_edges:
                for t in range(1, T + 1):
                    obj += trans_cost_unit * TR[(p, src, dst, t)]

        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()

        print(m.Status)
        if m.SolCount > 0:
            print(m.objVal)
        """
    return (
        dedent(template)
        .replace("__PERIODS__", str(periods))
        .replace("__PRODUCTS__", repr(products))
        .replace("__LOCATIONS__", repr(locations))
        .replace("__TIME_LIMIT__", "60")
        .strip()
    )


def _default_llm():
    # Uses OpenAI-compatible endpoint. Qwen users set:
    # OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    # OPENAI_MODEL=qwen-plus-latest
    # OPENAI_API_KEY (or DASHSCOPE_API_KEY)
    return ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        ),
        api_key=os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY"),
        temperature=0.2,
    )


def build_tools(
    data: Dict[str, Any],
    data_path: str,
    request_iis: bool,
    timeout_s: float,
    run_root: Optional[str] = None,
):
    def _schema_summary():
        summary = {
            "periods": data.get("periods"),
            "products": data.get("products", []),
            "locations": data.get("locations", []),
            "shelf_life_type": type(data.get("shelf_life")).__name__,
            "shelf_life_keys": list(data.get("shelf_life", {}).keys()),
            "demand_curve_type": type(data.get("demand_curve")).__name__,
            "production_cap_type": type(data.get("production_cap")).__name__,
            "constraints": data.get("constraints", {}),
            "network_keys": list(data.get("network", {}).keys()),
        }
        # Peek into first product for demand/production shapes
        products = data.get("products", [])
        if products:
            p0 = products[0]
            dc = data.get("demand_curve", {}).get(p0)
            pc = data.get("production_cap", {}).get(p0)
            summary["demand_curve_sample"] = {
                "product": p0,
                "type": type(dc).__name__,
                "len": len(dc) if hasattr(dc, "__len__") else None,
            }
            summary["production_cap_sample"] = {
                "product": p0,
                "type": type(pc).__name__,
                "len": len(pc) if hasattr(pc, "__len__") else None,
            }
        return summary

    @tool("sandbox_execute", return_direct=False)
    def sandbox_execute(code: str) -> str:
        """
        Execute a gurobipy script in a sandbox with `data` preloaded.
        Returns JSON with compilation status, feasibility, objective, and IIS/semantic info.
        """
        contract = check_contract(code)
        if not contract.ok:
            return json.dumps(
                {
                    "ok": False,
                    "reason": "contract violation",
                    "details": contract.reasons,
                }
            )
        prelude = f"""
from pathlib import Path
import json

data_path = Path({repr(str(data_path))})
if not data_path.is_absolute():
    data_path = (Path.cwd() / data_path).resolve()

with data_path.open("r", encoding="utf-8") as f:
    data = json.load(f)
"""

        if run_root:
            ensure_dir(run_root)
            save_text(os.path.join(run_root, "llm_generated.py"), prelude + code)
        result: ExecutionResult = execute_script(
            prelude + code, None, timeout_s=timeout_s, request_iis=request_iis
        )
        sem: SemanticReport = semantic_check(data, result)
        payload = {
            "ok": result.success,
            "status": result.status_str,
            "objective": result.objective,
            "feasible": result.feasible,
            "semantic_valid": sem.valid,
            "missing": sem.missing_prefixes,
            "unexpected": sem.unexpected_modules,
            "stderr": result.stderr,
            "traceback": result.traceback,
            "iis": result.iis_constraints[:10],
            "schema_hint": _schema_summary(),
        }
        return json.dumps(payload)

    @tool("show_schema", return_direct=False)
    def show_schema(_: str = "") -> str:
        """Show the JSON keys available in `data` (for quick inspection)."""
        return json.dumps({"keys": list(data.keys())})

    @tool("get_reference_script", return_direct=False)
    def get_reference_script(_: str = "") -> str:
        """Return a robust starter gurobipy script that matches the RetailOpt schema."""
        return _build_reference_script(data)

    return [sandbox_execute, show_schema, get_reference_script]


def run_langchain_agent(
    scenario_id: str,
    data_path: str,
    prompts_dir: str,
    workdir: str,
    max_iters: int = 6,
    request_iis: bool = True,
    timeout_s: float = 60.0,
    llm: Any = None,
) -> Dict[str, Any]:
    system_prompt, user_prompt = load_prompt_for_scenario(scenario_id, prompts_dir)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    run_root = os.path.join(workdir, scenario_id, time.strftime("%Y%m%d_%H%M%S"))
    ensure_dir(run_root)

    llm = llm or _default_llm()
    tools = build_tools(
        data, data_path, request_iis=request_iis, timeout_s=timeout_s, run_root=run_root
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", user_prompt),
            (
                "system",
                "Schema: shelf_life is dict product->int; demand_curve and production_cap are dict->list with length periods; "
                "cold_capacity/cold_usage are dicts by location/product; network.sub_edges and network.trans_edges may appear; "
                "lead_time and return_rate can be product-indexed; labor_cap is location-indexed by period. "
                "Use tools (get_reference_script, sandbox_execute, show_schema) to build, run, and repair the MILP script. "
                "Start by calling get_reference_script to get a schema-safe template, then run it via sandbox_execute; "
                "iterate until feasible and semantically valid.",
            ),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=max_iters,
        return_intermediate_steps=True,
    )

    messages_path = os.path.join(run_root, "messages.jsonl")
    trace_path = os.path.join(run_root, "training_trace.jsonl")

    start = time.time()
    result = executor.invoke({"input": "Solve this scenario with valid gurobipy code."})
    duration = time.time() - start

    # Log intermediate steps as conversation
    steps: List[Tuple[Any, str]] = result.get("intermediate_steps", [])
    records = []
    round_id = 0
    for action, observation in steps:
        round_id += 1
        records.append(
            {
                "round": round_id,
                "tool": getattr(action, "tool", ""),
                "tool_input": getattr(action, "tool_input", ""),
                "observation": observation,
                "log": getattr(action, "log", ""),
                "timestamp": time.time(),
            }
    )
    append_jsonl(trace_path, records)
    append_jsonl(messages_path, records)

    # Persist the latest executed gurobi script for manual reruns.
    final_code = None
    for rec in reversed(records):
        if rec.get("tool") == "sandbox_execute":
            ti = rec.get("tool_input")
            if isinstance(ti, dict) and "code" in ti:
                final_code = ti["code"]
                break
    if final_code:
        # Emit a runnable script that self-loads the scenario data, so it can be
        # executed directly with `python llm_generated.py`.
        prelude = (
            "from pathlib import Path\n"
            "import json\n\n"
            f"data_path = Path({json.dumps(data_path)})\n"
            "with data_path.open('r', encoding='utf-8') as f:\n"
            "    data = json.load(f)\n\n"
        )
        save_text(os.path.join(run_root, "llm_generated.py"), prelude + final_code)

    # Save final output
    output = {
        "scenario_id": scenario_id,
        "output": result.get("output", ""),
        "intermediate_steps": records,
        "runtime_s": duration,
    }
    save_text(os.path.join(run_root, "summary.json"), json.dumps(output, indent=2))
    return output
