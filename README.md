# ReLoop: RetailOpt-190 Benchmark and Codebase

ReLoop is the public codebase for a research project on **retail supply-chain optimization** and **LLM-based text-to-MILP modeling**.

This repository currently releases **RetailOpt-190**, a vertically focused **retail text-to-optimization benchmark**, together with a **universal retail MILP solver**, and is intended to eventually host the **full code for the ReLoop paper**, including:

- A universal retail MILP formulation implemented in Gurobi
- Scenario generators and synthetic retail data
- Evaluation pipelines for classical solvers
- Prompt templates and tooling for LLM-based optimization agents

At the moment, the focus is on the **RetailOpt-190 benchmark and reference solver**, which can already be used as an open testbed.

---

## 1. Repository Structure (current)

The current layout is:

```text
reloop/
├── solvers/
│   └── universal_retail_solver.py       # Universal Retail Solver (reference MILP)
├── tools/
│   └── retail_benchmark_generator.py    # Generator for 38 archetypes × 5 variations
├── scenarios/
│   └── retailopt_190/
│       ├── spec/
│       │   ├── retail_spec.md          # Structural and semantic specification
│       │   └── retail_prompts.md       # Prompt templates for LLM agents
│       ├── data/                       # JSON instances (RetailOpt-190 benchmark)
│       └── prompts/                    # Per-instance SYSTEM/USER prompt files
└── eval/
    ├── run_benchmark.py                # Batch evaluation script
    └── benchmark_results.csv           # Reference solver results (if generated)
````

Over time, additional modules (e.g., agent runners, experiment scripts) will be added to support the full set of experiments in the ReLoop paper.

---

## 2. RetailOpt-190 Benchmark (short description)

The `scenarios/retailopt_190/` tree contains **RetailOpt-190**, a synthetic retail operations benchmark designed for both

* classical optimization solvers, and
* LLM-based text-to-optimization agents.

Key points:

* **38 structural archetypes** grouped into 8 families
  (operations, assortment, resources, dynamics, feasibility, logistics, network, omni-channel).
* Each archetype has **5 numerical variants**, giving a total of **190 JSON instances**.
* All instances share a **single JSON schema** and are solved by the same universal MILP in `universal_retail_solver.py`.

For the **full specification and semantics** of the dataset, see:

* `scenarios/retailopt_190/spec/retail_spec.md` – retail stories, active mechanisms, and structural intent.
* `scenarios/retailopt_190/spec/retail_prompts.md` – prompt templates for LLM agents (system / user / repair prompts) and an example instance.

The per-instance JSON files live in:

* `scenarios/retailopt_190/data/`

The corresponding LLM prompts (one `.txt` per JSON) live in:

* `scenarios/retailopt_190/prompts/`

---

## 3. Installation

### 3.1 Requirements

* Python ≥ 3.9
* [Gurobi Optimizer](https://www.gurobi.com/) with a valid license
* Python packages:

  * `gurobipy`
  * `numpy`
  * `pandas`
  * `pyyaml`
  * (plus any extra utilities listed in `requirements.txt`, if present)

Install dependencies with:

```bash
pip install -r requirements.txt
```

or manually, e.g.:

```bash
pip install gurobipy numpy pandas pyyaml
```

---

## 4. Quick Start

### 4.1 Clone the repository

```bash
git clone https://github.com/junbolian/ReLoop.git
cd ReLoop
```

### 4.2 Solve a single JSON instance with the reference solver

The universal solver reads one retail JSON, builds the MILP, and calls Gurobi.

Example (adjust arguments if needed according to the script):

```bash
python -m reloop.solvers.universal_retail_solver \
  --json scenarios/retailopt_190/data/retail_f1_52_weeks_v0.json
```

The script will print (at least):

* solver status (e.g., `OPTIMAL`, `TIME_LIMIT`, `INFEASIBLE`)
* total objective value (e.g., total cost)

See `universal_retail_solver.py` for the exact command-line interface and options.

### 4.3 Run the full benchmark

To run the reference solver on **all** JSON files and write a summary CSV:

```bash
python -m reloop.eval.run_benchmark
```

This produces:

```text
reloop/eval/benchmark_results.csv
```

with at least the following columns:

* `scenario` – scenario name (JSON filename without `.json`)
* `status` – mapped solver status
* `objective` – total cost (or `N/A` if infeasible / no incumbent)

---

## 5. LLM Benchmarking (high level)

The repository is also designed to support **LLM-based text-to-MILP agents** on top of RetailOpt-190:

* A **system prompt** describes the agent’s role as a retail optimization modeling assistant.
* A **user prompt** for each JSON instance provides:

  * the family and archetype ID,
  * a business narrative and structural cues,
  * instructions on how to read the JSON via a Python variable `data`,
  * a request to output a plain Python script that builds and solves a MILP using `gurobipy`.

The exact prompt templates and an example user prompt are documented in:

* `scenarios/retailopt_190/spec/retail_prompts.md`

Per-instance prompt files under `scenarios/retailopt_190/prompts/` have the structure:

```text
### SYSTEM PROMPT ###
<system prompt text>

### USER PROMPT ###
<user prompt text>
```

An external evaluation harness can load these files, call an LLM, inject the JSON into the runtime as `data`, and compare the resulting solutions against the reference solver.

---

## 6. Roadmap

This repository is under active development. Planned additions include:

* Full experiment scripts for all results reported in the ReLoop paper
* Agent runners for multiple LLM providers
* Closed-loop (generator–verifier–repair) runners using IIS feedback
* Additional diagnostics and visualization tools

Once those components are public, this README will be updated with detailed “reproduce-the-paper” instructions.

---

## 7. Citation

If you use ReLoop or the **RetailOpt-190** benchmark in academic work, please cite the accompanying paper (once available). A temporary placeholder BibTeX entry is:

```bibtex
@misc{reloop2026,
  author       = {Yujun Sam Sun and Junbo Jacob Lian and Diego Klabjan},
  title        = {ReLoop: Solver-Guided LLMs for Semantically Reliable Text-to-Optimization in Retail Supply Chains},
  year         = {2026},
}
```

---

## 8. License

This repository is released for research and educational use.
Please refer to the `LICENSE` file for the exact terms and conditions. If no license is included yet, all rights are reserved by the authors until a license is added.
