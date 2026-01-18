# ReLoop-SFT: Training Data for Supply Chain Optimization Agents

SFT (Supervised Fine-Tuning) dataset for training retail supply chain optimization agents.

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total Samples | **14,125** |
| Scenarios | 2,500 |
| Industries | 18 |
| Feature Combinations | 48 |
| Data Format | JSONL (Chat Format) |

## Files

| File | Samples | Task Type |
|------|---------|-----------|
| `v2_step1_contract.jsonl` | 2,500 | Task contract extraction |
| `v2_step2_spec_sheet.jsonl` | 2,500 | Model specification generation |
| `v2_step3_templates.jsonl` | 2,500 | Constraint template generation |
| `v2_step4_codegen.jsonl` | 2,500 | GurobiPy code generation |
| `v2_step05_repair_json.jsonl` | 1,000 | JSON format repair |
| `v2_step06_repair_runtime.jsonl` | 625 | Runtime error repair |
| `v2_step07_repair_probe.jsonl` | 2,500 | Semantic probe failure repair |

---

## Data Format

Each sample follows the standard Chat format, compatible with OpenAI / HuggingFace / LLaMA-Factory:

```json
{
  "messages": [
    {"role": "system", "content": "System prompt with global rules..."},
    {"role": "user", "content": "Scenario description + data profile + task instruction"},
    {"role": "assistant", "content": "Model output (JSON or Python code)"}
  ],
  "metadata": {
    "scenario_id": "dairy_0042",
    "step": "step4_codegen",
    "industry": "dairy"
  }
}
```

---

## Agent Pipeline

This dataset covers the complete ReLoop agent pipeline:

```
Step 1: Contract     →  Extract optimization task contract from scenario (JSON)
Step 2: Spec Sheet   →  Generate mathematical model specification (JSON)
Step 3: Templates    →  Generate constraint template formulas (JSON)
Step 4: Code Gen     →  Generate executable GurobiPy code (Python)
Step 05: JSON Repair →  Fix malformed JSON outputs
Step 06: Code Repair →  Fix runtime errors in generated code
Step 07: Probe Repair → Fix logic errors detected by semantic probes
```

---

## Scenario Diversity

### 18 Industries

| Category | Industries |
|----------|------------|
| Food & Beverage | dairy, bakery, produce, meat, seafood, frozen, beverage, prepared_foods |
| Healthcare | pharmacy, baby_food |
| Retail | cosmetics, pet_food, organic, floral |
| Specialty | specialty_cheese, meal_kit, wine |

### 48 Feature Combinations

Covering various combinations of supply chain features:

- **Network Features**: substitution, transshipment
- **Time Features**: lead_time, shelf_life variations
- **Constraint Features**: moq (minimum order quantity), pack_size, budget, labor, waste_limit
- **Capacity Features**: storage capacity, production capacity

---

## Merge Training Data

```bash
# Merge all training data into a single file
cat v2_step*.jsonl > v2_train_all.jsonl
```

---

## Quick Start

### Using HuggingFace Transformers

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

# Load data (after merging)
dataset = load_dataset("json", data_files="v2_train_all.jsonl")

# Load model
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Formatting function
def format_chat(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False
    )

# Training
training_args = TrainingArguments(
    output_dir="./reloop-sft",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=2,
    bf16=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    formatting_func=format_chat,
    max_seq_length=8192,
)
trainer.train()
```

### Using LLaMA-Factory

1. Add to `dataset_info.json`:

```json
{
  "reloop_sft": {
    "file_name": "v2_train_all.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    }
  }
}
```

2. Training configuration:

```yaml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
stage: sft
do_train: true
dataset: reloop_sft
template: qwen
output_dir: ./output/reloop
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-5
num_train_epochs: 2
bf16: true
cutoff_len: 8192
```

---

## Training Recommendations

### Recommended Hyperparameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Learning Rate | 1e-5 ~ 2e-5 | Higher values may damage pretrained capabilities |
| Batch Size | 2-4 | With gradient accumulation |
| Epochs | 2-3 | More may cause overfitting |
| Context Length | 8192+ | Step 4 code can be lengthy |
| LoRA Rank | 64-128 | If using LoRA |

### Training Strategies

1. **Full Training**: Use `v2_train_all.jsonl` for all capabilities
2. **Code Generation Focus**: Use only `v2_step4_codegen.jsonl` + repair files
3. **Staged Training**: Train Steps 1-4 first, then repair samples

---

## Task Details

### Step 1: Task Contract Extraction

Extract structured optimization task definitions from natural language scenario descriptions:

- Optimization objective (minimize total cost)
- Decision variables (Q, I, y, S, X, L, W)
- Hard constraints (capacity, conservation, aging)
- Soft constraints (lost sales penalty)

### Step 2: Model Specification

Generate MILP model mathematical specifications:

- Set definitions (P, L, T, A)
- Decision variables and their indices
- Objective function terms
- Constraint family definitions
- Edge case handling

### Step 3: Constraint Templates

Transform specifications into parameterized mathematical formula templates:

- Flow conservation constraints
- Capacity constraints
- Aging dynamics constraints
- Substitution/transshipment constraints

### Step 4: Code Generation

Generate complete executable GurobiPy MILP solver code:

- Variable creation (I, y, Q, W, L, S, X)
- Constraint addition
- Objective function
- Solver invocation

### Steps 05-07: Error Repair

Train the model's self-repair capabilities:

| Repair Type | Error Example | Correct Approach |
|-------------|---------------|------------------|
| JSON Repair | `{'key': 1}` | `{"key": 1}` |
| Runtime Repair | `KeyError: 'sub_edges'` | Use nested `.get()` access |
| Probe Repair | Missing t=1 initialization | Add `I[p,l,1,a]=0` constraint |

---

## Semantic Probe Coverage

Training data ensures generated code passes these semantic probes:

| Probe Name | Tests | Failure Symptom |
|------------|-------|-----------------|
| `substitution_basic` | Substitution flow correctness | Wrong objective |
| `demand_route_constraint` | Demand routing constraint | UNBOUNDED |
| `no_substitution` | Empty sub_edges handling | Incorrect S variables |
| `production_capacity` | Production capacity constraint | Over-production |
| `storage_capacity` | Storage capacity constraint | INFEASIBLE |
| `aging_dynamics` | Aging dynamics constraint | Wrong waste |
| `lost_sales_slack` | L variable as slack | INFEASIBLE |
| `inventory_nonnegativity` | Non-negative inventory | Negative inventory |
| `initialization` | t=1 initialization | Objective ≈ 0 |
| `lead_time` | Lead time handling | Index error |
| `moq` | Minimum order quantity | MOQ not enforced |
| `transshipment` | Transshipment network | Wrong flow |
| `labor_capacity` | Labor constraint | Over labor capacity |
| `holding_cost` | Holding cost (I-y) | Objective 60%+ too high |

---

## Data Quality Assurance

- All Step 4 code samples validated against 14 semantic probes
- Spec Sheet structure compatible with Sanity Checker's 6 validation rules
- Scenario data fully synthetic, no benchmark data leakage

---

## Variable Naming Convention

Core variables (always created):
- `I[p,l,t,a]`: Start-of-period inventory by remaining life bucket
- `y[p,l,t,a]`: Sales from each life bucket
- `W[p,l,t]`: Waste (expired inventory from bucket a=1)
- `Q[p,l,t]`: Orders/production
- `L[p,l,t]`: Lost sales (slack variable)

Optional variables (only if feature is active):
- `S[p_from,p_to,l,t]`: Substitution flow (only if sub_edges nonempty)
- `X[p,src,dst,t]`: Transshipment flow (only if trans_edges nonempty)
- `z[p,l,t]`: Binary order indicator (only if moq > 0 or fixed_order > 0)
- `n[p,l,t]`: Integer pack count (only if pack_size > 1)

---

## Critical Implementation Rules

1. **Initialization at t=1**: All non-fresh inventory buckets must start empty (`I[p,l,1,a]=0` for `a < shelf_life[p]`)
2. **Aging at t=T**: Do NOT create aging constraints for the final period
3. **Fresh Inflow**: Check `t > lead_time` before accessing past orders
4. **Holding Cost**: Apply only to life buckets `a >= 2`, use `(I - y)` not just `I`
5. **Data Access**: Use nested access `data.get('network', {}).get('sub_edges', [])`, NOT `data['sub_edges']`

---

## Citation

If you use this dataset, please cite the ReLoop project.

## License

Part of the ReLoop research project. See main repository LICENSE.
