# ReLoop 训练数据集 (内部版本)

本目录包含用于 ReLoop 零售供应链优化 Agent 的 SFT 训练数据。

## 数据集概览

| 统计项 | 数值 |
|--------|------|
| 总样本数 | **14,125** |
| 场景数量 | 2,500 |
| 行业覆盖 | 18 个 |
| 特征组合 | 48 种 |

## 文件说明

### 训练数据文件

| 文件名 | 样本数 | 任务类型 |
|--------|--------|----------|
| `v2_step1_contract.jsonl` | 2,500 | 任务契约提取 |
| `v2_step2_spec_sheet.jsonl` | 2,500 | 模型规格表生成 |
| `v2_step3_templates.jsonl` | 2,500 | 约束模板生成 |
| `v2_step4_codegen.jsonl` | 2,500 | GurobiPy 代码生成 |
| `v2_step05_repair_json.jsonl` | 1,000 | JSON 格式修复 |
| `v2_step06_repair_runtime.jsonl` | 625 | 运行时错误修复 |
| `v2_step07_repair_probe.jsonl` | 2,500 | 语义探针失败修复 |

### 工具脚本

| 文件名 | 用途 |
|--------|------|
| `generate_synthetic_training_data_v2.py` | 训练数据生成器 |
| `validate_training_data.py` | 数据验证脚本 |

### 文档

| 文件名 | 用途 |
|--------|------|
| `README.md` | 本文件（内部使用） |
| `README_opensource.md` | 开源版本说明（英文） |

---

## 合并训练数据

```bash
# 合并所有训练数据为单个文件
cat v2_step*.jsonl > v2_train_all.jsonl
```

---

## 生成训练数据

```bash
# 生成完整训练数据
python generate_synthetic_training_data_v2.py

# 可修改脚本末尾参数:
# - num_scenarios: 场景数量（默认 2500）
# - include_repairs: 是否包含 repair 样本
# - repairs_per_scenario: 每场景 probe repair 样本数
```

---

## Step Prompts 对应关系

| Step | Prompt 文件 | 训练数据 |
|------|-------------|----------|
| 全局规则 | `00_global_guardrails.txt` | 所有样本 |
| Step 1 | `01_step1_contract.txt` | `v2_step1_contract.jsonl` |
| Step 2 | `02_step2_spec_sheet.txt` | `v2_step2_spec_sheet.jsonl` |
| Step 3 | `03_step3_constraint_templates.txt` | `v2_step3_templates.jsonl` |
| Step 4 | `04_step4_codegen.txt` | `v2_step4_codegen.jsonl` |
| Step 05 | `05_format_repair_json.txt` | `v2_step05_repair_json.jsonl` |
| Step 06 | `06_repair_code.txt` | `v2_step06_repair_runtime.jsonl` |
| Step 07 | `07_repair_audit_probe.txt` | `v2_step07_repair_probe.jsonl` |

---

## 与 Sanity Checker 的适配

生成的 Spec Sheet 结构兼容 `sanity_checker.py` 的 6 项检查：

1. `hard_equals_really_hard` - L 变量存在且在约束中
2. `unit_consistency` - I, y 使用 [p,l,t,a] 索引
3. `time_alignment` - lead_time 正确处理
4. `conservation_closed` - substitution 在 balance 约束中
5. `boundary_handled` - t=1 初始化, t=T 边界
6. `extreme_cases_behavior` - 零需求/容量处理

---

## 与 Semantic Probes 的适配

生成的代码通过 `semantic_probes.py` 的 14 个探针：

| 探针 | 测试内容 | 代码中的对应 |
|------|----------|--------------|
| `substitution_basic` | 替代流 | `S[(pf, pt, l, t)]` |
| `demand_route_constraint` | 需求路由 | `demand_route_` 约束 |
| `no_substitution` | 空替代边 | `if sub_edges:` 检查 |
| `production_capacity` | 生产能力 | `prod_cap_` 约束 |
| `storage_capacity` | 存储容量 | `storage_cap_` 约束 |
| `aging_dynamics` | 老化动态 | `aging_` 约束, `range(1, T)` |
| `lost_sales_slack` | 松弛变量 | `L[(p, l, t)]` 在 sales_conservation |
| `inventory_nonnegativity` | 非负库存 | `lb=0.0` |
| `initialization` | t=1初始化 | `init_` 约束 |
| `lead_time` | 提前期 | `arrival_expr()`, `t_ord >= 1` |
| `moq` | 最小订货量 | `moq_lb_`, `moq_ub_` |
| `transshipment` | 转运 | `X[(p, src, dst, t)]` |
| `labor_capacity` | 劳动力 | `labor_cap_` 约束 |
| `holding_cost` | 持有成本 | `I[...] - y[...]` |

---

## 训练建议

### 推荐配置

| 参数 | 推荐值 |
|------|--------|
| Learning Rate | 1e-5 ~ 2e-5 |
| Batch Size | 2-4 (配合 gradient accumulation) |
| Epochs | 2-3 |
| Context Length | 8192+ |
| LoRA Rank | 64-128 |

### HuggingFace 示例

```python
from datasets import load_dataset
from trl import SFTTrainer

dataset = load_dataset("json", data_files="train_set/v2_train_all.jsonl")
# ... 训练代码
```

### LLaMA-Factory 配置

```json
{
  "reloop_sft": {
    "file_name": "v2_train_all.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages"}
  }
}
```

---

## 开源发布

开源时：
1. 删除 `generate_synthetic_training_data_v2.py`
2. 删除 `validate_training_data.py`
3. 删除本文件 `README.md`
4. 将 `README_opensource.md` 重命名为 `README.md`
