reloop/
├── solvers/
│   └── universal_retail_solver.py       # [核心] 通用求解器
├── tools/
│   └── retail_benchmark_generator.py    # [生成] 通用零售数据生成器 
├── scenarios/
│   └── retail_comprehensive/            # [数据] 全面零售场景
│       ├── spec/
│       │   └── retail_spec.md
│       └── data/                        # 190个通用零售 JSON 文件
└── eval/
    └── run_benchmark.py                 # [评估] 验证脚本