@echo off
set OPENAI_API_KEY=sk-ERIYsfsWZCYyjbhTfJpPXCLOo8yoG7EuT93m0OIViO2WIdYj
set OPENAI_BASE_URL=https://yinli.one/v1
python run_test_with_log.py --scenario retail_f1_base_v1 --model gpt-5.1-2025-11-13 --mode both
