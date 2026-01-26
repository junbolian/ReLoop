"""Run test with environment variables set."""
import os
import subprocess
import sys

# Set environment variables
os.environ['OPENAI_API_KEY'] = 'sk-ERIYsfsWZCYyjbhTfJpPXCLOo8yoG7EuT93m0OIViO2WIdYj'
os.environ['OPENAI_BASE_URL'] = 'https://yinli.one/v1'
# For Claude models, use the same API key/URL
os.environ['ANTHROPIC_API_KEY'] = 'sk-ERIYsfsWZCYyjbhTfJpPXCLOo8yoG7EuT93m0OIViO2WIdYj'
os.environ['ANTHROPIC_BASE_URL'] = 'https://yinli.one/v1'

# Run the test
result = subprocess.run(
    [sys.executable, 'run_test_with_log.py',
     '--scenario', 'retail_f1_52_weeks_v0',
     '--model', 'claude-opus-4-5-20251101',
     '--compare'],
    cwd=r'e:\reloop'
)
sys.exit(result.returncode)
