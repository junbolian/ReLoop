# ==============================================================================
# FILE: run_benchmark.py
# LOCATION: reloop/eval/
#
# DESCRIPTION:
#   Executes the Universal Retail Solver on all JSON instances in the
#   "retail_comprehensive/data" directory (e.g., the Retail-160 benchmark and
#   any extensions). For each scenario it records:
#     - scenario name
#     - solver status (e.g., OPTIMAL, OPTIMAL (TL), INFEASIBLE)
#     - objective string as printed by the solver
#     - objective_numeric (parsed float if available)
#     - time_sec (wall-clock time for the solve)
#
#   Results are saved to "benchmark_results.csv" and serve as ground truth
#   for evaluating LLM agents.
# ==============================================================================

import os
import glob
import subprocess
import sys
import csv
import time
from collections import Counter


def parse_objective_to_float(obj_str: str):
    """Convert an objective string like '76,027.00' to a float. Return empty
    string if parsing fails (keeps CSV simple and backward compatible)."""
    obj_str = obj_str.strip()
    if not obj_str or obj_str.upper() == "N/A":
        return ""
    try:
        clean = obj_str.replace(",", "")
        return float(clean)
    except Exception:
        return ""


def main():
    # 1. Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "scenarios", "retail_comprehensive", "data")
    solver_script = os.path.join(base_dir, "solvers", "universal_retail_solver.py")

    output_csv = os.path.join(base_dir, "eval", "benchmark_results.csv")

    # 2. Sanity checks
    if not os.path.exists(data_dir):
        print(f"ERROR: Data dir not found: {data_dir}")
        print("Please run 'tools/retail_benchmark_generator.py' first.")
        return

    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    num_instances = len(json_files)

    print(f"Found {num_instances} instances.")
    print(f"Results will be saved to: {os.path.abspath(output_csv)}")

    # 3. Prepare CSV in-memory buffer
    results = []

    print("-" * 90)
    print(f"| {'SCENARIO NAME':<40} | {'STATUS':<15} | {'OBJECTIVE':<20} |")
    print("-" * 90)

    start_time = time.time()

    # 4. Batch execution
    for i, json_file in enumerate(json_files, start=1):
        scenario_name = os.path.basename(json_file).replace(".json", "")

        # Show "RUNNING..." status inline (use \r to overwrite)
        print(
            f"| {scenario_name:<40} | {'RUNNING...':<15} | {'...':<20} |",
            end="\r",
            flush=True,
        )

        t0 = time.time()
        status = "ERROR"
        objective = "N/A"

        try:
            proc_result = subprocess.run(
                [sys.executable, solver_script, "--file", json_file],
                capture_output=True,
                text=True,
            )
            elapsed = time.time() - t0
            output = proc_result.stdout.strip()

            # The solver prints a line like:
            # | retail_f1_base_v0                 | OPTIMAL       | 12345.00       |
            found_summary = False
            for line in output.split("\n"):
                line_stripped = line.strip()
                if line_stripped.startswith("|") and scenario_name in line_stripped:
                    parts = [p.strip() for p in line_stripped.split("|")]
                    if len(parts) >= 4:
                        status = parts[2]
                        objective = parts[3]
                        found_summary = True
                        break

            if not found_summary:
                err_msg = proc_result.stderr.strip().replace("\n", " ")
                if len(err_msg) > 80:
                    err_msg = err_msg[:77] + "..."
                status = f"CRASH: {err_msg}" if err_msg else "CRASH"
        except Exception as e:
            elapsed = time.time() - t0
            status = "SYSTEM_ERROR"
            objective = "N/A"

        # Final print for this scenario (overwrite RUNNING line)
        print(
            f"| {scenario_name:<40} | {status:<15} | {objective:<20} |",
            flush=True,
        )

        results.append(
            {
                "scenario": scenario_name,
                "status": status,
                "objective": objective,
                "objective_numeric": parse_objective_to_float(objective),
                "time_sec": f"{elapsed:.4f}",
            }
        )

    # 5. Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    fieldnames = ["scenario", "status", "objective", "objective_numeric", "time_sec"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    duration = time.time() - start_time
    print("-" * 90)
    print(f"Done. Processed {num_instances} instances in {duration:.2f} seconds.")
    print(f"Ground truth saved to: {os.path.abspath(output_csv)}")

    # 6. Simple summary by status
    status_counts = Counter(r["status"] for r in results)
    print("Status summary:")
    for s, c in sorted(status_counts.items()):
        print(f"  {s:15s}: {c:4d}")


if __name__ == "__main__":
    main()
