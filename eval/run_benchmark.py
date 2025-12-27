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
import re
from collections import Counter


SUMMARY_ROW_RE = re.compile(
    r'^\|\s*(?P<name>[^|]+?)\s*\|\s*(?P<status>[^|]+?)\s*\|\s*(?P<obj>[^|]+?)\s*\|\s*$'
)


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


def _extract_summary(stdout_text: str, expected_name: str):
    """
    Extract the summary row printed by universal_retail_solver.py:
    | scenario_name | STATUS | OBJ |
    Returns (status, objective) or (None, None) if not found.
    """
    for line in stdout_text.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        m = SUMMARY_ROW_RE.match(s)
        if not m:
            continue

        name = m.group("name").strip()
        status = m.group("status").strip()
        obj = m.group("obj").strip()

        # Skip header row if any
        if name.upper() == "SCENARIO NAME":
            continue

        # Prefer exact match; fall back to contains-match to be tolerant to padding
        if name == expected_name or expected_name == name:
            return status, obj

    # If exact match not found, try a second pass with a soft contains-match
    for line in stdout_text.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        m = SUMMARY_ROW_RE.match(s)
        if not m:
            continue

        name = m.group("name").strip()
        status = m.group("status").strip()
        obj = m.group("obj").strip()

        if name.upper() == "SCENARIO NAME":
            continue

        if expected_name in name or name in expected_name:
            return status, obj

    return None, None


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
        elapsed = 0.0

        try:
            proc_result = subprocess.run(
                [sys.executable, solver_script, "--file", json_file],
                capture_output=True,
                text=True,
                timeout=120,  # hard guard; solver itself has TimeLimit
            )
            elapsed = time.time() - t0

            stdout = (proc_result.stdout or "").strip()
            stderr = (proc_result.stderr or "").strip()

            parsed_status, parsed_obj = _extract_summary(stdout, expected_name=scenario_name)

            if parsed_status is not None:
                status, objective = parsed_status, parsed_obj
            else:
                # If solver didn't print a summary row, treat as crash/misbehavior.
                if proc_result.returncode != 0:
                    err_msg = stderr.replace("\n", " ").strip()
                    if len(err_msg) > 120:
                        err_msg = err_msg[:117] + "..."
                    status = f"CRASH: {err_msg}" if err_msg else "CRASH"
                else:
                    # Return code 0 but no summary line -> still mark as CRASH for consistency
                    err_msg = stderr.replace("\n", " ").strip()
                    if len(err_msg) > 120:
                        err_msg = err_msg[:117] + "..."
                    status = f"NO_SUMMARY: {err_msg}" if err_msg else "NO_SUMMARY"
                objective = "N/A"

        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            status = "TIMEOUT"
            objective = "N/A"
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
