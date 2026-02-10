#!/usr/bin/env python
"""
Analyze ablation CSV to assess each verification layer's contribution.

Reads an ablation_report.csv and computes:
- Pass counts at each stage (cot, l1, l2, final)
- Layer-by-layer transitions (fail→pass, pass→fail)
- Crash recovery stats
- Net contribution per layer

Usage:
    python analyze_layers.py <path_to_ablation_report.csv>
    python analyze_layers.py experiment_results/RetailOpt-190/deepseek-v3.1/ablation_report.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


STAGES = ["cot", "l1", "l2", "final"]
TOLERANCES = [
    ("1e4", "0.01%"),
    ("1e2", "1%"),
    ("5pct", "5%"),
]


def load_csv(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_bool(val: str) -> bool:
    return val.strip().lower() == "true"


def parse_float(val: str) -> Optional[float]:
    val = val.strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def analyze(rows: List[Dict], tol_key: str, tol_label: str) -> Dict:
    """Analyze layer contributions at a specific tolerance level."""
    n = len(rows)

    # Pass counts per stage
    pass_counts = {}
    for stage in STAGES:
        col = f"{stage}_pass_{tol_key}"
        pass_counts[stage] = sum(1 for r in rows if parse_bool(r.get(col, "")))

    # Objective availability (execution success)
    exec_counts = {}
    for stage in STAGES:
        col = f"{stage}_obj"
        exec_counts[stage] = sum(1 for r in rows if parse_float(r.get(col, "")) is not None)

    # Layer transitions
    transitions = {}
    stage_pairs = list(zip(STAGES[:-1], STAGES[1:]))
    for prev, curr in stage_pairs:
        prev_col = f"{prev}_pass_{tol_key}"
        curr_col = f"{curr}_pass_{tol_key}"
        helped = []  # fail→pass
        hurt = []    # pass→fail
        unchanged_pass = 0
        unchanged_fail = 0
        for r in rows:
            pid = r.get("problem_id", r.get("index", "?"))
            p = parse_bool(r.get(prev_col, ""))
            c = parse_bool(r.get(curr_col, ""))
            if not p and c:
                helped.append(pid)
            elif p and not c:
                hurt.append(pid)
            elif p and c:
                unchanged_pass += 1
            else:
                unchanged_fail += 1
        transitions[f"{prev}→{curr}"] = {
            "helped": helped,
            "hurt": hurt,
            "unchanged_pass": unchanged_pass,
            "unchanged_fail": unchanged_fail,
        }

    # Crash recovery: cot_obj empty → stage_obj non-empty
    cot_crashes = [r for r in rows if parse_float(r.get("cot_obj", "")) is None]
    crash_recovery = {}
    for stage in STAGES[1:]:
        obj_col = f"{stage}_obj"
        pass_col = f"{stage}_pass_{tol_key}"
        recovered = [r for r in cot_crashes if parse_float(r.get(obj_col, "")) is not None]
        recovered_pass = [r for r in recovered if parse_bool(r.get(pass_col, ""))]
        crash_recovery[stage] = {
            "total_crashes": len(cot_crashes),
            "recovered": len(recovered),
            "recovered_and_pass": len(recovered_pass),
        }

    return {
        "n": n,
        "tol_label": tol_label,
        "pass_counts": pass_counts,
        "exec_counts": exec_counts,
        "transitions": transitions,
        "crash_recovery": crash_recovery,
    }


def print_report(results: List[Dict]):
    n = results[0]["n"]
    print(f"\n{'='*72}")
    print(f"  LAYER CONTRIBUTION ANALYSIS  ({n} problems)")
    print(f"{'='*72}")

    # --- Pass counts table ---
    print(f"\n{'─'*72}")
    print("  Pass Counts by Stage and Tolerance")
    print(f"{'─'*72}")
    header = f"  {'Stage':<12}"
    for r in results:
        header += f"  {r['tol_label']:>10}"
    header += f"  {'Exec%':>10}"
    print(header)
    print(f"  {'─'*10}  " + "  ".join(["─"*10]*len(results)) + "  " + "─"*10)

    for stage in STAGES:
        line = f"  {stage:<12}"
        for r in results:
            cnt = r["pass_counts"][stage]
            pct = 100 * cnt / r["n"]
            line += f"  {cnt:>4}/{r['n']:<4} ({pct:4.1f}%)"
        exec_cnt = results[0]["exec_counts"][stage]
        exec_pct = 100 * exec_cnt / n
        line += f"  {exec_cnt:>4}/{n:<4} ({exec_pct:4.1f}%)"
        print(line.rstrip())

    # --- Layer transitions ---
    for r in results:
        tol = r["tol_label"]
        print(f"\n{'─'*72}")
        print(f"  Layer Transitions (tolerance={tol})")
        print(f"{'─'*72}")
        for key, t in r["transitions"].items():
            helped = t["helped"]
            hurt = t["hurt"]
            net = len(helped) - len(hurt)
            sign = "+" if net >= 0 else ""
            print(f"\n  {key}:  helped={len(helped)}, hurt={len(hurt)}, net={sign}{net}")
            if helped:
                print(f"    Helped (fail→pass): {', '.join(str(x) for x in helped[:20])}"
                      + (f" ... (+{len(helped)-20} more)" if len(helped) > 20 else ""))
            if hurt:
                print(f"    Hurt (pass→fail):   {', '.join(str(x) for x in hurt[:20])}"
                      + (f" ... (+{len(hurt)-20} more)" if len(hurt) > 20 else ""))

    # --- Crash recovery ---
    print(f"\n{'─'*72}")
    print("  Crash Recovery (CoT crashed → later stage recovered)")
    print(f"{'─'*72}")
    # Use 5% tolerance for crash recovery
    r_5pct = [r for r in results if r["tol_label"] == "5%"][0] if any(r["tol_label"] == "5%" for r in results) else results[-1]
    cr = r_5pct["crash_recovery"]
    total_crashes = cr[STAGES[1]]["total_crashes"]
    print(f"  Total CoT crashes: {total_crashes}/{n}")
    for stage in STAGES[1:]:
        info = cr[stage]
        print(f"  {stage:>8}: recovered={info['recovered']}, "
              f"recovered+pass(5%)={info['recovered_and_pass']}")

    # --- Summary verdict ---
    print(f"\n{'─'*72}")
    print("  Layer Contribution Summary (5% tolerance)")
    print(f"{'─'*72}")
    r_5pct_transitions = r_5pct["transitions"]
    for key, t in r_5pct_transitions.items():
        helped = len(t["helped"])
        hurt = len(t["hurt"])
        net = helped - hurt
        if net > 0:
            verdict = f"POSITIVE (+{net})"
        elif net == 0:
            verdict = "NEUTRAL (0)"
        else:
            verdict = f"NEGATIVE ({net})"
        layer_name = key.split("→")[1]
        print(f"  {layer_name:<8}: {verdict}  (helped={helped}, hurt={hurt})")

    print(f"\n{'='*72}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation CSV layer contributions.")
    parser.add_argument("csv_path", help="Path to ablation_report.csv")
    args = parser.parse_args()

    if not Path(args.csv_path).exists():
        print(f"File not found: {args.csv_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_csv(args.csv_path)
    if not rows:
        print("No data rows found in CSV.", file=sys.stderr)
        sys.exit(1)

    results = []
    for tol_key, tol_label in TOLERANCES:
        results.append(analyze(rows, tol_key, tol_label))

    print_report(results)


if __name__ == "__main__":
    main()
