"""
Analyze chat_logs.jsonl to determine what each verification layer actually did
for each problem in the DeepSeek V3.1 RetailOpt-190 experiment.
"""

import json
import csv
from collections import Counter, defaultdict

CHAT_LOGS = "e:/reloop/experiment_results/RetailOpt-190/deepseek-v3.1/chat_logs.jsonl"
ABLATION_CSV = "e:/reloop/experiment_results/RetailOpt-190/deepseek-v3.1/ablation_report.csv"


def classify_log(log):
    """Classify a single log entry by its role."""
    if "error" in log:
        return "error"

    msgs = log.get("messages", [])
    sys_content = ""
    user_content = ""
    for m in msgs:
        if m["role"] == "system":
            sys_content = m["content"]
        elif m["role"] == "user":
            user_content = m["content"]

    # Order matters: more specific patterns first
    if user_content.startswith("Extract ALL numerical parameters"):
        return "data_extract"
    if "chain-of-thought" in sys_content or "step-by-step reasoning" in sys_content:
        return "cot_generate"
    if "fix broken optimization code" in sys_content or "failed to execute" in user_content:
        return "l1_regenerate"
    if "analyzing parameter behavior" in sys_content or "determine what behavior" in sys_content:
        return "l2_verify"
    if "Review these L2 direction" in user_content or "decide how to handle" in user_content:
        return "l2_review"
    if "extract the KEY CONSTRAINTS" in user_content or "CONSTRAINTS that should be present" in user_content:
        return "l3_cpt"
    if "violated safety rules" in user_content or "REJECTED because it violated safety rules" in user_content:
        return "repair_safety_retry"
    if "repair expert" in sys_content:
        return "repair"
    return "UNKNOWN"


def load_chat_logs():
    """Load and classify all chat log entries."""
    results = {}
    with open(CHAT_LOGS, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            idx = record["index"]
            pid = record["problem_id"]
            logs = record.get("logs", [])
            classifications = [classify_log(log) for log in logs]
            counts = Counter(classifications)
            results[idx] = {
                "problem_id": pid,
                "classifications": classifications,
                "counts": counts,
                "num_logs": len(logs),
            }
    return results


def load_ablation_csv():
    """Load ablation report CSV."""
    rows = {}
    with open(ABLATION_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["index"])
            rows[idx] = row
    return rows


def safe_float(val):
    """Convert a value to float, returning None if empty or invalid."""
    if val is None or val == "" or val == "None":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def main():
    print("=" * 80)
    print("DeepSeek V3.1 RetailOpt-190: Chat Log Analysis")
    print("=" * 80)

    chat_data = load_chat_logs()
    ablation = load_ablation_csv()

    n = len(chat_data)
    print(f"\nTotal problems analyzed: {n}")

    # ---------- Per-problem counts ----------
    problems_with = {
        "l1_regenerate": [],
        "l2_verify": [],
        "l2_review": [],
        "l3_cpt": [],
        "repair": [],
        "repair_safety_retry": [],
        "error": [],
    }

    total_calls = Counter()
    log_type_histogram = Counter()  # number of logs per problem

    for idx in sorted(chat_data.keys()):
        info = chat_data[idx]
        counts = info["counts"]
        log_type_histogram[info["num_logs"]] += 1
        for cat in problems_with:
            if counts.get(cat, 0) > 0:
                problems_with[cat].append(idx)
            total_calls[cat] += counts.get(cat, 0)

        # Also track unknowns
        if counts.get("UNKNOWN", 0) > 0:
            total_calls["UNKNOWN"] += counts["UNKNOWN"]

    # ---------- Summary: How many problems had each type ----------
    print("\n" + "-" * 60)
    print("SECTION 1: Problem-Level Activity Summary")
    print("-" * 60)
    print(f"  Problems with L1 regeneration:      {len(problems_with['l1_regenerate']):4d} / {n} ({100*len(problems_with['l1_regenerate'])/n:.1f}%)")
    print(f"  Problems with L2 verify calls:      {len(problems_with['l2_verify']):4d} / {n} ({100*len(problems_with['l2_verify'])/n:.1f}%)")
    print(f"  Problems with L2 review calls:      {len(problems_with['l2_review']):4d} / {n} ({100*len(problems_with['l2_review'])/n:.1f}%)")
    print(f"  Problems with L3 CPT:               {len(problems_with['l3_cpt']):4d} / {n} ({100*len(problems_with['l3_cpt'])/n:.1f}%)")
    print(f"  Problems with repair calls:         {len(problems_with['repair']):4d} / {n} ({100*len(problems_with['repair'])/n:.1f}%)")
    print(f"  Problems with repair safety retries: {len(problems_with['repair_safety_retry']):4d} / {n} ({100*len(problems_with['repair_safety_retry'])/n:.1f}%)")
    print(f"  Problems with errors:               {len(problems_with['error']):4d} / {n} ({100*len(problems_with['error'])/n:.1f}%)")

    # ---------- Total call counts ----------
    print("\n" + "-" * 60)
    print("SECTION 2: Total LLM Call Counts (across all problems)")
    print("-" * 60)
    all_types = ["data_extract", "cot_generate", "l1_regenerate", "l2_verify", "l2_review",
                 "l3_cpt", "repair", "repair_safety_retry", "error", "UNKNOWN"]
    for cat in all_types:
        c = total_calls.get(cat, 0)
        if c > 0:
            print(f"  {cat:25s}: {c:5d} calls")
    print(f"  {'TOTAL':25s}: {sum(total_calls.values()):5d} calls")

    # ---------- Distribution of L2 rounds ----------
    print("\n" + "-" * 60)
    print("SECTION 3: L2 Adversarial Round Distribution")
    print("-" * 60)
    l2_round_dist = Counter()
    for idx in sorted(chat_data.keys()):
        counts = chat_data[idx]["counts"]
        l2_rounds = counts.get("l2_verify", 0)  # each verify+review = 1 round
        l2_round_dist[l2_rounds] += 1
    for rounds in sorted(l2_round_dist.keys()):
        print(f"  {rounds} L2 verify calls: {l2_round_dist[rounds]:4d} problems")

    # ---------- Distribution of repair calls ----------
    print("\n" + "-" * 60)
    print("SECTION 4: Repair Call Distribution")
    print("-" * 60)
    repair_dist = Counter()
    for idx in sorted(chat_data.keys()):
        counts = chat_data[idx]["counts"]
        repairs = counts.get("repair", 0)
        repair_dist[repairs] += 1
    for repairs in sorted(repair_dist.keys()):
        print(f"  {repairs} repair calls: {repair_dist[repairs]:4d} problems")

    # ---------- Distribution of L1 regeneration calls ----------
    print("\n" + "-" * 60)
    print("SECTION 5: L1 Regeneration Call Distribution")
    print("-" * 60)
    l1_dist = Counter()
    for idx in sorted(chat_data.keys()):
        counts = chat_data[idx]["counts"]
        l1_regens = counts.get("l1_regenerate", 0)
        l1_dist[l1_regens] += 1
    for regens in sorted(l1_dist.keys()):
        print(f"  {regens} L1 regeneration calls: {l1_dist[regens]:4d} problems")

    # ---------- Distribution of safety retries ----------
    print("\n" + "-" * 60)
    print("SECTION 6: Repair Safety Retry Distribution")
    print("-" * 60)
    safety_dist = Counter()
    for idx in sorted(chat_data.keys()):
        counts = chat_data[idx]["counts"]
        safety = counts.get("repair_safety_retry", 0)
        safety_dist[safety] += 1
    for s in sorted(safety_dist.keys()):
        print(f"  {s} safety retries: {safety_dist[s]:4d} problems")

    # ---------- Cross-reference with ablation CSV ----------
    print("\n" + "=" * 80)
    print("SECTION 7: Cross-Reference with Ablation Results")
    print("=" * 80)

    # 7a: L1 regeneration and crash recovery
    print("\n--- 7a: L1 Regeneration & Crash Recovery ---")
    l1_crash_recovered = 0
    l1_crash_still_failed = 0
    l1_already_had_obj = 0
    for idx in problems_with["l1_regenerate"]:
        if idx not in ablation:
            continue
        row = ablation[idx]
        cot_obj = safe_float(row.get("cot_obj"))
        l1_obj = safe_float(row.get("l1_obj"))
        final_obj = safe_float(row.get("final_obj"))
        if cot_obj is None and final_obj is not None:
            l1_crash_recovered += 1
        elif cot_obj is None and final_obj is None:
            l1_crash_still_failed += 1
        else:
            l1_already_had_obj += 1

    print(f"  Problems with L1 regeneration: {len(problems_with['l1_regenerate'])}")
    print(f"    Crash recovered (cot=None -> final=value): {l1_crash_recovered}")
    print(f"    Still failed (cot=None -> final=None):     {l1_crash_still_failed}")
    print(f"    Already had cot_obj (regen after L1 pass): {l1_already_had_obj}")

    # What about all problems with cot_obj=None?
    cot_none_count = 0
    cot_none_recovered = 0
    for idx in sorted(ablation.keys()):
        row = ablation[idx]
        cot_obj = safe_float(row.get("cot_obj"))
        final_obj = safe_float(row.get("final_obj"))
        if cot_obj is None:
            cot_none_count += 1
            if final_obj is not None:
                cot_none_recovered += 1
    print(f"\n  All problems with cot_obj=None: {cot_none_count}")
    print(f"    Recovered to final_obj!=None:  {cot_none_recovered}")

    # 7b: L2 adversarial activity and objective changes
    print("\n--- 7b: L2 Activity & Objective Changes ---")
    l2_changed_obj = 0
    l2_unchanged_obj = 0
    l2_no_cot = 0
    l2_no_final = 0
    l2_both_none = 0
    for idx in problems_with["l2_verify"]:
        if idx not in ablation:
            continue
        row = ablation[idx]
        cot_obj = safe_float(row.get("cot_obj"))
        l2_obj = safe_float(row.get("l2_obj"))
        final_obj = safe_float(row.get("final_obj"))

        if cot_obj is None and final_obj is None:
            l2_both_none += 1
        elif cot_obj is None:
            l2_no_cot += 1
        elif final_obj is None:
            l2_no_final += 1
        elif abs(cot_obj - final_obj) > 0.01:
            l2_changed_obj += 1
        else:
            l2_unchanged_obj += 1

    print(f"  Problems with L2 activity: {len(problems_with['l2_verify'])}")
    print(f"    Final obj differs from cot obj:   {l2_changed_obj}")
    print(f"    Final obj same as cot obj:        {l2_unchanged_obj}")
    print(f"    cot=None (crash, L2 ran anyway):  {l2_no_cot}")
    print(f"    final=None (never recovered):     {l2_no_final}")
    print(f"    Both None:                        {l2_both_none}")

    # 7c: Did L2 or repair improve or worsen results?
    print("\n--- 7c: Impact on Pass Rate ---")
    # Compare cot_pass vs final_pass for problems with different activity
    for tolerance in ["1e4", "1e2", "5pct"]:
        cot_key = f"cot_pass_{tolerance}"
        final_key = f"final_pass_{tolerance}"
        cot_pass = sum(1 for idx, row in ablation.items() if row.get(cot_key) == "True")
        final_pass = sum(1 for idx, row in ablation.items() if row.get(final_key) == "True")
        print(f"  Tolerance {tolerance}: cot_pass={cot_pass}, final_pass={final_pass}, delta={final_pass - cot_pass:+d}")

    # 7d: Detailed look at problems where final improved from cot
    print("\n--- 7d: Problems Where Pipeline Improved Result (cot_pass=False -> final_pass=True, tol=5pct) ---")
    improved = []
    for idx in sorted(ablation.keys()):
        row = ablation[idx]
        cot_pass = row.get("cot_pass_5pct") == "True"
        final_pass = row.get("final_pass_5pct") == "True"
        if not cot_pass and final_pass:
            cot_obj = safe_float(row.get("cot_obj"))
            final_obj = safe_float(row.get("final_obj"))
            gt = safe_float(row.get("ground_truth"))
            cats = chat_data[idx]["counts"] if idx in chat_data else {}
            improved.append((idx, cot_obj, final_obj, gt, dict(cats)))
    print(f"  Total improved: {len(improved)}")
    for idx, cot, final, gt, cats in improved[:30]:
        l1r = cats.get("l1_regenerate", 0)
        l2v = cats.get("l2_verify", 0)
        l3c = cats.get("l3_cpt", 0)
        rep = cats.get("repair", 0)
        saf = cats.get("repair_safety_retry", 0)
        print(f"    Problem {idx:3d}: cot={cot!s:>15s} -> final={final!s:>15s} (GT={gt!s:>12s})  L1r={l1r} L2v={l2v} L3c={l3c} rep={rep} safe={saf}")

    # 7e: Problems where pipeline worsened result
    print("\n--- 7e: Problems Where Pipeline Worsened Result (cot_pass=True -> final_pass=False, tol=5pct) ---")
    worsened = []
    for idx in sorted(ablation.keys()):
        row = ablation[idx]
        cot_pass = row.get("cot_pass_5pct") == "True"
        final_pass = row.get("final_pass_5pct") == "True"
        if cot_pass and not final_pass:
            cot_obj = safe_float(row.get("cot_obj"))
            final_obj = safe_float(row.get("final_obj"))
            gt = safe_float(row.get("ground_truth"))
            cats = chat_data[idx]["counts"] if idx in chat_data else {}
            worsened.append((idx, cot_obj, final_obj, gt, dict(cats)))
    print(f"  Total worsened: {len(worsened)}")
    for idx, cot, final, gt, cats in worsened:
        l1r = cats.get("l1_regenerate", 0)
        l2v = cats.get("l2_verify", 0)
        l3c = cats.get("l3_cpt", 0)
        rep = cats.get("repair", 0)
        saf = cats.get("repair_safety_retry", 0)
        print(f"    Problem {idx:3d}: cot={cot!s:>15s} -> final={final!s:>15s} (GT={gt!s:>12s})  L1r={l1r} L2v={l2v} L3c={l3c} rep={rep} safe={saf}")

    # ---------- Call sequence patterns ----------
    print("\n" + "=" * 80)
    print("SECTION 8: Common Call Sequence Patterns")
    print("=" * 80)
    seq_counter = Counter()
    for idx in sorted(chat_data.keys()):
        info = chat_data[idx]
        # Compact representation: skip data_extract and cot_generate (always present)
        seq = tuple(c for c in info["classifications"] if c not in ("data_extract", "cot_generate"))
        seq_counter[seq] += 1

    print(f"\n  Unique patterns: {len(seq_counter)}")
    print(f"\n  Top 25 most common call sequences (after data_extract + cot_generate):")
    for seq, count in seq_counter.most_common(25):
        seq_str = " -> ".join(seq) if seq else "(none)"
        print(f"    {count:3d}x: {seq_str}")

    # ---------- Log count histogram ----------
    print("\n" + "-" * 60)
    print("SECTION 9: Total Log Entries Per Problem")
    print("-" * 60)
    for num_logs in sorted(log_type_histogram.keys()):
        print(f"  {num_logs:2d} logs: {log_type_histogram[num_logs]:4d} problems")


if __name__ == "__main__":
    main()
