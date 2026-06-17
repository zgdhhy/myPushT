from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from mypusht.evaluation.config import EVAL_RES_BASE


def mean(values):
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return sum(clean) / len(clean) if clean else float("nan")


def collect_all_csvs(base_dir: Path):
    csv_files = []
    for res_dir in sorted(base_dir.glob("res_*")):
        results_dir = res_dir / "results"
        if results_dir.is_dir():
            for csv_path in sorted(results_dir.glob("*.csv")):
                csv_files.append(csv_path)
    return csv_files


def read_rows(csv_files: list[Path]):
    rows = []
    for csv_path in tqdm(csv_files, desc="Reading CSVs", unit="file"):
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


# -----------------------------------------------
# eval_policy.py CSV fields:
#   seed, success, truncated, steps, reward,
#   final_xy_error, final_yaw_error, action_smoothness,
#   elapsed_sec, inference_ms_mean, policy, split, video_path
# -----------------------------------------------

def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["policy"], row["split"])
        groups[key].append(row)

    summary_rows = []
    for (policy, split), items in sorted(groups.items()):
        successes = [int(r.get("success", 0)) for r in items]
        steps_list = [int(r.get("steps", 0)) for r in items]
        rewards = [float(r.get("reward", float("nan"))) for r in items]
        xy_errors = [float(r.get("final_xy_error", float("nan"))) for r in items]
        yaw_errors = [float(r.get("final_yaw_error", float("nan"))) for r in items]
        smoothness = [float(r.get("action_smoothness", float("nan"))) for r in items]
        inference_ms = [float(r.get("inference_ms_mean", float("nan"))) for r in items]

        summary_rows.append({
            "policy": policy,
            "split": split,
            "episodes": len(items),
            "success_rate": round(mean(successes), 4),
            "mean_steps": round(mean(steps_list), 1),
            "mean_reward": round(mean(rewards), 4),
            "mean_final_xy_error": round(mean(xy_errors), 4),
            "mean_final_yaw_error": round(mean(yaw_errors), 4),
            "mean_action_smoothness": round(mean(smoothness), 4),
            "mean_inference_ms": round(mean(inference_ms), 3),
        })
    return summary_rows


def write_csv(rows, out_path: Path):
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    if not rows:
        print("No results to display.")
        return

    headers = [
        ("policy", 12), ("split", 10), ("episodes", 8),
        ("success", 10), ("steps", 8), ("reward", 10),
        ("xy_error", 10), ("yaw_error", 10), ("smoothness", 10), ("infer_ms", 10),
    ]

    col_key = {
        "success":    "success_rate",
        "steps":      "mean_steps",
        "reward":     "mean_reward",
        "xy_error":   "mean_final_xy_error",
        "yaw_error":  "mean_final_yaw_error",
        "smoothness": "mean_action_smoothness",
        "infer_ms":   "mean_inference_ms",
    }

    def fmt(val):
        if isinstance(val, float) and math.isfinite(val):
            return f"{val:.4f}"
        return str(val)

    header_line = "  ".join(f"{h:<{w}}" for h, w in headers)
    print(header_line)
    print("-" * len(header_line))

    for row in rows:
        parts = []
        for h, w in headers:
            key = col_key.get(h, h)
            v = row.get(key, "")
            parts.append(f"{fmt(v):<{w}}")
        print("  ".join(parts))


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-episode eval CSV(s) into a comparison table."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Single results directory. If not set, scans all res_*/results/ "
             "under outputs/eval/.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path. Defaults to outputs/eval/tables/policy_comparison.csv.",
    )
    args = parser.parse_args()

    if args.results_dir is not None:
        results_dir = Path(args.results_dir)
        csv_files = sorted(results_dir.glob("*.csv"))
        if not csv_files:
            raise SystemExit(f"No result csv files found in {results_dir}")
    else:
        csv_files = collect_all_csvs(EVAL_RES_BASE)
        if not csv_files:
            raise SystemExit(f"No result csv files found under {EVAL_RES_BASE}")

    print(f"Found {len(csv_files)} CSV(s):")
    for f in csv_files:
        print(f"  {f.relative_to(EVAL_RES_BASE.parent)}")

    rows = read_rows(csv_files)
    print(f"Loaded {len(rows)} episodes.")

    summary_rows = aggregate(rows)

    if args.out is not None:
        out_path = Path(args.out)
    else:
        out_path = EVAL_RES_BASE / "tables" / "policy_comparison.csv"
    write_csv(summary_rows, out_path)
    print(f"\nSaved comparison table to: {out_path}")

    print_table(summary_rows)


if __name__ == "__main__":
    main()
