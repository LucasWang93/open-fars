"""Evaluator: aggregate metrics, generate summary.csv, fig1.png, analysis.md.

Supports both single-metric (eval_loss) and multi-metric (GUI Agent) modes.
"""

import csv
import json
import statistics
from pathlib import Path

from .plotting import plot_comparison
from ..utils.log import get_logger

LOGGER = get_logger(__name__)

MULTI_METRICS = ["element_accuracy", "action_f1", "step_success_rate"]


def _load_all_metrics(runs_dir):
    results = []
    for rd in sorted(runs_dir.iterdir()):
        mf = rd / "metrics.json"
        if mf.exists():
            results.append(json.loads(mf.read_text()))
    return results


def _extract_values(all_m, group, metric_name):
    values = []
    for m in all_m:
        if m.get("group") != group or m.get("status") != "SUCCESS":
            continue
        sm = m.get("secondary_metrics", {})
        if metric_name in sm:
            values.append(sm[metric_name])
        elif m.get("primary_metric", {}).get("name") == metric_name:
            values.append(m["primary_metric"]["value"])
    return values


def _stats(values):
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "n": len(values),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def run_evaluation(project_dir):
    runs_dir = project_dir / "02_exp" / "runs"
    results_dir = project_dir / "03_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_m = _load_all_metrics(runs_dir)
    if not all_m:
        LOGGER.warning("no metrics found in %s", runs_dir)
        return

    primary_name = all_m[0].get("primary_metric", {}).get("name", "step_success_rate")
    higher_better = all_m[0].get("primary_metric", {}).get("higher_is_better", True)

    has_multi = any(
        metric in all_m[0].get("secondary_metrics", {})
        for metric in MULTI_METRICS
    )
    metrics_to_report = MULTI_METRICS if has_multi else [primary_name]

    rows = []
    all_stats = {}
    for group in ("baseline", "treatment"):
        row = {"group": group}
        for mn in metrics_to_report:
            vals = _extract_values(all_m, group, mn)
            s = _stats(vals)
            row[f"{mn}_mean"] = s["mean"]
            row[f"{mn}_std"] = s["std"]
            row[f"{mn}_n"] = s["n"]
            all_stats[(group, mn)] = s
        rows.append(row)

    fieldnames = ["group"]
    for mn in metrics_to_report:
        fieldnames.extend([f"{mn}_mean", f"{mn}_std", f"{mn}_n"])
    with open(results_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    bl_primary = _extract_values(all_m, "baseline", primary_name)
    tr_primary = _extract_values(all_m, "treatment", primary_name)
    plot_comparison(bl_primary, tr_primary, primary_name, results_dir / "fig1.png")

    bl_s = all_stats.get(("baseline", primary_name), _stats([]))
    tr_s = all_stats.get(("treatment", primary_name), _stats([]))
    d = tr_s["mean"] - bl_s["mean"]
    imp = d > 0 if higher_better else d < 0

    lines = ["# Analysis", "", f"**Primary Metric:** {primary_name}", ""]

    if has_multi:
        lines.append("## Multi-Metric Summary")
        lines.append("| Metric | Baseline | Treatment | Diff |")
        lines.append("|--------|----------|-----------|------|")
        for mn in metrics_to_report:
            bs = all_stats.get(("baseline", mn), _stats([]))
            ts = all_stats.get(("treatment", mn), _stats([]))
            diff = ts["mean"] - bs["mean"]
            lines.append(
                f"| {mn} | {bs['mean']:.4f} +/- {bs['std']:.4f} "
                f"| {ts['mean']:.4f} +/- {ts['std']:.4f} "
                f"| {diff:+.4f} |"
            )
        lines.append("")

    lines.append("## Primary Metric Summary")
    lines.append(f"- Baseline: mean={bl_s['mean']:.4f}, std={bl_s['std']:.4f}")
    lines.append(f"- Treatment: mean={tr_s['mean']:.4f}, std={tr_s['std']:.4f}")
    lines.append(f"- Difference: {d:+.4f} ({'improvement' if imp else 'negative result'})")

    lines.extend(["", "## Interpretation"])
    if imp:
        lines.append(f"The treatment shows a positive effect ({d:+.4f}) on {primary_name}.")
    else:
        lines.append(f"The treatment did not improve {primary_name} (diff={d:+.4f}). Negative result.")
    lines.extend(["", "## Figures", "![Baseline vs Treatment](fig1.png)", ""])

    (results_dir / "analysis.md").write_text("\n".join(lines))
    LOGGER.info("evaluation done: bl=%.4f tr=%.4f diff=%+.4f", bl_s["mean"], tr_s["mean"], d)
