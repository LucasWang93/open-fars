"""Evaluator: aggregate metrics, generate summary.csv, fig1.png, analysis.md."""
import csv, json, statistics
from pathlib import Path
from .plotting import plot_comparison
from ..utils.log import get_logger
LOGGER = get_logger(__name__)

def _load_all_metrics(runs_dir):
    results = []
    for rd in sorted(runs_dir.iterdir()):
        mf = rd / "metrics.json"
        if mf.exists():
            results.append(json.loads(mf.read_text()))
    return results

def run_evaluation(project_dir):
    runs_dir = project_dir / "02_exp" / "runs"
    results_dir = project_dir / "03_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    all_m = _load_all_metrics(runs_dir)
    bl = [m["primary_metric"]["value"] for m in all_m if m.get("group")=="baseline" and m["status"]=="SUCCESS"]
    tr = [m["primary_metric"]["value"] for m in all_m if m.get("group")=="treatment" and m["status"]=="SUCCESS"]
    mn = all_m[0]["primary_metric"]["name"] if all_m else "score"
    rows = []
    for gn, sc in [("baseline",bl),("treatment",tr)]:
        if sc:
            rows.append({"group":gn,"mean":round(statistics.mean(sc),4),"std":round(statistics.stdev(sc),4) if len(sc)>1 else 0.0,"n":len(sc),"min":round(min(sc),4),"max":round(max(sc),4)})
    with open(results_dir/"summary.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["group","mean","std","n","min","max"]); w.writeheader(); w.writerows(rows)
    plot_comparison(bl, tr, mn, results_dir/"fig1.png")
    bm=rows[0]["mean"] if len(rows)>0 else 0; tm=rows[1]["mean"] if len(rows)>1 else 0
    bs=rows[0]["std"] if len(rows)>0 else 0; ts_=rows[1]["std"] if len(rows)>1 else 0
    d=tm-bm; imp=d>0
    lines=["# Analysis","","**Metric:** "+mn,"","## Summary"]
    lines.append("- Baseline: mean={:.4f}, std={:.4f}".format(bm,bs))
    lines.append("- Treatment: mean={:.4f}, std={:.4f}".format(tm,ts_))
    lines.append("- Difference: {:+.4f} ({})".format(d,"improvement" if imp else "negative result"))
    lines+=[""  ,"## Interpretation"]
    if imp:
        lines.append("The treatment shows a positive effect ({:+.4f}) on {}. Variance should be considered.".format(d,mn))
    else:
        lines.append("The treatment did not improve {} (diff={:+.4f}). Negative result.".format(mn,d))
    lines+=["","## Figures","![Baseline vs Treatment](fig1.png)"]
    (results_dir/"analysis.md").write_text(chr(10).join(lines)+chr(10))
    LOGGER.info("evaluation done: bl=%.4f tr=%.4f diff=%+.4f",bm,tm,d)
