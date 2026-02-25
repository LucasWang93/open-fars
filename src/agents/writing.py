"""Writing agent: render paper.md + reproducibility.md from templates."""

import csv
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ..utils.log import get_logger

LOGGER = get_logger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "paper" / "templates"


def run_writing(project_dir: Path) -> None:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))

    idea = json.loads((project_dir / "00_idea" / "idea.json").read_text())
    plan = json.loads((project_dir / "01_plan" / "plan.json").read_text())
    analysis = (project_dir / "03_results" / "analysis.md").read_text()

    summary_rows = []
    csv_path = project_dir / "03_results" / "summary.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            summary_rows = list(csv.DictReader(f))

    meta = json.loads((project_dir / "meta.json").read_text())
    project_id = meta["project_id"]

    improved = False
    if len(summary_rows) >= 2:
        bl_mean = float(summary_rows[0]["mean"])
        tr_mean = float(summary_rows[1]["mean"])
        improved = tr_mean > bl_mean

    if improved:
        conclusion = (
            "The results support the hypothesis. The treatment configuration "
            "outperformed the baseline on the primary metric."
        )
    else:
        conclusion = (
            "The results do not support the hypothesis. The treatment did not "
            "improve upon the baseline. This negative result is still informative."
        )

    summary_start = None
    analysis_summary = analysis
    analysis_lines = analysis.split(chr(10))
    for i, line in enumerate(analysis_lines):
        if line.startswith("## Summary"):
            summary_start = i + 1
        if summary_start and line.startswith("## ") and i > summary_start:
            analysis_summary = chr(10).join(analysis_lines[summary_start:i])
            break
    else:
        if summary_start:
            analysis_summary = chr(10).join(analysis_lines[summary_start:])

    ctx = {
        "title": "Experiment Report: " + idea["hypothesis"],
        "hypothesis": idea["hypothesis"],
        "actions": ", ".join(idea["actions"]),
        "metric": plan.get("metric", "score"),
        "control": plan.get("control", "baseline"),
        "treatment": plan.get("treatment", "modified config"),
        "n_seeds": len(plan.get("seeds", [])),
        "analysis_summary": analysis_summary.strip(),
        "summary_rows": summary_rows,
        "conclusion": conclusion,
        "project_id": project_id,
    }

    paper_dir = project_dir / "04_paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    paper_tmpl = env.get_template("paper.md.j2")
    (paper_dir / "paper.md").write_text(paper_tmpl.render(ctx))

    repro_tmpl = env.get_template("reproducibility.md.j2")
    (paper_dir / "reproducibility.md").write_text(repro_tmpl.render(ctx))

    LOGGER.info("writing done: paper.md + reproducibility.md")
