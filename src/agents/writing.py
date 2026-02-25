"""Writing agent: GPT-4o writes paper, then render via Jinja2 templates."""

import csv
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ..llm.router import get_router
from ..llm.prompts import WRITING_SYSTEM, WRITING_USER
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

    summary_csv_text = csv_path.read_text() if csv_path.exists() else "N/A"

    router = get_router("azure_gpt4o")
    user_prompt = WRITING_USER.format(
        project_id=project_id,
        hypothesis=idea["hypothesis"],
        analysis=analysis,
        summary_csv=summary_csv_text,
    )
    paper_text = router.generate(WRITING_SYSTEM, user_prompt, json_mode=False)

    if "## Results" not in paper_text:
        paper_text = "## Results\n\n" + paper_text
    if "## Method" not in paper_text:
        paper_text = "## Method\n\nSee plan.md for details.\n\n" + paper_text
    if "## Limitations" not in paper_text:
        paper_text += "\n\n## Limitations\n\n- Toy experiment with limited seeds.\n"

    paper_dir = project_dir / "04_paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    (paper_dir / "paper.md").write_text(paper_text)

    repro_tmpl = env.get_template("reproducibility.md.j2")
    repro_ctx = {"project_id": project_id}
    (paper_dir / "reproducibility.md").write_text(repro_tmpl.render(repro_ctx))

    LOGGER.info("writing done (GPT-4o): paper.md + reproducibility.md")
