import json
from pathlib import Path
from typing import Dict

from ..utils.time import utc_now, timestamp_id


PROJECT_SUBDIRS = [
    "00_idea",
    "01_plan",
    "02_exp",
    "03_results",
    "04_paper",
]


def create_project(root: Path) -> Dict[str, str]:
    project_id = timestamp_id()
    project_dir = root / project_id
    project_dir.mkdir(parents=True, exist_ok=False)

    for subdir in PROJECT_SUBDIRS:
        (project_dir / subdir).mkdir(parents=True, exist_ok=True)

    meta = {
        "project_id": project_id,
        "state": "IDEA",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "retry_count": 0,
        "budget": {"tokens": 200000, "gpu_hours": 0, "wall_minutes": 90},
        "best": {"run_id": None, "score": None},
        "failure_reason": None,
    }

    meta_path = project_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    return {
        "project_id": project_id,
        "project_dir": str(project_dir),
        "meta_path": str(meta_path),
    }
