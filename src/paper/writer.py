"""Publisher: copy to published/ and git init."""
import shutil, subprocess
from pathlib import Path
from ..utils.log import get_logger
LOGGER = get_logger(__name__)

def run_publish(project_dir):
    repo_root = project_dir.parent.parent
    pub = repo_root / "published" / project_dir.name
    pub.mkdir(parents=True, exist_ok=True)
    for item in project_dir.iterdir():
        dest = pub / item.name
        if item.is_dir():
            if dest.exists(): shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    if not (pub / ".git").exists():
        subprocess.run(["git","init"], cwd=str(pub), capture_output=True)
        subprocess.run(["git","add","."], cwd=str(pub), capture_output=True)
        subprocess.run(["git","commit","-m","Initial publish"], cwd=str(pub), capture_output=True)
    LOGGER.info("published to %s", pub)
