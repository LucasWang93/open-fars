"""Generate sbatch scripts for experiment submission."""

from pathlib import Path


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -p {partition}
#SBATCH --gres={gres}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH -o {log_dir}/%j.out
#SBATCH -e {log_dir}/%j.err

source /home/sw2572/Keys/env.sh
source $FARS_VENV/bin/activate
cd $FARS_ROOT

python3 src/compute/run_experiment_standalone.py {project_dir} {repo_root}
"""


def generate_sbatch_script(
    project_dir: Path,
    repo_root: Path,
    log_dir: Path,
    slurm_cfg: dict = None,
) -> Path:
    cfg = slurm_cfg or {}
    content = SBATCH_TEMPLATE.format(
        job_name="fars_" + project_dir.name,
        partition=cfg.get("partition", "gpu"),
        gres=cfg.get("gres", "gpu:rtx_5000_ada:4"),
        time=cfg.get("time", "02:00:00"),
        mem=cfg.get("mem", "120G"),
        cpus_per_task=cfg.get("cpus_per_task", "16"),
        log_dir=str(log_dir),
        project_dir=str(project_dir),
        repo_root=str(repo_root),
    )

    script_path = project_dir / "02_exp" / "run.sbatch"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(content)
    script_path.chmod(0o755)
    return script_path
