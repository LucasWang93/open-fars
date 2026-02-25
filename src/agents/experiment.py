"""Experiment agent: run baseline + treatment across seeds, produce metrics."""

import hashlib
import json
import math
import random
from pathlib import Path

import yaml

from ..utils.log import get_logger

LOGGER = get_logger(__name__)


def _toy_score(config: dict, seed: int) -> float:
    """Deterministic toy scoring function that responds to config changes."""
    random.seed(seed)
    base = 0.5
    lr = config.get("train", {}).get("lr", 0.0001)
    temp = config.get("gen", {}).get("temperature", 1.0)

    lr_effect = -abs(math.log10(lr) - math.log10(0.00015)) * 0.1
    temp_effect = -abs(temp - 0.9) * 0.05
    noise = random.gauss(0, 0.02)

    return round(max(0, min(1, base + lr_effect + temp_effect + noise)), 4)


def run_experiment(project_dir: Path, repo_root: Path) -> None:
    plan_dir = project_dir / "01_plan"
    config_path = plan_dir / "config.yaml"
    configs = yaml.safe_load(config_path.read_text())

    seeds = configs["seeds"]
    runs_dir = project_dir / "02_exp" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_idx = 0
    for group_name in ("baseline", "treatment"):
        cfg = configs[group_name]
        cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]

        for seed in seeds:
            run_idx += 1
            run_id = f"run_{run_idx:04d}"
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            score = _toy_score(cfg, seed)

            metrics = {
                "run_id": run_id,
                "group": group_name,
                "primary_metric": {
                    "name": "score",
                    "value": score,
                    "higher_is_better": True,
                },
                "secondary_metrics": {"loss": round(1.0 - score, 4)},
                "seed": seed,
                "config_hash": f"sha256:{cfg_hash}",
                "config": cfg,
                "status": "SUCCESS",
            }

            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            LOGGER.info("%s seed=%d score=%.4f (%s)", run_id, seed, score, group_name)

    LOGGER.info("experiment done: %d runs in %s", run_idx, runs_dir)
