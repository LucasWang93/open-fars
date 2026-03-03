"""Adaptive failure diagnosis and recovery for the FARS pipeline."""

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from ..utils.log import get_logger

LOGGER = get_logger(__name__)


@dataclass
class RecoveryAction:
    strategy: str
    retry: bool
    description: str = ""
    config_patch: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

_INFRA_PATTERNS = [
    r"ModuleNotFoundError",
    r"ImportError",
    r"FileNotFoundError",
    r"PermissionError",
    r"NODE_FAIL",
    r"PREEMPTED",
    r"ConnectionError",
    r"OSError.*No space left",
]

_RESOURCE_PATTERNS = [
    r"CUDA out of memory",
    r"OutOfMemoryError",
    r"OUT_OF_MEMORY",
    r"torch\.cuda\.OutOfMemoryError",
    r"RuntimeError.*CUDA",
    r"Killed",
    r"exit code 137",
]

_LOGIC_PATTERNS = [
    r"NaN",
    r"nan",
    r"loss is nan",
    r"JSONDecodeError",
    r"ValidationError",
    r"KeyError",
    r"ValueError",
    r"IndexError",
    r"TIMEOUT",
]


def classify_error(error_msg: str) -> str:
    """Classify error into: infrastructure, resource, logic, or unknown."""
    for pattern in _RESOURCE_PATTERNS:
        if re.search(pattern, error_msg, re.IGNORECASE):
            return "resource"
    for pattern in _INFRA_PATTERNS:
        if re.search(pattern, error_msg, re.IGNORECASE):
            return "infrastructure"
    for pattern in _LOGIC_PATTERNS:
        if re.search(pattern, error_msg, re.IGNORECASE):
            return "logic"
    return "unknown"


# ---------------------------------------------------------------------------
# Recovery strategies
# ---------------------------------------------------------------------------

def _fix_infrastructure(error_msg: str, project_dir: Path, stage: str) -> RecoveryAction:
    """Attempt to fix infrastructure issues."""
    if "ModuleNotFoundError" in error_msg or "ImportError" in error_msg:
        module_match = re.search(r"No module named ['\"](\S+)['\"]", error_msg)
        if module_match:
            module = module_match.group(1).split(".")[0]
            try:
                subprocess.run(
                    ["pip", "install", module],
                    capture_output=True, timeout=120,
                )
                return RecoveryAction(
                    strategy="auto_install_module",
                    retry=True,
                    description=f"Installed missing module: {module}",
                )
            except Exception:
                pass

    if "PREEMPTED" in error_msg:
        return RecoveryAction(
            strategy="resubmit_preempted",
            retry=True,
            description="Job was preempted on scavenge_gpu, resubmitting",
        )

    if "NODE_FAIL" in error_msg:
        return RecoveryAction(
            strategy="exclude_failed_node",
            retry=True,
            description="Excluding failed node and resubmitting",
        )

    return RecoveryAction(
        strategy="infra_retry",
        retry=True,
        description=f"Infrastructure error, retrying: {error_msg[:200]}",
    )


def _fix_resource(error_msg: str, project_dir: Path) -> RecoveryAction:
    """Reduce resource usage to handle OOM/resource errors."""
    config_path = project_dir / "01_plan" / "config.yaml"
    if not config_path.exists():
        return RecoveryAction(strategy="resource_no_config", retry=False,
                              description="Cannot reduce resources: config.yaml not found")

    config = yaml.safe_load(config_path.read_text())
    patch = {}

    for group in ("baseline", "treatment"):
        cfg = config.get(group, {})
        train = cfg.get("train", {})

        current_batch = train.get("per_device_train_batch_size", 4)
        current_seq = train.get("max_seq_length", 2048)
        current_grad = train.get("gradient_accumulation_steps", 4)

        if current_batch > 1:
            new_batch = max(1, current_batch // 2)
            new_grad = current_grad * 2
            train["per_device_train_batch_size"] = new_batch
            train["gradient_accumulation_steps"] = new_grad
            patch[f"{group}.batch_size"] = f"{current_batch} -> {new_batch}"
        elif current_seq > 512:
            new_seq = max(512, current_seq // 2)
            train["max_seq_length"] = new_seq
            patch[f"{group}.max_seq_length"] = f"{current_seq} -> {new_seq}"

        cfg["train"] = train
        data = cfg.get("data", {})
        current_train_n = data.get("max_train_samples", 2000)
        if current_train_n > 500:
            data["max_train_samples"] = max(500, current_train_n // 2)
            patch[f"{group}.max_train_samples"] = f"{current_train_n} -> {data['max_train_samples']}"
        cfg["data"] = data
        config[group] = cfg

    config_path.write_text(yaml.dump(config, default_flow_style=False))

    return RecoveryAction(
        strategy="reduce_resources",
        retry=True,
        description=f"Reduced resource usage: {patch}",
        config_patch=patch,
    )


def _fix_logic(error_msg: str, project_dir: Path, stage: str) -> RecoveryAction:
    """Handle logic errors with config/prompt adjustments."""
    if stage == "IDEA":
        return RecoveryAction(
            strategy="retry_ideation",
            retry=True,
            description="Ideation output invalid, retrying with same prompt",
        )

    if stage == "PLAN":
        return RecoveryAction(
            strategy="retry_planning",
            retry=True,
            description="Planning output invalid, retrying",
        )

    if stage == "RUN":
        if "nan" in error_msg.lower():
            config_path = project_dir / "01_plan" / "config.yaml"
            if config_path.exists():
                config = yaml.safe_load(config_path.read_text())
                for group in ("baseline", "treatment"):
                    train = config.get(group, {}).get("train", {})
                    current_lr = train.get("learning_rate", 2e-5)
                    train["learning_rate"] = current_lr * 0.5
                    train["warmup_ratio"] = min(0.2, train.get("warmup_ratio", 0.1) + 0.05)
                    config[group]["train"] = train
                config_path.write_text(yaml.dump(config, default_flow_style=False))
            return RecoveryAction(
                strategy="fix_nan_loss",
                retry=True,
                description="NaN loss detected: halved learning rate, increased warmup",
            )

        if "TIMEOUT" in error_msg:
            return RecoveryAction(
                strategy="extend_timeout",
                retry=True,
                description="Job timed out, will extend time limit",
            )

    if stage in ("ANALYZE", "WRITE"):
        return RecoveryAction(
            strategy="skip_with_retry",
            retry=True,
            description=f"Non-critical stage {stage} failed, retrying",
        )

    return RecoveryAction(
        strategy="logic_retry",
        retry=True,
        description=f"Logic error in {stage}, retrying",
    )


# ---------------------------------------------------------------------------
# Main diagnosis entry point
# ---------------------------------------------------------------------------

def diagnose_and_recover(
    project_dir: Path,
    meta: dict,
    error_msg: str,
    stage: str,
) -> RecoveryAction:
    """Analyze an error and return a recovery action.

    Args:
        project_dir: Path to the project directory
        meta: Project metadata dict
        error_msg: The error message/traceback
        stage: Current pipeline stage (IDEA, PLAN, RUN, ANALYZE, WRITE, PUBLISH)

    Returns:
        RecoveryAction with strategy, retry flag, and description
    """
    category = classify_error(error_msg)
    LOGGER.info("error classified as '%s' in stage %s", category, stage)

    recovery_history = meta.get("recovery_history", [])
    past_strategies = {r.get("strategy") for r in recovery_history}

    if category == "infrastructure":
        action = _fix_infrastructure(error_msg, project_dir, stage)
    elif category == "resource":
        if "reduce_resources" in past_strategies:
            action = RecoveryAction(
                strategy="escalate_to_revise",
                retry=False,
                description="Already reduced resources once, escalating to REVISE",
            )
        else:
            action = _fix_resource(error_msg, project_dir)
    elif category == "logic":
        action = _fix_logic(error_msg, project_dir, stage)
    else:
        action = RecoveryAction(
            strategy="unknown_retry",
            retry=True,
            description=f"Unknown error type, retrying: {error_msg[:200]}",
        )

    recovery_history.append({
        "stage": stage,
        "category": category,
        "strategy": action.strategy,
        "description": action.description,
        "error_snippet": error_msg[:500],
    })
    meta["recovery_history"] = recovery_history

    LOGGER.info("recovery action: strategy=%s, retry=%s, desc=%s",
                action.strategy, action.retry, action.description)

    return action


# ---------------------------------------------------------------------------
# REVISE: LLM-driven failure reflection
# ---------------------------------------------------------------------------

def run_revise(project_dir: Path, repo_root: Path) -> bool:
    """Use LLM to analyze failures and generate a revised idea.

    Returns True if a revised idea was generated, False otherwise.
    """
    from ..llm.prompts import REVISE_SYSTEM, REVISE_USER
    from ..llm.router import get_router

    idea_path = project_dir / "00_idea" / "idea.json"
    if not idea_path.exists():
        LOGGER.warning("no idea.json found for REVISE")
        return False

    idea = json.loads(idea_path.read_text())
    meta = json.loads((project_dir / "meta.json").read_text())
    recovery_history = meta.get("recovery_history", [])

    failure_lines = []
    error_lines = []
    for r in recovery_history:
        failure_lines.append(f"- Stage: {r['stage']}, Category: {r['category']}, "
                             f"Strategy: {r['strategy']}, Result: {r['description']}")
        error_lines.append(r.get("error_snippet", ""))

    user_prompt = REVISE_USER.format(
        title=idea.get("title", "Untitled"),
        hypothesis=idea.get("hypothesis", ""),
        method_json=json.dumps(idea.get("method", {}), indent=2),
        failure_history="\n".join(failure_lines) or "No recorded failures",
        error_messages="\n---\n".join(error_lines[-3:]) or "No error messages",
    )

    try:
        router = get_router("azure_gpt4o")
        raw = router.generate(REVISE_SYSTEM, user_prompt, json_mode=True)
        revised_idea = json.loads(raw)

        revised_idea["revised_from"] = idea.get("title", "unknown")
        revised_idea["revision_count"] = idea.get("revision_count", 0) + 1

        (project_dir / "00_idea" / "idea.json").write_text(
            json.dumps(revised_idea, indent=2)
        )

        LOGGER.info("REVISE generated revised idea: %s", revised_idea.get("title", "?"))
        return True

    except Exception as exc:
        LOGGER.error("REVISE failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Slurm-specific recovery helpers
# ---------------------------------------------------------------------------

def handle_slurm_failure(job_state: str, project_dir: Path, meta: dict) -> RecoveryAction:
    """Handle Slurm-specific job failure states."""
    if job_state == "PREEMPTED":
        return RecoveryAction(
            strategy="resubmit_preempted",
            retry=True,
            description="Job preempted on scavenge partition, resubmitting",
        )

    if job_state == "TIMEOUT":
        slurm_cfg_path = project_dir.parent.parent / "config" / "system.yaml"
        if slurm_cfg_path.exists():
            sys_cfg = yaml.safe_load(slurm_cfg_path.read_text())
            current_time = sys_cfg.get("slurm", {}).get("time", "06:00:00")
            hours = int(current_time.split(":")[0])
            new_hours = min(hours * 2, 24)
            sys_cfg.setdefault("slurm", {})["time"] = f"{new_hours:02d}:00:00"
            slurm_cfg_path.write_text(yaml.dump(sys_cfg, default_flow_style=False))

        return RecoveryAction(
            strategy="extend_slurm_timeout",
            retry=True,
            description=f"Extended Slurm time limit",
        )

    if job_state == "NODE_FAIL":
        return RecoveryAction(
            strategy="exclude_node",
            retry=True,
            description="Node failed, will retry on different node",
        )

    if job_state == "OUT_OF_MEMORY":
        return _fix_resource("OUT_OF_MEMORY", project_dir)

    return RecoveryAction(
        strategy="slurm_generic_retry",
        retry=True,
        description=f"Slurm job ended with {job_state}, retrying",
    )
