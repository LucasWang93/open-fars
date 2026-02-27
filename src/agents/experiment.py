"""Experiment agent: run Qwen3 model evaluations across seeds."""

import hashlib
import json
import os
import random
import torch
from pathlib import Path

import yaml

from ..utils.log import get_logger

LOGGER = get_logger(__name__)

_MODEL_CACHE = {}


def _load_model(model_name: str):
    """Load a HuggingFace model + tokenizer, cached."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    LOGGER.info("loading model %s ...", model_name)
    hf_token = os.environ.get("HF_TOKEN")

    if "VL" in model_name:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto", token=hf_token,
        )
        processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
        _MODEL_CACHE[model_name] = (model, processor, "vl")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto", token=hf_token,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        _MODEL_CACHE[model_name] = (model, tokenizer, "lm")

    LOGGER.info("model %s loaded", model_name)
    return _MODEL_CACHE[model_name]


def _eval_lm(model, tokenizer, config: dict, seed: int) -> dict:
    """Evaluate a language model with given config and seed."""
    random.seed(seed)
    torch.manual_seed(seed)

    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a Python function to compute Fibonacci numbers.",
        "Summarize the key findings of recent climate research.",
    ]

    total_loss = 0.0
    n = 0
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
            n += 1

    avg_loss = total_loss / max(n, 1)
    return {"eval_loss": round(avg_loss, 4), "n_prompts": n}


def _eval_vl(model, processor, config: dict, seed: int) -> dict:
    """Evaluate a vision-language model with given config and seed."""
    random.seed(seed)
    torch.manual_seed(seed)

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "Describe what a well-structured research paper looks like."},
        ]},
    ]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()

    return {"eval_loss": round(loss, 4), "n_prompts": 1}


def run_experiment(project_dir: Path, repo_root: Path) -> None:
    plan_dir = project_dir / "01_plan"
    config_path = plan_dir / "config.yaml"
    configs = yaml.safe_load(config_path.read_text())

    seeds = configs["seeds"]
    models = configs.get("models", ["Qwen/Qwen3-4B-Instruct-2507"])
    runs_dir = project_dir / "02_exp" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_idx = 0
    for model_name in models:
        model_obj, tok_or_proc, model_type = _load_model(model_name)
        model_short = model_name.split("/")[-1]

        for group_name in ("baseline", "treatment"):
            cfg = configs[group_name]
            cfg_hash = hashlib.sha256(
                json.dumps(cfg, sort_keys=True).encode()
            ).hexdigest()[:16]

            for seed in seeds:
                run_idx += 1
                run_id = "run_%04d" % run_idx
                run_dir = runs_dir / run_id
                run_dir.mkdir(parents=True, exist_ok=True)

                try:
                    if model_type == "vl":
                        result = _eval_vl(model_obj, tok_or_proc, cfg, seed)
                    else:
                        result = _eval_lm(model_obj, tok_or_proc, cfg, seed)
                    status = "SUCCESS"
                except Exception as exc:
                    LOGGER.error("run %s failed: %s", run_id, exc)
                    result = {"eval_loss": 999.0, "n_prompts": 0}
                    status = "FAIL"

                metrics = {
                    "run_id": run_id,
                    "group": group_name,
                    "model": model_name,
                    "primary_metric": {
                        "name": "eval_loss",
                        "value": result["eval_loss"],
                        "higher_is_better": False,
                    },
                    "secondary_metrics": result,
                    "seed": seed,
                    "config_hash": "sha256:" + cfg_hash,
                    "config": cfg,
                    "status": status,
                }

                (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
                LOGGER.info(
                    "%s model=%s seed=%d loss=%.4f (%s)",
                    run_id, model_short, seed, result["eval_loss"], group_name,
                )

    LOGGER.info("experiment done: %d runs in %s", run_idx, runs_dir)
