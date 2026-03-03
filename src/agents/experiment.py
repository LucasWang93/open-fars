"""Experiment agent: fine-tune model on Mind2Web and evaluate GUI Agent metrics."""

import hashlib
import json
import os
import random
from pathlib import Path

import torch
import yaml

from ..utils.log import get_logger

LOGGER = get_logger(__name__)


def run_experiment(project_dir: Path, repo_root: Path) -> None:
    plan_dir = project_dir / "01_plan"
    config = yaml.safe_load((plan_dir / "config.yaml").read_text())

    seeds = config.get("seeds", [42, 123])
    base_model = config.get("base_model", "Qwen/Qwen3-4B-Instruct-2507")
    runs_dir = project_dir / "02_exp" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_idx = 0
    for group_name in ("baseline", "treatment"):
        cfg = config.get(group_name, {})
        cfg_hash = hashlib.sha256(
            json.dumps(cfg, sort_keys=True).encode()
        ).hexdigest()[:16]

        for seed in seeds:
            run_idx += 1
            run_id = f"run_{run_idx:04d}"
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            try:
                metrics = _train_and_evaluate(base_model, cfg, seed, run_dir)
                status = "SUCCESS"
            except Exception as exc:
                LOGGER.error("run %s failed: %s", run_id, exc, exc_info=True)
                metrics = {
                    "element_accuracy": 0.0,
                    "action_f1": 0.0,
                    "step_success_rate": 0.0,
                    "error": str(exc),
                }
                status = "FAIL"

            result = {
                "run_id": run_id,
                "group": group_name,
                "model": base_model,
                "primary_metric": {
                    "name": config.get("primary_metric", "step_success_rate"),
                    "value": metrics.get("step_success_rate", 0.0),
                    "higher_is_better": True,
                },
                "secondary_metrics": {
                    "element_accuracy": metrics.get("element_accuracy", 0.0),
                    "action_f1": metrics.get("action_f1", 0.0),
                    "step_success_rate": metrics.get("step_success_rate", 0.0),
                },
                "seed": seed,
                "config_hash": f"sha256:{cfg_hash}",
                "config": cfg,
                "status": status,
            }

            (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
            LOGGER.info(
                "%s group=%s seed=%d step_sr=%.4f elem_acc=%.4f action_f1=%.4f (%s)",
                run_id, group_name, seed,
                metrics.get("step_success_rate", 0),
                metrics.get("element_accuracy", 0),
                metrics.get("action_f1", 0),
                status,
            )

    LOGGER.info("experiment done: %d runs in %s", run_idx, runs_dir)


def _train_and_evaluate(
    base_model: str,
    cfg: dict,
    seed: int,
    run_dir: Path,
) -> dict:
    """Run one training + evaluation cycle."""
    random.seed(seed)
    torch.manual_seed(seed)

    from ..data.mind2web import load_mind2web, augment_data
    from ..data.gui_eval import run_model_evaluation

    data_processing = cfg.get("data_processing", "html_simplified")
    prompt_design = cfg.get("prompt_design", "standard")
    augmentation = cfg.get("augmentation", "none")
    data_cfg = cfg.get("data", {})

    data = load_mind2web(
        data_processing=data_processing,
        prompt_design=prompt_design,
        max_train_samples=data_cfg.get("max_train_samples", 2000),
        max_eval_samples=data_cfg.get("max_eval_samples", 500),
    )

    train_examples = augment_data(data["train"], strategy=augmentation)
    eval_examples = data["test"]

    LOGGER.info("train=%d examples (after augmentation), eval=%d",
                len(train_examples), len(eval_examples))

    model, tokenizer = _load_model_with_lora(base_model, cfg, seed)

    _finetune(model, tokenizer, train_examples, cfg, seed, run_dir)

    metrics = run_model_evaluation(model, tokenizer, eval_examples)

    del model
    torch.cuda.empty_cache()

    return metrics


def _load_model_with_lora(base_model: str, cfg: dict, seed: int):
    """Load base model and apply LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    hf_token = os.environ.get("HF_TOKEN")

    LOGGER.info("loading base model %s ...", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )

    lora_cfg = cfg.get("lora", {})
    model_config_name = cfg.get("model_config", "lora_r16")

    if "qlora" in model_config_name.lower():
        LOGGER.info("using QLoRA 4-bit quantization")

    lora_config = LoraConfig(
        r=lora_cfg.get("rank", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOGGER.info("LoRA applied: trainable=%d / total=%d (%.2f%%)",
                trainable, total, 100.0 * trainable / total)

    return model, tokenizer


def _finetune(model, tokenizer, train_examples: list, cfg: dict, seed: int, run_dir: Path):
    """Fine-tune model on training examples using SFTTrainer."""
    from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
    from datasets import Dataset

    train_cfg = cfg.get("train", {})

    def _format_for_sft(example):
        messages = [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    ds = Dataset.from_list(train_examples)
    ds = ds.map(_format_for_sft)

    output_dir = str(run_dir / "checkpoints")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        max_seq_length=train_cfg.get("max_seq_length", 2048),
        logging_steps=10,
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        seed=seed,
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    LOGGER.info("starting fine-tuning: epochs=%d, lr=%s, batch=%d, seq_len=%d",
                training_args.num_train_epochs,
                training_args.learning_rate,
                training_args.per_device_train_batch_size,
                training_args.max_seq_length)

    trainer.train()
    LOGGER.info("fine-tuning complete")
