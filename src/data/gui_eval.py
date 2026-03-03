"""Evaluation module for GUI Agent action planning on Mind2Web."""

import re
from collections import Counter
from typing import Optional

from ..utils.log import get_logger

LOGGER = get_logger(__name__)


def parse_model_output(text: str) -> dict:
    """Parse model output into structured action prediction.

    Expected format:
        Action: CLICK|TYPE|SELECT
        Element: <element description>
        Value: <value or N/A>
    """
    action_match = re.search(r'Action:\s*(CLICK|TYPE|SELECT|click|type|select)', text, re.IGNORECASE)
    element_match = re.search(r'Element:\s*(.+?)(?:\n|$)', text)
    value_match = re.search(r'Value:\s*(.+?)(?:\n|$)', text)

    return {
        "action_type": action_match.group(1).upper() if action_match else "UNKNOWN",
        "element": element_match.group(1).strip() if element_match else "",
        "value": value_match.group(1).strip() if value_match else "N/A",
    }


def _element_match(pred_element: str, gold_element: str, fuzzy: bool = True) -> bool:
    """Check if predicted element matches gold element."""
    pred = pred_element.lower().strip()
    gold = gold_element.lower().strip()

    if pred == gold:
        return True

    if fuzzy:
        pred_words = set(pred.split())
        gold_words = set(gold.split())
        if pred_words and gold_words:
            overlap = len(pred_words & gold_words) / max(len(gold_words), 1)
            if overlap >= 0.5:
                return True

    return False


def evaluate_predictions(
    predictions: list[dict],
    gold_labels: list[dict],
) -> dict:
    """Compute GUI Agent evaluation metrics.

    Args:
        predictions: list of parsed model outputs [{action_type, element, value}]
        gold_labels: list of gold labels [{action_type, element, value}]

    Returns:
        dict with element_accuracy, action_f1, step_success_rate, per_action_f1
    """
    assert len(predictions) == len(gold_labels), (
        f"predictions ({len(predictions)}) != gold ({len(gold_labels)})"
    )

    n = len(predictions)
    if n == 0:
        return {
            "element_accuracy": 0.0,
            "action_f1": 0.0,
            "step_success_rate": 0.0,
            "n_samples": 0,
        }

    element_correct = 0
    action_correct = 0
    step_correct = 0

    action_tp = Counter()
    action_fp = Counter()
    action_fn = Counter()

    for pred, gold in zip(predictions, gold_labels):
        pred_action = pred.get("action_type", "UNKNOWN").upper()
        gold_action = gold.get("action_type", "UNKNOWN").upper()

        pred_elem = pred.get("element", "")
        gold_elem = gold.get("element", "")

        action_ok = pred_action == gold_action
        element_ok = _element_match(pred_elem, gold_elem)

        if action_ok:
            action_correct += 1
            action_tp[gold_action] += 1
        else:
            action_fp[pred_action] += 1
            action_fn[gold_action] += 1

        if element_ok:
            element_correct += 1

        if action_ok and element_ok:
            step_correct += 1

    element_accuracy = round(element_correct / n, 4)
    action_accuracy = round(action_correct / n, 4)
    step_success_rate = round(step_correct / n, 4)

    all_actions = set(list(action_tp.keys()) + list(action_fp.keys()) + list(action_fn.keys()))
    per_action_f1 = {}
    for action in all_actions:
        tp = action_tp[action]
        fp = action_fp[action]
        fn = action_fn[action]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_action_f1[action] = round(f1, 4)

    macro_f1 = round(sum(per_action_f1.values()) / max(len(per_action_f1), 1), 4)

    return {
        "element_accuracy": element_accuracy,
        "action_f1": macro_f1,
        "action_accuracy": action_accuracy,
        "step_success_rate": step_success_rate,
        "n_samples": n,
        "per_action_f1": per_action_f1,
    }


def run_model_evaluation(
    model,
    tokenizer,
    eval_examples: list[dict],
    max_new_tokens: int = 256,
    batch_size: int = 1,
) -> dict:
    """Run model inference on eval examples and compute metrics.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        eval_examples: list of {input, output, meta} dicts
        max_new_tokens: max tokens to generate

    Returns:
        dict with all metrics
    """
    import torch

    predictions = []
    gold_labels = []

    model.eval()
    LOGGER.info("evaluating on %d examples...", len(eval_examples))

    for i, ex in enumerate(eval_examples):
        messages = [{"role": "user", "content": ex["input"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt", truncation=True,
                           max_length=tokenizer.model_max_length).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)

        pred = parse_model_output(response)
        predictions.append(pred)

        gold_labels.append({
            "action_type": ex["meta"]["action_type"],
            "element": ex["meta"]["element"],
            "value": ex["meta"]["value"],
        })

        if (i + 1) % 50 == 0:
            LOGGER.info("evaluated %d / %d", i + 1, len(eval_examples))

    metrics = evaluate_predictions(predictions, gold_labels)
    LOGGER.info(
        "evaluation done: elem_acc=%.4f, action_f1=%.4f, step_sr=%.4f",
        metrics["element_accuracy"],
        metrics["action_f1"],
        metrics["step_success_rate"],
    )
    return metrics
