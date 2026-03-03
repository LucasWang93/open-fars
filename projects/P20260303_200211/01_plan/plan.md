# Experiment Plan

**Title:** Simplified Advantage-Guided Curriculum Learning for Web Navigation
**Hypothesis:** Using a simplified advantage-guided curriculum learning approach with reduced model and data complexity will improve step_success_rate by enabling stable training and effective task sequencing.

**Based on pattern:** None

## Summary
This experiment compares a baseline vanilla supervised fine-tuning (SFT) approach with a simplified advantage-guided curriculum learning method for web navigation. The treatment reduces data and model complexity while restructuring prompts for direct action guidance.

## Design
- **Variables:** training_strategy, data_processing, prompt_design, learning_rate, num_train_epochs, per_device_train_batch_size, max_seq_length, gradient_accumulation_steps
- **Metric:** step_success_rate
- **Seeds:** [42, 123]
- **Model:** Qwen/Qwen3-4B-Instruct-2507
- **Budget:** ~360 minutes

## Baseline
```yaml
augmentation: none
data:
  max_eval_samples: 500
  max_train_samples: 2000
data_processing: html_simplified
lora:
  alpha: 32
  dropout: 0.05
  rank: 16
  target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
model_config: lora_r16
prompt_design: standard
train:
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-05
  max_seq_length: 2048
  num_train_epochs: 3
  per_device_train_batch_size: 4
  warmup_ratio: 0.1
training_strategy: vanilla_sft
```

## Treatment
```yaml
augmentation: none
data:
  max_eval_samples: 500
  max_train_samples: 2000
data_processing: html_only
lora:
  alpha: 32
  dropout: 0.05
  rank: 16
  target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
model_config: lora_r16
prompt_design: direct_action
train:
  gradient_accumulation_steps: 2
  learning_rate: 0.0001
  max_seq_length: 512
  num_train_epochs: 5
  per_device_train_batch_size: 8
  warmup_ratio: 0.1
training_strategy: basic_curriculum_learning
```