# Experiment Plan

**Hypothesis:** Using a lower learning rate combined with a reduced LoRA rank will lead to more stable optimization and efficient fine-tuning for the task.

**Based on pattern:** pat_32f9328cb109

**Rationale:** Pattern pat_32f9328cb109 highlights that differential learning rates for low-rank matrices improve optimization stability and lead to faster convergence. By lowering both the learning rate and LoRA rank, we expect to enhance stability and achieve better task-specific adaptation.

## Design
- **Control:** baseline config
- **Treatment:** Lower the learning_rate to 0.00005 and reduce the LoRA rank to 8.
- **Variables:** learning_rate, lora.rank
- **Metric:** eval_loss
- **Seeds:** [42, 123, 7]
- **Models:** Qwen/Qwen3-4B-Instruct-2507
- **Budget:** ~90 minutes

## Configs
```yaml
baseline:
  data:
    batch_size: 4
    max_seq_length: 512
  lora:
    alpha: 32
    dropout: 0.05
    rank: 16
  train:
    learning_rate: 0.0001
    num_train_epochs: 1
    warmup_ratio: 0.03
models:
- Qwen/Qwen3-4B-Instruct-2507
seeds:
- 42
- 123
- 7
treatment:
  data:
    batch_size: 4
    max_seq_length: 512
  lora:
    alpha: 32
    dropout: 0.05
    rank: 8
  train:
    learning_rate: 5.0e-05
    num_train_epochs: 1
    warmup_ratio: 0.03
```
