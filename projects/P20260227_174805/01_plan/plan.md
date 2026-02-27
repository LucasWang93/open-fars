# Experiment Plan

**Hypothesis:** Increasing the learning rate and LoRA rank together will lead to more efficient fine-tuning and improved representational rank for task-specific adaptation.

**Based on pattern:** pat_86f7f6506b31

**Rationale:** The pattern indicates that combining differential learning rates with modifications to low-rank matrices, such as increasing LoRA rank, improves task-specific adaptation and representational capacity. A higher learning rate complements this by enabling faster updates, potentially enhancing adaptation further.

## Design
- **Control:** baseline config
- **Treatment:** Increase learning_rate and lora.rank together. Test configurations with learning_rate values such as 0.0002, 0.0005, and 0.001, while increasing lora.rank to 32, 64, and 128 respectively.
- **Variables:** learning_rate, lora.rank
- **Metric:** eval_loss
- **Seeds:** [42, 123, 7]
- **Models:** Qwen/Qwen3-4B-Instruct-2507
- **Budget:** ~240 minutes

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
    rank: 32
  train:
    learning_rate: 0.0002
    num_train_epochs: 1
    warmup_ratio: 0.03
```
