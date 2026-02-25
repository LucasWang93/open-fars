# Experiment Plan

**Hypothesis:** Increasing the learning rate and warmup ratio will improve the model's convergence and reduce eval_loss.

## Design
- **Control:** baseline config
- **Treatment:** Adjust learning_rate to 0.0002 and warmup_ratio to 0.1.
- **Variables:** learning_rate, warmup_ratio
- **Metric:** eval_loss
- **Seeds:** [42, 123, 7]
- **Models:** Qwen/Qwen3-4B-Instruct-2507
- **Budget:** ~120 minutes

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
    rank: 16
  train:
    learning_rate: 0.0002
    num_train_epochs: 1
    warmup_ratio: 0.1
```
