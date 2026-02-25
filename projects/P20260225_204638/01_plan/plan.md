# Experiment Plan

**Hypothesis:** Decreasing the learning rate and increasing the warmup ratio will lead to a more stable training process, potentially improving eval_loss.

## Design
- **Control:** baseline config
- **Treatment:** Learning rate will be decreased, and warmup ratio will be increased. Specific values will be tested against the baseline for seeds [42, 123, 7].
- **Variables:** learning_rate, warmup_ratio
- **Metric:** eval_loss
- **Seeds:** [42, 123, 7]
- **Models:** Qwen/Qwen3-VL-4B-Instruct, Qwen/Qwen3-4B-Instruct-2507
- **Budget:** ~180 minutes

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
- Qwen/Qwen3-VL-4B-Instruct
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
    learning_rate: 5.0e-05
    num_train_epochs: 1
    warmup_ratio: 0.1
```
