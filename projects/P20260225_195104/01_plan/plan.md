# Experiment Plan

**Hypothesis:** Increasing the learning rate and the number of training epochs will reduce eval_loss by allowing the model to learn more effectively over a longer period and with a faster update rate.

## Design
- **Control:** baseline config
- **Treatment:** Increase learning_rate to 0.001 and num_train_epochs to 3 while keeping all other parameters constant.
- **Variables:** learning_rate, num_train_epochs
- **Metric:** eval_loss
- **Seeds:** [42, 123, 7]
- **Models:** Qwen/Qwen3-VL-4B-Instruct, Qwen/Qwen3-4B-Instruct-2507
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
    learning_rate: 0.0002
    num_train_epochs: 3
    warmup_ratio: 0.03
```
