# Experiment Plan

**Hypothesis:** Increasing the warmup ratio alongside the number of training epochs will improve optimization stability and convergence, leading to lower eval_loss.

**Based on pattern:** pat_c2bd73372434

**Rationale:** The pattern highlights the importance of learning rate schedules, which are closely tied to warmup strategies, in achieving balanced exploration and convergence. Extending warmup can stabilize early optimization dynamics, while increasing epochs allows the model to utilize this stability for more robust convergence.

## Design
- **Control:** baseline config
- **Treatment:** Increase warmup ratio and number of training epochs systematically to assess their effect on optimization stability. Example configurations might include warmup_ratio = 0.1 and num_train_epochs = 3.
- **Variables:** warmup_ratio, num_train_epochs
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
    learning_rate: 0.0001
    num_train_epochs: 3
    warmup_ratio: 0.1
```
