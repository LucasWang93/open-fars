# Experiment Plan

**Hypothesis:** Applying lr_down, lr_up to the baseline will improve score.

## Design
- **Control:** baseline config
- **Treatment:** apply lr_down, lr_up
- **Variables:** train.lr
- **Metric:** score
- **Seeds:** [11, 22, 33]
- **Budget:** ~6 minutes

## Configs
```yaml
baseline:
  gen:
    temperature: 1.0
  train:
    epochs: 5
    lr: 0.0001
seeds:
- 11
- 22
- 33
treatment:
  gen:
    temperature: 1.0
  train:
    epochs: 5
    lr: 0.0002
```
