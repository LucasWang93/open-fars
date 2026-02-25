# Idea

**Hypothesis:** Increasing the learning rate and warmup ratio will help the model converge faster and reduce eval_loss.

**Actions:** lr_up, warmup_more

- `lr_up`: {"train.learning_rate": 0.0002}

- `warmup_more`: {"train.warmup_ratio": 0.1}
