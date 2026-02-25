# Idea

**Hypothesis:** Decreasing the learning rate and increasing the warmup ratio will lead to a more stable training process, potentially improving eval_loss.

**Actions:** lr_down, warmup_more

- `lr_down`: {"train.learning_rate": 5e-05}

- `warmup_more`: {"train.warmup_ratio": 0.1}
