# Idea

**Hypothesis:** Increasing the learning rate and the number of training epochs will reduce eval_loss by allowing the model to learn more effectively over a longer period and with a faster update rate.

**Actions:** lr_up, epochs_up

- `lr_up`: {"train.learning_rate": 0.0002}

- `epochs_up`: {"train.num_train_epochs": 3}
