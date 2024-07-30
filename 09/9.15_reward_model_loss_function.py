import torch.nn as nn
from transformers import Trainer


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Compute rewards for the first set of inputs
        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]

        # Compute rewards for the second set of inputs
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]

        # Calculate loss as the negative log-sigmoid of the difference between rewards_j and rewards_k
        # This ensures that the model learns to prefer higher rewards
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()

        # Return the loss value, and optionally return the computed rewards if requested
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}

        return loss
