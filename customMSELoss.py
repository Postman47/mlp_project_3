import torch
import torch.nn as nn

class CustomMSELoss(nn.Module):
    def __init__(self, margin=0.025):
        super(CustomMSELoss, self).__init__()
        self.margin = margin  # Range around the target, e.g., ± 2/(max_temperature - min_temperature)

    def forward(self, input, target):
        # Define lower and upper bounds
        lower_bound = target - self.margin
        upper_bound = target + self.margin

        # Calculate the squared error only for inputs outside the range
        loss = torch.where(
            (input < lower_bound) | (input > upper_bound),  # Condition
            (input - target) ** 2,                         # Apply MSE
            torch.tensor(0.0, device=input.device)          # Zero loss for in-range values
        )

        return loss.mean()  # Return the mean loss