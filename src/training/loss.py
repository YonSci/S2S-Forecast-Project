import torch
import torch.nn as nn

class GANLoss(nn.Module):
    """
    Adversarial Loss (MSE or BCE).
    MSE is often more stable for LSGAN (Least Squares GAN).
    """
    def __init__(self, mode='mse'):
        super(GANLoss, self).__init__()
        self.mode = mode
        if mode == 'mse':
            self.loss = nn.MSELoss()
        elif mode == 'bce':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"GAN Loss mode {mode} not implemented.")

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            return torch.ones_like(prediction)
        else:
            return torch.zeros_like(prediction)

    def forward(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)

class WeightedL1Loss(nn.Module):
    """
    L1 Loss (MAE) with optional weighting for extreme values.
    S2S Problem: Models tend to underestimate heavy rain.
    Solution: Weight the loss higher where target rain > threshold.
    """
    def __init__(self, weight=5.0, threshold=0.8):
        # Threshold is in normalized space [-1, 1] or [0, 1]
        super(WeightedL1Loss, self).__init__()
        self.weight = weight
        self.threshold = threshold
        self.l1 = nn.L1Loss(reduction='none')

    def forward(self, prediction, target):
        loss = self.l1(prediction, target)
        
        # Create a boolean mask where rainfall is extreme
        # Assuming Data is normalized to [-1, 1], 0.8 is heavily positive (wet)
        mask = target > self.threshold
        
        # Apply weight
        weighted_loss = loss * (1 + (self.weight - 1) * mask.float())
        
        return weighted_loss.mean()
