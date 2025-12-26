import torch
import numpy as np

class ClimatologyBaseline:
    def __init__(self, normalizer):
        """
        Climatology Baseline.
        Predictions are just the historical mean (0 anomaly).
        
        Args:
            normalizer (S2SNormalizer): Contains fitted climatology statistics.
        """
        self.normalizer = normalizer

    def predict(self, lead_week_idx, shape):
        """
        Return the climatology for the given week index.
        Rank-3 Tensor: (1, H, W)
        """
        # Get mean precip for this week
        # normalizer.climatology['precip'] is shaped (53, H, W)
        clim = self.normalizer.climatology['precip'][lead_week_idx]
        return torch.from_numpy(clim).float().unsqueeze(0)

class PersistenceBaseline:
    def __init__(self):
        """
        Persistence Baseline.
        Prediction for Week T+Lead is simply the observation at Week T.
        """
        pass

    def predict(self, current_observation):
        """
        Return the current week's observation as next week's forecast.
        """
        return current_observation
