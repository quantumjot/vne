import torch

import numpy as np


class Encoder3D(torch.nn.Module):
    """Really simple encoder."""

    def __init__(self):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Conv3d(1, 8, 3, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv3d(8, 16, 3, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv3d(16, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv3d(32, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Flatten(),
        )

        unflat_shape = (64, 2, 2, 2)
        self.flat_shape = np.prod(unflat_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
