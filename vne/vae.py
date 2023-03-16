from typing import Tuple

import numpy as np
import torch
from torch import nn

from .base import SpatialDims


class ShapeSimilarityLoss:
    """Shape similarity loss based on pre-calculated shape similarity.

    Parameters
    ----------
    lookup : torch.Tensor (M, M)
        A square symmetric matrix where each column and row is the index of an
        object from the training set, consisting of M different objects. The
        value at (i, j) is a scalar value encoding the shape similarity between
        objects i and j, pre-calculated using some shape (or other) metric. The
        identity of the matrix should be 1 since these objects are the same
        shape. The shape similarity should be normalized to the range (-1, 1).

    Notes
    -----
    The final loss is calculated using L1-norm. This could be changed, e.g.
    L2-norm. Not sure what the best one is yet.
    """

    def __init__(self, lookup: torch.Tensor):
        self.lookup = lookup
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.l1loss = nn.L1Loss()

    def __call__(
        self, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> torch.Tensor:
        """Return the shape similarity loss.

        Parameters
        ----------
        y_true : torch.Tensor (N, )
            A vector of N objects in the mini-batch of the indices representing
            the identity of the objects. These indices should correspond to the
            rows and columns of the `lookup` table.
        y_pred : torch.Tensor (N, latent_dims)
            An array of latent encodings of the N objects.

        Returns
        -------
        loss : torch.Tensor
            The shape similarity loss.
        """
        # first calculate the shape similarity for the real classes
        c = torch.combinations(y_true, r=2, with_replacement=False)
        shape_similarity = self.lookup[c[:, 0], c[:, 1]]

        # now calculate the latent similarity
        z_id = torch.tensor(list(range(y_pred.shape[0])))
        c = torch.combinations(z_id, r=2, with_replacement=False)
        latent_similarity = self.cos(y_pred[c[:, 0], :], y_pred[c[:, 1], :])

        loss = self.l1loss(latent_similarity, shape_similarity)
        return loss


def dims_after_pooling(start: int, n_pools: int) -> int:
    """Calculate the size of a layer after n pooling ops."""
    return start // (2**n_pools)


class ShapeVAE(nn.Module):
    """Shape regularized variational autoencoder.

    Parameters
    ----------
    input_shape : tuple
        A tuple representing the input shape of the data, e.g. (1, 64, 64) for
        images or (1, 64, 64, 64) for a volume with 1 channel.
    latent_dims : int
        The size of the latent representation.
    pose_dims : int
        The size of the pose representation.
    """

    def __init__(
        self,
        input_shape: Tuple[int] = (1, 64, 64),
        latent_dims: int = 8,
        pose_dims: int = 1,
    ):
        super(ShapeVAE, self).__init__()

        channels = input_shape[0]
        spatial_dims = input_shape[1:]
        ndim = len(spatial_dims)

        if ndim not in SpatialDims:
            raise ValueError(
                f"`input_shape` must be have 2 or 3 dimensions, got: {ndim}."
            )

        if ndim == SpatialDims.TWO:
            conv = nn.Conv2d
            conv_T = nn.ConvTranspose2d
        elif ndim == SpatialDims.THREE:
            conv = nn.Conv3d
            conv_T = nn.ConvTranspose3d

        unflat_shape = tuple(
            [
                64,
            ]
            + [dims_after_pooling(ax) for ax in spatial_dims]
        )
        flat_shape = np.prod(unflat_shape)

        self.encoder = nn.Sequential(
            conv(channels, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            conv(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            conv(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            conv(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dims + pose_dims, flat_shape),
            nn.Unflatten(-1, unflat_shape),
            conv_T(64, 32, 3, stride=2),
            nn.ReLU(True),
            conv_T(32, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            conv_T(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            conv_T(8, channels, 2, stride=2, padding=1),
        )

        self.mu = nn.Linear(flat_shape, latent_dims)
        self.log_var = nn.Linear(flat_shape, latent_dims)
        self.pose = nn.Linear(flat_shape, pose_dims)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        mu, log_var, pose = self.encode(x)
        z = self.reparameterise(mu, log_var)
        z_pose = torch.cat([pose, z], dim=-1)
        x = self.decode(z_pose)
        return x, z, z_pose, mu, log_var

    def reparameterise(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        pose = self.pose(encoded)
        return mu, log_var, pose

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
