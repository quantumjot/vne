import itertools
import os
from typing import Tuple

import binvox
import numpy as np
from scipy.ndimage import zoom


def bounding_box(img: np.ndarray) -> Tuple[int]:
    """Calculate the bounding box for a volume."""
    dims = img.ndim
    out = []
    for ax in itertools.combinations(reversed(range(dims)), dims - 1):
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return tuple(out)


def load_binvox(
    filename: os.PathLike, *, size: int = 64, centre: bool = True
) -> np.ndarray:
    """Load a binvox file.

    Parameters
    ----------
    filename : str, path
        A filename for the binvox file.
    size : int
        The size of the output.
    centre : bool
        Centre the object based on the calculated bounding box.

    Returns
    -------
    voxels : array
        A numpy array representing the voxels.
    """
    bv = binvox.Binvox.read(filename, "dense")
    voxels = bv.numpy()

    if centre:
        bb = bounding_box(voxels)
        crop = voxels[
            slice(bb[0], bb[1], 1),
            slice(bb[2], bb[3], 1),
            slice(bb[4], bb[5], 1),
        ]

        dx = (voxels.shape[0] // 2) - ((bb[1] - bb[0]) // 2)
        dy = (voxels.shape[1] // 2) - ((bb[3] - bb[2]) // 2)
        dz = (voxels.shape[2] // 2) - ((bb[5] - bb[4]) // 2)

        centred = np.zeros_like(voxels)
        centred[
            slice(dx, dx + crop.shape[0], 1),
            slice(dy, dy + crop.shape[1], 1),
            slice(dz, dz + crop.shape[2], 1),
        ] = crop
        voxels = centred

    # TODO(arl): fix this to resample the voxels to the correct size
    voxels = zoom(voxels, 0.5, order=0)
    return voxels
