import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP


def image_to_atoms(image: np.array, *, scale: float = 1.0) -> Atoms:
    """Convert a binary image to an atomic description compatible with SOAPs.

    Parameters
    ----------
    image : np.array
        The binary image to convert, can be 2- or 3-dimensional.

    Returns
    -------
    atoms : ase.Atoms
        The description of the binary image as Atoms.
    """
    if image.ndim not in (2, 3):
        raise ValueError("Image data should be 2- or 3- dimensional.")

    coords = np.nonzero(image)
    vol = list(image.shape)

    if image.ndim == 2:
        coords += (np.zeros_like(coords[0]),)
        vol += [0]

    # centre the coordinates
    coords = np.stack(coords, axis=-1) + 0.5

    for idx in range(coords.shape[-1]):
        coords[:, idx] -= vol[idx] / 2.0

    # convert to Angstroms
    coords = coords * scale
    positions = coords.tolist()

    symbols = ["H"] * len(positions)
    atoms = Atoms(symbols=symbols, positions=positions)
    return atoms


def image_to_features(
    image: np.array, *, scale: float = 1.0, use_center: bool = True
) -> np.array:
    """Calculate the features from an image.

    Parameters
    ----------
    image : np.ndarray
        The binary image to convert, can be 2- or 3-dimensional.

    Returns
    -------
    features : np.array
        The SOAP descriptor features.
    """

    if image.ndim not in (2, 3):
        raise ValueError("Image data should be 2- or 3- dimensional.")

    # set up the SOAP calculation
    centers = [[0.0, 0.0, 0.0]] if use_center else None
    species = ["H"]
    rcut = 64.0 if use_center else 6
    nmax = 8
    lmax = 6

    # Setting up the SOAP descriptor
    soap = SOAP(
        species=species,
        periodic=False,
        r_cut=rcut,
        n_max=nmax,
        l_max=lmax,
    )

    atoms = image_to_atoms(image, scale=scale)

    features = soap.create(
        atoms,
        centers=centers,
    )
    return features  # normalize(features)
