import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP

species = ["H"]
rcut = 12.0
nmax = 16
lmax = 9

# Setting up the SOAP descriptor
soap = SOAP(species=species, periodic=False, rcut=rcut, nmax=nmax, lmax=lmax,)


def image_to_atoms(image: np.array) -> Atoms:
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

    if image.ndim == 2:
        coords += (np.zeros_like(coords[0]),)

    coords = np.stack(coords, axis=-1) + 0.5
    positions = coords.tolist()
    symbols = ["H"] * len(positions)
    atoms = Atoms(symbols=symbols, positions=positions)
    return atoms


def image_to_features(image: np.array) -> np.array:
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

    positions = [dim / 2 for dim in image.shape]

    if image.ndim == 2:
        positions += [0.5]

    atoms = image_to_atoms(image)
    features = soap.create(atoms, positions=[positions],)
    return features  # normalize(features)
