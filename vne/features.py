import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP

# from sklearn.preprocessing import normalize

species = ["H"]
rcut = 12.0
nmax = 16
lmax = 9

# Setting up the SOAP descriptor
soap = SOAP(species=species, periodic=False, rcut=rcut, nmax=nmax, lmax=lmax,)


def image_to_atoms(image: np.ndarray) -> Atoms:
    """Convert a binary image to an atomic description compatible with SOAPs.

    Parameters
    ----------
    image : np.ndarray
        The binary image to convert.

    Returns
    -------
    atoms : ase.Atoms
        The description of the binary image as Atoms.
    """
    coords = np.nonzero(image)
    positions = [(i + 0.5, j + 0.5, 0.0) for i, j in zip(*coords)]
    symbols = ["H"] * len(positions)
    atoms = Atoms(symbols=symbols, positions=positions)
    return atoms


def image_to_features(image: np.ndarray) -> np.ndarray:
    """Calculate the features from an image."""

    atoms = image_to_atoms(image)
    features = soap.create(atoms, positions=([32, 32, 0],))
    return features  # normalize(features)
