from typing import Callable

import numpy as np
from dscribe.kernels import REMatchKernel
from sklearn.preprocessing import normalize

# Calculates the similarity with the REMatch kernel and a linear metric. The
# result will be a full similarity matrix.
RE_KERNEL = REMatchKernel(metric="linear", alpha=1, threshold=1e-6)


def _distance_metric(dist: float) -> float:
    """A distance metric."""
    if dist < 1.0:
        return 1 + 1 / np.log10(1 - dist)
    else:
        return 1


def similarity(
    *args: np.ndarray,
    metric: Callable = _distance_metric,
    normalize_features: bool = True
) -> float:
    """Calculate the similarity of two feature vectors.

    Parameters
    ----------
    *args : np.ndarray
        The feature vectors.
    metric : Callable
        A function to calculate the similarity from the distance.
    normalize_features : bool
        Normalize the feature vectors before calculating the similarity.

    Returns
    -------
    similarity : float
        A scalar value representing the similarity of the feature vectors.
    """

    # we can only deal with two inputs at the moment. However, if we got
    # multiple features, perhaps we could calculate a similarity matrix?
    assert len(args) == 2
    assert all([isinstance(arg, np.ndarray) for arg in args])

    features = [normalize(f) for f in args] if normalize_features else args

    # calculate global comparison matrix
    re_kernel = RE_KERNEL.create(features)

    # global similarity (average of local over matched pairs of environments)
    dist = RE_KERNEL.get_global_similarity(re_kernel)

    return metric(dist)
