import numpy as np

from .simulate import create_heterogeneous_image


class VNEModel:
    def __init__(self):
        pass

    def create_heterogeneous_image(
        self, dummy: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        return create_heterogeneous_image(*args, **kwargs)
