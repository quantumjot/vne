from __future__ import annotations

import enum
from pathlib import Path
from typing import Union

import numpy as np

from .. import base
from .voxels import load_binvox

SHAPENET_PATH = Path(
    "/media/quantumjot/DataIII/Data/Turing/ShapeNet/ShapeNetCore.v2"
)


class ShapeNetTaxonomy(str, enum.Enum):
    CHAIRS = "03001627"


class ShapeNetDataset(base.Datasource):
    """ShapeNET dataset.

    Parameters
    ----------
    filepath : path
        A path to the ShapeNET core library.
    synsetId : str
        A string identifier for the object class.
    """

    def __init__(self, filepath: Path, synsetId: Union[str, ShapeNetTaxonomy]):
        self.filepath = filepath
        self.synsetId = synsetId
        self._keys = [
            path.name
            for path in (self.filepath / self.synsetId).iterdir()
            if path.is_dir()
        ]
        self._cache = {}

    def __call__(self, model_id: str) -> np.ndarray:
        if model_id not in self._keys:
            raise KeyError(f"Model {model_id} not found.")

        if model_id in self._cache:
            return self._cache[model_id]

        model_filename = (
            self.filepath
            / self.synsetId
            / model_id
            / "models"
            / "model_normalized.solid.binvox"
        )

        voxels = load_binvox(model_filename)
        self._cache[model_id] = voxels
        return voxels
