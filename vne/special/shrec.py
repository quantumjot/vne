import enum
import itertools
import mrcfile
import torch
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd

from pathlib import Path
from scipy.ndimage import rotate
from typing import List, Tuple
from tqdm import tqdm

from vne.utils.utils import InfinitePaddedImage


class SHRECModelType(str, enum.Enum):
    RECONSTRUCTION = "reconstruction"
    GRANDMODEL = "grandmodel"
    CLASSMASK = "classmask"


class SHRECModel:
    def __init__(
        self,
        model_path: Path,
        *,
        model_type: SHRECModelType = "reconstruction",
        exclude: List[str] = ["vesicle", "fiducial", "4V94"],
        boxsize: Tuple[int, int, int] = (32, 32, 32),
        augment: bool = False,
    ) -> None:
        self.model_path = model_path
        self.model_type = SHRECModelType(model_type)
        particles = pd.read_csv(
            model_path / "particle_locations.txt",
            names=[
                "class",
                "x",
                "y",
                "z",
                "rotation_Z1",
                "rotation_X",
                "rotation_Z2",
            ],
            sep=" ",
        )
        self.exclude = exclude
        self.particles = particles[~particles["class"].isin(exclude)]
        self.data = None
        self.boxsize = np.array(boxsize)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.particles)

    def keys(self) -> List[str]:
        return list(set(self.particles["class"].tolist()))

    def __getitem__(self, idx: int) -> Tuple[npt.NDArray, str]:
        """Get the particle and class."""

        if self.data is None:
            self._load_volume()

        particle = self.particles.iloc[idx]

        # if augmenting, crop a slightly larger volume
        cropsize = (
            np.array(self.boxsize) + 16 if self.augment else self.boxsize
        )

        slices = [
            slice(
                particle[dim] - cropsize[d] // 2,
                particle[dim] - cropsize[d] // 2 + cropsize[d],
                1,
            )
            for d, dim in enumerate(["z", "y", "x"])
        ]

        sz, sy, sx = slices
        subvolume = self.data[sz, sy, sx]
        return subvolume, str(particle["class"])

    def _load_volume(self) -> None:
        if self.data is not None:
            return

        data_fn = self.model_path / f"{self.model_type}.mrc"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with mrcfile.open(data_fn, permissive=True) as mrc:
                self.data = InfinitePaddedImage(np.asarray(mrc.data))


class SHRECDataset(torch.utils.data.Dataset):
    def __init__(self, root_path: Path, **kwargs):
        super().__init__()

        self.boxsize = kwargs.get("boxsize", (32, 32, 32))
        self.augment = kwargs.get("augment", True)
        self.model_type = kwargs.get("model_type", SHRECModelType.GRANDMODEL)

        self._subtomo_fn = (
            Path(root_path) / f"subtomograms_{self.model_type.upper()}.npz"
        )

        if self._subtomo_fn.exists():
            print(f"Loading dataset: {self._subtomo_fn}...")
            dataset = np.load(self._subtomo_fn)
            self._subvolumes = dataset["volumes"]
            self._molecule_ids = dataset["molecule_ids"]
            self._keys = dataset["keys"].tolist()
            self._n_molecules = self._subvolumes.shape[0]
            return

        self.models = [
            SHRECModel(model_path, **kwargs)
            for model_path in root_path.iterdir()
            if model_path.stem.startswith("model_")
        ]
        # self.models = self.models[0:1]
        self._n_molecules = sum(len(model) for model in self.models)
        self.extract_subvolumes()

    def extract_subvolumes(self):
        subvolume_shape = self.models[0][0][0].shape
        self._subvolumes = np.zeros(
            (self._n_molecules, *subvolume_shape), dtype=np.float32
        )
        self._molecule_ids = np.zeros((self._n_molecules,), dtype=np.uint8)
        jdx = 0
        for model in tqdm(self.models, desc="Extracting subvolumes"):
            for idx in range(len(model)):
                self._subvolumes[jdx, ...], molecule_id = model[idx]
                self._molecule_ids[jdx] = self._keys.index(molecule_id)
                jdx += 1

        self._keys = list(
            set(
                itertools.chain.from_iterable(
                    [model.keys() for model in self.models]
                )
            )
        )

        np.savez(
            self._subtomo_fn,
            volumes=self._subvolumes,
            molecule_ids=self._molecule_ids,
            keys=self._keys,
        )

    def keys(self) -> List[str]:
        return self._keys

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        subvolume = self._subvolumes[idx]
        molecule_idx = self._molecule_ids[idx]

        if self.augment:
            theta = np.random.uniform(low=-30.0, high=30, size=(3,))

            for d in range(3):
                axis = (d, (d + 1) % 3)
                subvolume = rotate(subvolume, theta[d], axis, reshape=False)

            subvolume = subvolume[8:-8, 8:-8, 8:-8]
            assert subvolume.shape == tuple(self.boxsize), subvolume.shape

        if self.model_type == SHRECModelType.RECONSTRUCTION:
            subvolume = (subvolume - np.mean(subvolume)) / max(
                1 / np.sqrt(subvolume.size), np.std(subvolume)
            )

        subvolume = torch.as_tensor(subvolume[None, ...], dtype=torch.float32)
        return subvolume, molecule_idx

    def __len__(self):
        return self._n_molecules

    def examples(self) -> Tuple[torch.Tensor, List[str]]:
        """Return a set of examples of subvolumes."""
        x_idx = set()
        x_complete = set(range(len(self._keys)))
        examples = []
        examples_class = []
        idx = 0

        while x_complete.difference(x_idx) != set():
            vol, mol_idx = self[idx]
            if mol_idx not in x_idx:
                x_idx.add(mol_idx)
                examples.append(vol)
                examples_class.append(mol_idx)
            idx += 1

        return torch.stack(examples, axis=0), [
            self._keys[idx] for idx in examples_class
        ]
