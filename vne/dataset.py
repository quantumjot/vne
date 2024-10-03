from typing import Callable, Optional, Tuple

import numpy as np
import torch

from . import simulate

NUM_IMAGES = 100


class SimulatedDataset(torch.utils.data.Dataset):
    """SimulateDataset container.

    A PyTorch compatible `Dataset` that returns unique simulated datasets.

    Parameters
    ----------
    preprocessor : Callable
        A function that performs preprocessing on a simulated image.
    simulator : Callable
        A function that simulates an image.
    size : tuple
        The size of the simulated images, e.g. (512, 512)
    n_objects : tuple
        The range (low, high) of the number of objects to randomly generate
        per example image.
    return_masks : bool
        Return binary masks for simulated images.
    transforms :
        Transforms to apply for data augmentation.

    """

    def __init__(
        self,
        preprocessor: Optional[Callable] = None,
        simulator: Callable = simulate.create_heterogeneous_image,
        n_objects: Tuple[int] = (50, 150),
        size: Tuple[int] = (512, 512),
        return_masks: bool = False,
        transforms=None,
        rng=np.random.default_rng(),
    ):
        super().__init__()
        self.transforms = transforms
        self.preprocessor = (
            preprocessor if preprocessor is not None else lambda x: x
        )
        self.simulator = simulator
        self.n_objects = n_objects
        self.image_size = size
        self.return_masks = return_masks
        self.rng = rng

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, dict, int]:
        n_objects = self.rng.integers(*self.n_objects)

        img, boxes, labels = self.simulator(
            self.image_size,
            n_objects=n_objects,
            return_masks=self.return_masks,
            rng=self.rng,
        )

        # run the preprocessor to generate the final image
        img = self.preprocessor(img)

        # need to transpose the image to make sure W, H are correct for RPN
        img = (img - np.min(img)) / np.ptp(img)
        img = img.T

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # binary classification
        # labels = torch.as_tensor(labels, dtype=torch.int64, device=DEVICE)
        labels = torch.ones((n_objects,), dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((n_objects,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # TODO(arl): implement the transforms for data augmentation
        if self.transforms is not None:
            img = self.transforms(img)

        # TODO(arl): are we returning binary masks?

        # now convert to tensor
        img = torch.as_tensor(img[np.newaxis, ...], dtype=torch.float32)

        return img, target, image_id

    def __len__(self) -> int:
        return NUM_IMAGES
