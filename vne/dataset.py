from typing import Callable, Optional, Tuple
from . import simulate

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random

import mrcfile
import numpy as np

import os
import warnings

NUM_IMAGES = 100


class CustomMNIST(torch.utils.data.Dataset):
    def __init__(self, root, train=True):
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: random_rotate_and_resize(x)),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
        ])
        self.dataset = datasets.MNIST(root=root, train=train, transform=self.transform, download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label


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


class SubTomogram_dataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir,
                 IMAGES_PER_EPOCH,
                 molecule_list,
                 data_format
                 ):

        self.data_format = data_format
        self.image_per_epoch = IMAGES_PER_EPOCH
        self.root_dir = root_dir
        self.paths = sorted([(os.path.join(root_dir, f)) for f in os.listdir(root_dir) if
                             "." + data_format in f and f[:4] in molecule_list])  # list of all the subtomos
        self.mol_id = sorted([f[:4] for f in os.listdir(root_dir) if "." + data_format in f and f[
                                                                                                :4] in molecule_list])  # list of asbnotation molecule labels
        self.proteins = molecule_list  # list of all the classes
        print(self.mol_id)

    def __getitem__(self, idx):
        ## read the subtomogram 
        if self.data_format == "npy":
            data = np.load(self.paths[idx])

        elif self.data_format == "mrc":
            warnings.simplefilter('ignore')  # to mute some warnings produced when opening the tomos

            with mrcfile.open(self.paths[idx], mode='r+', permissive=True) as mrc:
                mrc.header.map = mrcfile.constants.MAP_ID
                mrc = mrc.data

            with mrcfile.open(self.paths[idx]) as mrc:
                data = np.array(mrc.data)

        data = padding(data, 64, 64)
        #### normalise the data convert to torch id and grab the molecule index
        mol = NormalizeData(data)
        mol = torch.as_tensor(mol[np.newaxis, ...], dtype=torch.float32)
        mol_id = list(self.proteins).index(self.mol_id[idx])
        return mol, mol_id

    def keys(self):
        return list(self.proteins)

    def __len__(self):
        return len(self.paths)


class alphanumDataset(torch.utils.data.Dataset):
    def __init__(self, THETA_0, THETA, molecule_list, IMAGES_PER_EPOCH, simulator):
        super().__init__()

        self.molecules = molecule_list
        self.min_theta = THETA_0
        self.max_theta = THETA
        self.simulator = simulator(molecule_list)
        self.image_per_epoch = IMAGES_PER_EPOCH

    def __getitem__(self, idx: int):
        mol = np.random.choice(self.molecules)
        angle = np.random.randint(self.min_theta, self.max_theta)
        density = self.simulator(mol, transform_euler_angles=angle, project=True)
        img = density
        img = np.clip(img, -1, 1)
        img = NormalizeData(img)
        img = torch.as_tensor(img[np.newaxis, ...], dtype=torch.float32)
        return img, self.molecules.index(mol)

    def __len__(self):
        return self.image_per_epoch


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def padding(array, xx, yy, zz=None):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :param zz: desired depth
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]
    if zz is not None:
        z = array.shape[2]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    if zz is not None:

        c = (zz - z) // 2
        cc = zz - c - z
        array = np.pad(array, pad_width=((a, aa), (b, bb), (c, cc)), mode='constant')
    else:
        array = np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')
    return array


# Define a function to apply random rotations and resize to 64x64
def random_rotate_and_resize(image):
    angle = random.uniform(-45, 45)  # Random rotation angle between -30 and 30 degrees
    image = transforms.functional.rotate(image, angle)
    image = transforms.functional.resize(image, (64, 64))
    return image
