import os
import random
import string
from typing import Tuple

from . import shapes

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import regionprops
from skimage.transform import rotate

CHARS = string.ascii_lowercase + string.digits
DEFAULT_FONT = ImageFont.load_default()


def set_default_font(filename: os.PathLike, size: int):
    """Set the feault font from a file."""
    font = ImageFont.truetype(filename, size)
    global DEFAULT_FONT
    DEFAULT_FONT = font


def create_example(text: str, angle: float = 0.0) -> np.ndarray:
    """Create an example character with a random translational offset and
    rotation."""
    raw = np.zeros((64, 64), dtype=np.uint8)
    img = Image.fromarray(raw)
    d = ImageDraw.Draw(img)

    offset = tuple(
        [
            (raw.shape[dim] - DEFAULT_FONT.getsize(text)[dim]) // 2
            for dim in range(raw.ndim)
        ]
    )

    d.text(offset, text, font=DEFAULT_FONT, fill=(255))
    img_arr = np.asarray(img)
    img_rot = rotate(img_arr, angle, preserve_range=True)
    return img_rot


def create_heterogeneous_image(
    shape: Tuple[int], n_objects: int = 100, return_masks: bool = False
) -> Tuple[np.ndarray]:
    """Create a big training image.

    Parameters
    ----------
    shape : tuple(int)
        The shape of the output image.
    n_objects : int
        The number of simulated objects.
    return_masks : bool
        Option to return binary masks for each object as (N, W, H) array.


    Returns
    -------
    image : np.ndarray
        The simulated image.
    bounding_boxes : list
        A list of bounding boxes for each of the objects.
    labels : list
        A list of labels for each of the objects.
    """

    assert len(shape) == 2

    big_image = np.zeros(shape, dtype=np.uint8)
    locations = np.random.randint(0, shape[0] - 64, (n_objects, 2))
    labels = np.random.choice(list(CHARS), size=n_objects)
    bounding_boxes = []

    if return_masks:
        masks = np.zeros((n_objects,) + shape, dtype=np.uint8)

    for i in range(n_objects):
        char = labels[i]
        angle = random.randint(0, 360)
        sx = slice(locations[i, 0], locations[i, 0] + 64)
        sy = slice(locations[i, 1], locations[i, 1] + 64)

        example = create_example(char, angle)

        # hack to get the rotated bounding box
        props = regionprops(example.astype(bool).astype(int))
        bbox = np.array(props[0].bbox) + np.concatenate([locations[i, :]] * 2)
        bounding_boxes.append(bbox)

        big_image[sx, sy] = np.maximum(big_image[sx, sy], example,)

        if return_masks:
            masks[i, sx, sy] = example.astype(bool).astype(np.uint8)

    if return_masks:
        return big_image, bounding_boxes, labels, masks

    return big_image, bounding_boxes, labels

def create_heterogeneous_image_with_shapes(
    shape: Tuple[int], 
    n_objects: int = 100, 
    return_masks: bool = False, 
    input_image: np.ndarray = None, 
    input_bbox: np.ndarray = None, 
    input_labels: np.ndarray = None,
) -> Tuple[np.ndarray]:
    """Create a big training image.
    Parameters
    ----------
    shape : tuple(int)
        The shape of the output image.
    n_objects : int
        The number of simulated objects.
    return_masks : bool
        Option to return binary masks for each object as (N, W, H) array.
    input_image : ndarray
        (Optional) The input image on which to add the shape.
    input_bbox : ndarray
        (Optional) The input array with all the bounding boxes.
    input_labels : ndarray
        (Optional) The input array of the labels for each bounding box.
    
    Returns
    -------
    image : np.ndarray
        The simulated image.
    bounding_boxes : list
        A list of bounding boxes for each of the objects.
    labels : list
        A list of labels for each of the objects.
    """

    assert len(shape) == 2
    
    if input_image is None:
        big_image, pre_bbox, pre_labels = simulate.create_heterogeneous_image(shape, n_objects = 100)
        if pre_bbox is None or pre_labels is None:
            print("error")
    else:
        big_image = input_image
        pre_bbox = input_bbox
        pre_labels = input_labels
    locations = np.random.randint(0, shape[0] - 64, (n_objects, 2))
    labels = np.random.choice(list(CHARS), size=n_objects)
    bounding_boxes = []
    x,y, _ = get_bezier_curve(locations)
    
    x = np.rint(x).astype(int)
    y = np.rint(y).astype(int)
    print(y)
    if return_masks:
        masks = np.zeros((n_objects,) + shape, dtype=np.uint8)

    #for i in range(n_objects):
    char = labels[0]
    angle = random.randint(0, 360)
    
        
        
    example = simulate.create_example(str(char), angle)
        #print(char.dtype,angle)
        # hack to get the rotated bounding box
    props = regionprops(example.astype(bool).astype(int))
    for j in range(x.shape[0]):
        sx = slice(x[j], x[j] + 64)
        sy = slice(y[j], y[j] + 64)
        bbox = np.array(props[0].bbox) + np.concatenate([np.array([int(x[j]),int(y[j])])]*2)
        bounding_boxes.append(bbox)

        big_image[sx, sy] = np.maximum(big_image[sx, sy], example,)

        if return_masks:
            masks[i, sx, sy] = example.astype(bool).astype(np.uint8)
                
    #print(len(bounding_boxes))
    for box in pre_bbox:
        bounding_boxes.append(box)
    #print(len(bounding_boxes))
    np.append(labels,pre_labels)
    if return_masks:
        return big_image, bounding_boxes, labels, masks

    return big_image, bounding_boxes, labels


if __name__ == "__main__":
    pass
