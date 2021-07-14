import os
import random
import string

import numpy as np
from PIL import Image, ImageDraw, ImageFont
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


def create_heterogeneous_image(shape: tuple, n_objects: int = 100):
    """Create a big training image."""
    big_image = np.zeros(shape, dtype=np.uint8)
    locations = np.random.randint(0, shape[0] - 64, (n_objects, 2))

    for i in range(n_objects):
        char = random.choice(CHARS)
        angle = random.randint(0, 360)
        sx = slice(locations[i, 0], locations[i, 0] + 64)
        sy = slice(locations[i, 1], locations[i, 1] + 64)
        big_image[sx, sy] = np.maximum(
            big_image[sx, sy], create_example(char, angle)
        )

    return big_image


if __name__ == "__main__":
    pass
