import numpy as np

from vne.simulate import create_heterogeneous_image, set_default_font

set_default_font()


def test_simulated_image():
    shape = (1024, 1024)
    n_objects = 10
    img, bb, lbl = create_heterogeneous_image(shape=shape, n_objects=n_objects)
    assert img.shape == shape

def test_simulated_image_rng():
    shape = (1024, 1024)
    n_objects = 10

    seed = 1234
    rng = np.random.default_rng(seed)
    data_rng1_0 = create_heterogeneous_image(shape=shape, n_objects=n_objects, rng=rng)
    data_rng1_1 = create_heterogeneous_image(shape=shape, n_objects=n_objects, rng=rng)
    data_rng2_0 = create_heterogeneous_image(shape=shape, n_objects=n_objects, rng=np.random.default_rng(seed))

    # image data (index 0) should be identical when using an identical prng
    assert (data_rng1_0[0] == data_rng2_0[0]).all()

    # different draws from same rng expected to produce different image
    assert (data_rng1_0[0] != data_rng1_1[0]).any()
