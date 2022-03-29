from vne.simulate import create_heterogeneous_image, set_default_font

set_default_font()


def test_simulated_image():
    shape = (1024, 1024)
    n_objects = 10
    img, bb, lbl = create_heterogeneous_image(shape=shape, n_objects=n_objects)
    assert img.shape == shape
