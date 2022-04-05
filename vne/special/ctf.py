import numpy as np
from scipy.fft import fft2, fftshift, ifft2

PLANCK_CONSTANT = 6.626e-34
ELECTRON_CHARGE = 1.603e-19
SPEED_OF_LIGHT = 3e8
ELECTRON_MASS = 9.109e-31


def electron_wavelength(U: float) -> float:
    """Electron wavelength in units of Angstroms."""
    numerator = PLANCK_CONSTANT / np.sqrt(
        2 * ELECTRON_MASS * ELECTRON_CHARGE * U
    )
    denominator = np.sqrt(
        1
        + ELECTRON_CHARGE
        * U
        / (2 * ELECTRON_MASS * SPEED_OF_LIGHT * SPEED_OF_LIGHT)
    )
    return numerator / denominator * 1e10


def contrast_transfer_function(
    pixel_size: float = 1.0,
    box_size: int = 300,
    defocus: float = 10e3,
    astigmatism: float = 200.0,
    astigmatism_angle: float = 90.0,
    spherical_abberation: float = 2.7,
    energy: float = 300,
    amplitude_contrast: float = 0.1,
    phase_shift: float = 0,
) -> np.ndarray:
    """Contrast Transfer Function.
    Parameters
    ----------
    pixel_size : float
        The pixel size of the calculated CTF in Angstroms per pixel.
    box_size : int
        The size of the output array in pixels.
    defocus : float
        The defocus in Angstroms.
    astigmatism : float
        The astigmatism in Angstroms.
    astigmatism_angle : float
        The astigmatism angle in degrees.
    spherical_abberation : float
        The spherical abberation in mm.
    energy : float
        The energy in kV.
    amplitude_contrast : float
        Amplitude contrast in a.u.
    phase_shift : float
        The phase shift in degrees.
    Returns
    -------
    ctf : np.ndarray
        The contrast transfer function as a numpy array in the range [-1, 1]
    Notes
    -----
    Python implementation of the RELION compatible CTF model written by
    Takanori Nakane at MRC-LMB. https://3dem.github.io/relion/ctf.html
    """

    wavelength = electron_wavelength(1e3 * energy)

    phase_contrast = np.sqrt(1 - amplitude_contrast * amplitude_contrast)
    k1 = np.pi / 2 * (1e7 * spherical_abberation) * (wavelength**3)
    k2 = np.pi * wavelength

    gs = (
        np.linspace(-box_size / 2, box_size / 2, box_size)
        / box_size
        / pixel_size
    )
    xx, yy = np.meshgrid(gs, gs)

    angle = np.arctan2(yy, xx) - astigmatism_angle
    local_defocus = defocus + astigmatism * np.cos(2 * angle)

    s2 = (yy * yy) + (xx * xx)
    gamma = k1 * s2 * s2 - k2 * s2 * local_defocus - phase_shift
    ctf = -phase_contrast * np.sin(gamma) + amplitude_contrast * np.cos(gamma)

    return ctf.astype(np.float32)


def convolve_with_ctf(
    density: np.ndarray,
    ctf: np.ndarray,
    add_poisson_noise: bool = True,
    rng=np.random.default_rng()
) -> np.ndarray:
    """Convolve density with the CTF"""

    # perform the convolution
    img = np.multiply(fft2(-density), fftshift(ctf))
    img = np.abs(ifft2(img))

    # calculate the PSF (just for fun)
    # psf = np.abs( fftshift(fft2(ctf)) )
    # add some poisson noise
    # NOTE(arl): this is not correct, just a starting point
    if add_poisson_noise:
        img = rng.poisson(img * 0.1)

    return img


if __name__ == "__main__":
    CTF = contrast_transfer_function()
