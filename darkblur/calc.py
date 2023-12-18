import math
from typing import Tuple

import numpy as np
from scipy import constants as C


def normalise(im_data):
    min_values = np.min(im_data)
    max_values = np.max(im_data)

    normalized_stack = (im_data - min_values) / (max_values - min_values)
    return normalized_stack


def construct_fftfreq_grid_2d(image_shape: Tuple[int, int], rfft: bool,
                              spacing: float | Tuple[float, float] = 1) -> np.ndarray:
    """
    Constructs a 2D grid of Fourier frequencies with spacing.

    Parameters:
    image_shape (Tuple[int, int]): The shape of the image (height, width).
    rfft (bool): If True, construct the grid for a real FFT, otherwise for a complex FFT.
    spacing (float | Tuple[float, float]): Sample spacing in the grid.

    Returns:
    np.ndarray: A 2D grid of Fourier frequencies.
    """
    dh, dw = spacing if isinstance(spacing, (tuple, list)) else (spacing, spacing)
    h, w = image_shape
    freq_y = np.fft.fftfreq(h, d=dh)[:, None]  # Column vector of y frequencies
    freq_x = (np.fft.rfftfreq(w, d=dw) if rfft else np.fft.fftfreq(w, d=dw))[None, :]  # Row vector of x frequencies

    return np.stack(np.meshgrid(freq_y, freq_x, indexing='ij'), axis=-1)  # Shape: (h, w, 2)


def calculate_relativistic_electron_wavelength(energy: float):
    """Calculate the relativistic electron wavelength in SI units.

    For derivation see:
    1.  Kirkland, E. J. Advanced Computing in Electron Microscopy.
        (Springer International Publishing, 2020). doi:10.1007/978-3-030-33260-0.

    2.  https://en.wikipedia.org/wiki/Electron_diffraction#Relativistic_theory

    Parameters
    ----------
    energy: float
        acceleration potential in volts.

    Returns
    -------
    wavelength: float
        relativistic wavelength of the electron in meters.
    """
    h = C.Planck
    c = C.speed_of_light
    m0 = C.electron_mass
    e = C.elementary_charge
    V = energy
    eV = e * V

    numerator = h * c
    denominator = math.sqrt(eV * (2 * m0 * c ** 2 + eV))
    return numerator / denominator


def calculate_ctf(
        defocus: float | np.ndarray,
        astigmatism: float | np.ndarray,
        astigmatism_angle: float | np.ndarray,
        voltage: float,
        spherical_aberration: float,
        amplitude_contrast: float,
        b_factor: float | np.ndarray,
        phase_shift: float | np.ndarray,
        pixel_size: float,
        image_shape: Tuple[int, int],
        rfft: bool,
        fftshift: bool,
):
    # Unit conversions
    defocus = np.atleast_1d(np.asarray(defocus, dtype=float)) * 1e4
    astigmatism = np.atleast_1d(np.asarray(astigmatism, dtype=float)) * 1e4
    astigmatism_angle = np.atleast_1d(np.asarray(astigmatism_angle, dtype=float)) * (C.pi / 180)
    pixel_size = np.atleast_1d(np.asarray(pixel_size))
    voltage = np.atleast_1d(np.asarray(voltage, dtype=float)) * 1e3
    spherical_aberration = np.atleast_1d(np.asarray(spherical_aberration, dtype=float)) * 1e7

    defocus_u = defocus + astigmatism
    defocus_v = defocus - astigmatism
    _lambda = calculate_relativistic_electron_wavelength(voltage) * 1e10
    k1 = -C.pi * _lambda
    k2 = C.pi / 2 * spherical_aberration * _lambda ** 3
    k3 = np.deg2rad(phase_shift)
    k4 = -b_factor / 4
    k5 = np.arctan(amplitude_contrast / np.sqrt(1 - amplitude_contrast ** 2))

    # Construct 2D frequency grids and rescale
    fftfreq_grid = construct_fftfreq_grid_2d(image_shape=image_shape, rfft=rfft) / pixel_size.reshape(-1, 1, 1, 1)
    fftfreq_grid_squared = fftfreq_grid ** 2

    # Astigmatism calculations
    c = np.cos(astigmatism_angle)
    c2 = c ** 2
    s = np.sin(astigmatism_angle)
    s2 = s ** 2

    yy2, xx2 = np.moveaxis(fftfreq_grid_squared, -1, 0)
    xy = np.prod(fftfreq_grid, axis=-1)
    n4 = np.sum(fftfreq_grid_squared, axis=-1) ** 2

    Axx = c2 * defocus_u + s2 * defocus_v
    Axx_x2 = Axx[..., np.newaxis, np.newaxis] * xx2
    Axy = c * s * (defocus_u - defocus_v)
    Axy_xy = Axy[..., np.newaxis, np.newaxis] * xy
    Ayy = s2 * defocus_u + c2 * defocus_v
    Ayy_y2 = Ayy[..., np.newaxis, np.newaxis] * yy2

    # Calculate CTF
    ctf = -np.sin(k1 * (Axx_x2 + (2 * Axy_xy) + Ayy_y2) + k2 * n4 - k3 - k5)
    if k4 > 0:
        ctf *= np.exp(k4 * n4)
    if fftshift:
        ctf = np.fft.fftshift(ctf, axes=(-2, -1))
    return ctf
