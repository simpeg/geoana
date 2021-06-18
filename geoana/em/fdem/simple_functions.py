import numpy as np
from scipy.constants import mu_0


def vertical_magnetic_field_horizontal_loop(
    frequencies, sigma, mu=mu_0, radius=1.0, current=1.0, turns=1, secondary=True
):
    """Vertical magnetic field at the center of a horizontal loop for each frequency

    Returns
    -------
    hz : np.ndarray
    """
    w = 2*np.pi*frequencies
    k = np.sqrt(-1j * w * mu * sigma)
    a = radius

    Hz = (
        -current
        / (k ** 2 * a ** 3)
        * (3 - (3 + 3 * 1j * k * a - k ** 2 * a ** 2) * np.exp(-1j * k * a))
    )

    if secondary:
        Hzp = current / 2.0 / a
        Hz = Hz - Hzp
    return turns * Hz


def vertical_magnetic_flux_horizontal_loop(
    frequencies, sigma, mu=mu_0, radius=1.0, current=1.0, turns=1, secondary=True
):
    """Vertical magnetic flux density at the center of a horizontal loop for each frequency

    Returns
    -------
    hz : np.ndarray
    """
    return mu_0 * vertical_magnetic_field_central_loop(
        frequencies, sigma, mu=mu, radius=radius, current=current, turns=turns, secondary=secondary
    )
