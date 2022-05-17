import numpy as np
from scipy.constants import mu_0


def vertical_magnetic_field_horizontal_loop(
    frequencies, sigma=1.0, mu=mu_0, radius=1.0, current=1.0, turns=1, secondary=True
):
    """Vertical magnetic field at the center of a horizontal loop for each frequency

    Simple function to calculate the vertical magnetic field due to a harmonic
    horizontal loop. The anlytic form is only available at the center of the loop.

    Parameters
    ----------
    frequencies : float, or numpy.ndarray
        frequencies in Hz
    sigma : float, complex, or numpy.ndarray, optional
        electrical conductivity in S/m. Can provide a complex conductivity
        at each frequency if dispersive (i.e. induced polarization)
    mu : float, complex, or numpy.ndarray, optional
        magnetic permeability in H/m. Can provide a complex permeability
        at each frequency if dispersive (i.e. viscous remanent magnetization)
    radius : float, optional
        radius of the loop in meters
    current: float, optional
        transmitter current in A
    turns : int, optional
        number of turns for the loop source
    secondary : bool, optional
        if ``True``, the secondary field is returned. If ``False``, the total
        field is returned. Default is ``True``

    Returns
    -------
    complex, or numpy.ndarray
        The vertical magnetic field (H/m). If *secondary* is ``True``, only
        the secondary field is returned. If *secondary* is ``False``, the
        total field is returned.

    Notes
    -----
    The inputs values will be broadcasted together following normal numpy rules, and
    will support general shapes. Therefore every input, except for the `secondary` flag,
    can be arrays of the same shape.

    The analytic expression for the total magnetic field from equation 4.94 in Ward and
    Hohmann 1988:

    .. math::
        H_z = - \\frac{I}{k^2 a^3}\\left[3 - \\left(3 + 3 i k a - k^2 a^2\\right)e^{-i k a}\\right]

    Examples
    --------

    This example reproduces figure 4.7 from Ward and Hohmann

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from geoana.em.fdem import vertical_magnetic_field_horizontal_loop

    Define the frequency range,

    >>> frequencies = np.logspace(-1, 6, 200)
    >>> hz = vertical_magnetic_field_horizontal_loop(frequencies, sigma=1E-2, radius=50, secondary=False)

    Then plot the values

    >>> plt.loglog(frequencies, hz.real, c='C0', label='Real')
    >>> plt.loglog(frequencies, -hz.real, '--', c='C0')
    >>> plt.loglog(frequencies, hz.imag, c='C1', label='Imaginary')
    >>> plt.loglog(frequencies, -hz.imag, '--', c='C1')
    >>> plt.xlabel('frequency (Hz)')
    >>> plt.ylabel('H$_z$ (A/m)')
    >>> plt.legend()
    >>> plt.show()
    """
    w = 2*np.pi*frequencies
    k = np.sqrt(-1j * w * mu * sigma)
    a = radius

    Hz = (
        -current
        / (k ** 2 * a ** 3)
        * (3 - (3 + 3j * k * a - k ** 2 * a ** 2) * np.exp(-1j * k * a))
    )

    if secondary:
        Hzp = current / 2.0 / a
        Hz = Hz - Hzp
    return turns * Hz


def vertical_magnetic_flux_horizontal_loop(
    frequencies, sigma, mu=mu_0, radius=1.0, current=1.0, turns=1, secondary=True
):
    """Vertical magnetic flux density at the center of a horizontal loop for each frequency

    Simple function to calculate the vertical magnetic field due to a harmonic
    horizontal loop. The anlytic form is only available at the center of the loop.

    Parameters
    ----------
    frequencies : float, or numpy.ndarray
        frequencies in Hz
    sigma : float, complex, or numpy.ndarray, optional
        electrical conductivity in S/m. Can provide a complex conductivity
        at each frequency if dispersive (i.e. induced polarization)
    mu : float, complex, or numpy.ndarray, optional
        magnetic permeability in H/m. Can provide a complex permeability
        at each frequency if dispersive (i.e. viscous remanent magnetization)
    radius : float, optional
        radius of the loop in meters
    current: float, optional
        transmitter current in A
    turns : int, optional
        number of turns for the loop source
    secondary : bool, optional
        if ``True``, the secondary field is returned. If ``False``, the total
        field is returned. Default is ``True``

    Returns
    -------
    complex, or numpy.ndarray
        The vertical magnetic flux density in Teslas. If *secondary* is ``True``, only
        the secondary field is returned. If *secondary* is ``False``, the
        total field is returned.

    Notes
    -----
    The inputs values will be broadcasted together following normal numpy rules, and
    will support general shapes. Therefore every input, except for the `secondary` flag,
    can be arrays of the same shape.

    See Also
    --------
    vertical_magnetic_field_horizontal_loop
    """
    return mu * vertical_magnetic_field_horizontal_loop(
        frequencies, sigma, mu=mu, radius=radius, current=current, turns=turns, secondary=secondary
    )
