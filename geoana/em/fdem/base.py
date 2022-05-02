from geoana.em.base import BaseEM
import numpy as np
from scipy.constants import mu_0, epsilon_0

###############################################################################
#                                                                             #
#                           Utility Functions                                 #
#                                                                             #
###############################################################################


def omega(frequency):
    """Compute angular frequencies.

    For input frequencies :math:`f`, this function returns the
    corresponding angular frequencies:

    .. math::
        \\omega = 2 \\pi f

    Parameters
    ----------
    frequency : float, (n) numpy.ndarray
        frequency or frequencies (Hz)

    Returns
    -------
    float, (n) numpy.ndarray
        Angular frequency or frequencies in rad/s

    """
    return 2*np.pi*frequency


def wavenumber(
    frequency, sigma, mu=mu_0, epsilon=epsilon_0, quasistatic=False
):
    """Compute wavenumber for an electromagnetic wave in a homogeneous isotropic medium.

    Where :math:`f` is the frequency of the EM wave in Hertz, :math:`\\sigma` is the
    electrical conductivity in S/m, :math:`\\mu` is the magnetic permeability in
    H/m and :math:`\\varepsilon` is the dielectric permittivity in F/m, the
    wavenumber is given by:

    .. math::
        k = \\sqrt{\\omega^2 \\mu \\varepsilon - i \\omega \\mu \\sigma}

    where

    .. math::
        \\omega = 2 \\pi f

    Parameters
    ----------
    frequency : float, numpy.ndarray
        frequency or frequencies at which you are computing the wavenumber (Hz)
    sigma : float
        electrical conductivity (S/m).
    mu : float (optional)
        magnetic permeability (H/m). Default: :math:`\\mu_0 = 4\\pi \\times 10^{-7}` H/m.
    epsilon : float (optional)
        dielectric permittivity (F/m). Default: :math:`\\epsilon_0 = 8.85 \\times 10^{-12}` F/m.
    quasistatic : bool (optional)
        use the quasi-static assumption; i.e. ignore the dielectric term. Default: False

    Returns
    -------
    complex, (n_freq) numpy.ndarray of complex
        Wavenumber for all frequencies provided

    """
    w = omega(frequency)
    if quasistatic is True:
        return np.sqrt(-1j * w * mu * sigma)
    return np.sqrt(w**2 * mu * epsilon - 1j * w * mu * sigma)


def skin_depth(frequency, sigma, mu=mu_0, epsilon=epsilon_0, quasistatic=True):
    r"""Compute skin depth for an electromagnetic wave in a homogeneous isotropic medium.

    The skin depth is the propagation distance at which an EM planewave has decayed by a factor
    of :math:`1/e`. For a homogeneous medium with non-dispersive electrical conductivity
    :math:`\sigma`, magnetic permeability :math:`\mu` and dielectric permittivity
    :math:`\varepsilon`, the skin depth for a wave at frequency :math:`f` is given by:

    .. math::
        \delta = \frac{1}{\omega} \Bigg (\frac{\mu \varepsilon}{2} \bigg [ \bigg (
        1 + \frac{\sigma^2}{\omega^2 \varepsilon^2} \bigg )^{1/2} - 1 \bigg ] \Bigg )^{1/2}

    where :math:`\omega` is the angular frequency:

    .. math::
        \omega = 2 \pi f

    For the quasistatic approximation, dielectric permittivity is ignore and the
    skin depth simplifies to:

    .. math::
        \delta = \sqrt{\frac{2}{\omega \sigma \mu}}

    Parameters
    ----------
    frequency : float, numpy.ndarray
        frequency or frequencies (Hz)
    sigma : float
        electrical conductivity (S/m)
    mu : float (optional)
        magnetic permeability (H/m). Default: :math:`\mu_0 = 4\pi \times 10^{-7}` H/m
    epsilon : float (optional)
        dielectric permittivity (F/m). Default: :math:`\epsilon_0 = 8.85 \times 10^{-12}` F/m.
    quasistatic : bool (optional)
        If ``True``, the quasistatic approximation for the skin depth is computed.

    Returns
    -------
    float, (n_frequencies) numpy.ndarray
        Skin depth for all frequencies provided

    """
    w = omega(frequency)
    if quasistatic:
        return np.sqrt(2./(w*sigma*mu))
    return np.sqrt((mu*epsilon/2) * (np.sqrt(1 + sigma**2/(w*epsilon)**2) - 1)) / w


def sigma_hat(frequency, sigma, epsilon=epsilon_0, quasistatic=False):
    """Compute the electrical conductivity including electric displacement.

    Where :math:`\\sigma` is the electrical conductivity, :math:`\\varepsilon` is the
    dielectric permittivity and :math:`\\omega` is the angular frequency, this function
    returns:

    .. math::

        \\hat{\\sigma} = \\sigma + i \\omega \\varepsilon

    Parameters
    ----------
    frequency : float, numpy.ndarray
        frequency or frequencies (Hz)
    sigma : float
        electrical conductivity (S/m)
    epsilon : float (optional)
        dielectric permittivity (F/m). Default: :math:`\\epsilon_0 = 8.85 \\times 10^{-12}` F/m
    quasistatic : bool (optional)
        use the quasi-static assumption; i.e. ignore the dielectric term. Default: ``False``

    Returns
    -------
    complex, (n_frequencies) numpy.ndarray
        Conductivity with electric displacement for all frequencies provided

    """
    if quasistatic is True:
        return sigma
    return sigma + 1j*omega(frequency)*epsilon


###############################################################################
#                                                                             #
#                                  Classes                                    #
#                                                                             #
###############################################################################

class BaseFDEM(BaseEM):
    r"""
    Base frequency domain electromagnetic class

    The base FDEM class is contructed to define rudimentary properties and methods
    for frequency-domain electromagnetics.

    Parameters
    ----------
    frequency : int, float or (n_freq) numpy.ndarray
        Frequency or frequencies used for all computations in Hz.
    quasistatic : bool
        If ``True``, we assume the quasistatic approximation and dielectric permittivity
        is neglected in all computations. Default is ``False``.
    """

    def __init__(self, frequency, quasistatic=False, **kwargs):

        self.frequency = frequency
        self.quasistatic = quasistatic
        super().__init__(**kwargs)

    @property
    def frequency(self):
        """Frequency (Hz) used for all computations

        Returns
        -------
        numpy.ndarray
            Frequency (or frequencies) in Hz used for all computations
        """
        return self._frequency

    @frequency.setter
    def frequency(self, value):

        # Ensure float or numpy array of float
        try:
            value = np.asarray(value, dtype=float)
        except:
            raise TypeError(f"frequencies are not a valid type")
        value = np.atleast_1d(value)

        # Enforce positivity and dimensions
        if (value < 0.).any():
            raise ValueError("All frequencies must be greater than 0")
        if value.ndim > 1:
            raise TypeError(f"frequencies must be ('*') array")

        self._frequency = value

    @property
    def omega(self):
        """Angular frequency

        Where :math:`f` is the frequency assigned when instantiating the base EM class,
        this method returns the corresponding angular frequency:

        .. math::
            \\omega = 2\\pi f

        Returns
        -------
        float
            Angular frequency (rad/s)
        """
        return omega(self.frequency)

    @property
    def sigma_hat(self):
        """Conductivity including electric displacement term.

        Where :math:`\\sigma` is the electrical conductivity, :math:`\\varepsilon` is
        the dielectric permittivity and :math:`\\omega = 2\\pi f` is the angular
        frequency, this property returns:

        .. math::
            \\hat{\\sigma} = \\sigma + i \\omega \\varepsilon

        Returns
        -------
        complex
            Electrical conductivity including electric displacement. Returns the electrical
            conductivity :math:`\\sigma` if the property `quasistatic` is ``True``.

        """
        sigma = sigma_hat(
            self.frequency, self.sigma, epsilon=self.epsilon,
            quasistatic=self.quasistatic
        )
        if (np.imag(sigma) == 0).all():
            sigma = np.real(sigma)
        return sigma

    @property
    def wavenumber(self):
        r"""Wavenumber for an electromagnetic planewave in a homogenous isotropic medium.

        Where :math:`\sigma` is the electrical conductivity, :math:`\mu` is the magnetic
        permeability, :math:`\varepsilon` is the dielectric permittivity and
        :math:`\omega = 2\pi f` is the angular frequency, the wavenumber is:

        .. math::
            k = \sqrt{\omega^2 \mu \varepsilon - i \omega \mu \sigma}

        In the quasistatic regime, this expression simplifies to:

        .. math::
            k = \sqrt{-i \omega \mu \sigma}

        Returns
        -------
        complex
            Wavenumber for an electromagnetic planewave in a homogeneous isotropic medium.
            Returns the quasistatic approximation if the property `quasistatic` of the
            class instance is ``True``.

        """
        return wavenumber(
            self.frequency, self.sigma, mu=self.mu, epsilon=self.epsilon, quasistatic=self.quasistatic
        )

    @property
    def skin_depth(self):
        r"""Returns the skin depth for an electromagnetic wave in a homogeneous isotropic medium.

        The skin depth is the propagation distance at which an EM planewave has decayed by a factor
        of :math:`1/e`. For a homogeneous medium with non-dispersive electrical conductivity
        :math:`\sigma`, magnetic permeability :math:`\mu` and dielectric permittivity
        :math:`\varepsilon`, the skin depth for a wave at frequency :math:`f` is given by:

        .. math::
            \delta = \frac{1}{\omega} \Bigg (\frac{\mu \varepsilon}{2} \bigg [ \bigg (
            1 + \frac{\sigma^2}{\omega^2 \varepsilon^2} \bigg )^{1/2} - 1 \bigg ] \Bigg )^{1/2}

        where :math:`\omega` is the angular frequency:

        .. math::
            \omega = 2 \pi f

        For the quasistatic approximation, dielectric permittivity is ignore and the
        skin depth simplifies to:

        .. math::

            \delta = \sqrt{\frac{2}{\omega \sigma \mu}}

        Returns
        -------
        float
            Skin depth for the EM planewave. Returns the quasistatic approximation
            if the property `quasistatic` of the class instance is ``True``.

        """
        return skin_depth(
            self.frequency, self.sigma, mu=self.mu, epsilon=self.epsilon, quasistatic=self.quasistatic
        )
