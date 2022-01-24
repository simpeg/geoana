from geoana.em.base import BaseEM
import numpy as np
from scipy.constants import mu_0, epsilon_0
import properties

###############################################################################
#                                                                             #
#                           Utility Functions                                 #
#                                                                             #
###############################################################################


def omega(frequency):
    """Compute angular frequency.

    For an input frequency :math:`f`, this function returns the
    corresponding angular frequency:

    .. math::

        \\omega = 2 \\pi f

    Parameters
    ----------
    frequency : float
        frequency (Hz)

    Returns:
    --------
    float
        Angular frequency (rad/s)

    """
    return 2*np.pi*frequency


def wavenumber(
    frequency, sigma, mu=mu_0, epsilon=epsilon_0, quasistatic=False
):
    """Compute wavenumber of an electromagnetic wave in a homogeneous isotropic medium.

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
    frequency : (float, numpy.ndarray)
        frequency or frequencies (Hz)
    sigma : float
        electrical conductivity (S/m)
    mu : float (optional)
        magnetic permeability (H/m). Default: :math:`\\mu_0 = 4\\pi \\times 10^{-7}` H/m
    epsilon : float (optional)
        dielectric permittivity (F/m). Default: :math:`\\epsilon_0 = 8.85 \\times 10^{-12}` F/m
    quasistatic : bool (optional)
        use the quasi-static assumption; i.e. ignore the dielectric term. Default: False

    Returns
    -------
    float, (n_frequencies) numpy.ndarray
        Wavenumber for all frequencies provided

    """
    w = omega(frequency)
    if quasistatic is True:
        return np.sqrt(-1j * w * mu * sigma)
    return np.sqrt(w**2 * mu * epsilon - 1j * w * mu * sigma)


def skin_depth(frequency, sigma, mu=mu_0):
    """Compute skin depth for an electromagnetic wave in a homogeneous isotropic medium.

    The skin depth propagation distance at which an EM planewave has decayed by a factor of :math:`1/e`.
    For a homogeneous medium with electrical conductivity :math:`\\sigma` and magnetic permeability
    :math:`\\mu`, the skin depth for a wave at frequency :math:`f` is given by:

    .. math::

        \\sqrt{\\frac{2}{\\omega \\sigma \\mu}}

    where

    .. math::
        \\omega = 2 \\pi f

    Parameters
    ----------
    frequency : (float, numpy.ndarray)
        frequency or frequencies (Hz)
    sigma : float
        electrical conductivity (S/m)
    mu : float (optional)
        magnetic permeability (H/m). Default: :math:`\\mu_0 = 4\\pi \\times 10^{-7}` H/m

    Returns
    -------
    float, (n_frequencies) numpy.ndarray
        Skin depth for all frequencies provided

    """
    w = omega(frequency)
    return np.sqrt(2./(w*sigma*mu))


def sigma_hat(frequency, sigma, epsilon=epsilon_0, quasistatic=False):
    """Compute the conductivity which includes electric displacement.

    Where :math:`\\sigma` is the electrical conductivity, :math:`\\varepsilon` is the
    dielectric permittivity and :math:`\\omega` is the angular frequency, this function
    returns:

    .. math::

        \\hat{\\sigma} = \\sigma + i \\omega \\varepsilon

    Parameters
    ----------
    frequency : (float, numpy.ndarray)
        frequency or frequencies (Hz)
    sigma : float
        electrical conductivity (S/m)
    epsilon : float (optional)
        dielectric permittivity (F/m). Default: :math:`\\epsilon_0 = 8.85 \\times 10^{-12}` F/m
    quasistatic : bool (optional)
        use the quasi-static assumption; i.e. ignore the dielectric term. Default: False

    Returns
    -------
    float, (n_frequencies) numpy.ndarray
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
    """
    Base frequency domain electromagnetic class
    """
    sigma = properties.Complex(
        "Electrical conductivity (S/m)",
        default=1.0,
        cast=True
    )

    frequency = properties.Float(
        "Source frequency (Hz)",
        default=1.,
        min=0.0
    )

    quasistatic = properties.Bool(
        "Use the quasi-static approximation and ignore displacement current?",
        default=False
    )

    @properties.validator('sigma')
    def _validate_real_part(self, change):
        if not np.real(change['value']) > 0:
            raise properties.ValidationError("The real part of sigma must be positive")

    @property
    def omega(self):
        """
        Angular frequency

        .. math::

            \\omega = 2\\pi f

        """
        return omega(self.frequency)

    @property
    def sigma_hat(self):
        """
        conductivity with displacement current contribution

        .. math::

            \\hat{\\sigma} = \\sigma + i \\omega \\varepsilon

        """
        sigma = sigma_hat(
            self.frequency, self.sigma, epsilon=self.epsilon,
            quasistatic=self.quasistatic
        )
        if np.all(np.imag(sigma) == 0):
            sigma = np.real(sigma)
        return sigma

    @property
    def wavenumber(self):
        """
        Wavenumber of an electromagnetic wave in a medium with constant
        physical properties

        .. math::

            k = \\sqrt{\\omega**2 \\mu \\varepsilon - i \\omega \\mu \\sigma}

        """
        return wavenumber(
            self.frequency, self.sigma, mu=self.mu, epsilon=self.epsilon,
            quasistatic=self.quasistatic
        )

    @property
    def skin_depth(self):
        """
        Distance at which an em wave has decayed by a factor of :math:`1/e` in
        a medium with constant physical properties

        .. math::

            \\sqrt{\\frac{2}{\\omega \\sigma \\mu}}

        """
        return skin_depth(self.frequency, self.sigma, mu=self.mu)
