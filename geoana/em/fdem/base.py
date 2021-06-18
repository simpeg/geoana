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
    """
    Angular frequency

    .. math::

        \\omega = 2 \\pi f

    **Required**
    :param frequency float: frequency (Hz)

    """
    return 2*np.pi*frequency


def wavenumber(
    frequency, sigma, mu=mu_0, epsilon=epsilon_0, quasistatic=False
):
    """
    Wavenumber of an electromagnetic wave in a medium with constant physical
    properties

    .. math::

        k = \\sqrt{\\omega^2 \\mu \\varepsilon - i \\omega \\mu \\sigma}


    **Required**

    :param (float, numpy.ndarray) frequency: frequency (Hz)
    :param float sigma: electrical conductivity (S/m)

    **Optional**

    :param float mu: magnetic permeability (H/m). Default: :math:`\\mu_0 = 4\\pi \\times 10^{-7}` H/m
    :param float epsilon: dielectric permittivity (F/m). Default: :math:`\\epsilon_0 = 8.85 \\times 10^{-12}` F/m
    :param bool quasistatic: use the quasi-static assumption? Default: False

    """
    w = omega(frequency)
    if quasistatic is True:
        return np.sqrt(-1j * w * mu * sigma)
    return np.sqrt(w**2 * mu * epsilon - 1j * w * mu * sigma)


def skin_depth(frequency, sigma, mu=mu_0):
    """
    Distance at which an em wave has decayed by a factor of :math:`1/e` in a
    medium with constant physical properties

    .. math::

        \\sqrt{\\frac{2}{\\omega \\sigma \\mu}}

    **Required**

    :param float frequency: frequency (Hz)
    :param float sigma: electrical conductivity (S/m)

    **Optional**
    :param float mu: magnetic permeability (H/m). Default: :math:`\mu_0 = 4\pi \\times 10^{-7}` H/m

    """
    w = omega(frequency)
    return np.sqrt(2./(w*sigma*mu))


def sigma_hat(frequency, sigma, epsilon=epsilon_0, quasistatic=False):
    """
    conductivity with displacement current contribution

    .. math::

        \hat{\sigma} = \sigma + i \omega \\varepsilon

    **Required**

    :param (float, numpy.array) frequency: frequency (Hz)
    :param float sigma: electrical conductivity (S/m)

    **Optional**

    :param float epsilon: dielectric permittivity. Default :math:`\\varepsilon_0`
    :param bool quasistatic: use the quasi-static assumption? Default: False

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
