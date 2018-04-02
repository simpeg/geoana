from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from scipy.constants import mu_0, epsilon_0
import numpy as np
import warnings
import properties

from .base import BaseElectricDipole, BaseMagneticDipole, BaseEM
from .. import spatial

__all__ = [
    'omega', 'wavenumber', 'skin_depth', 'sigma_hat',
    'ElectricDipoleWholeSpace', 'MagneticDipoleWholeSpace'
]


###############################################################################
#                                                                             #
#                           Utility Functions                                 #
#                                                                             #
###############################################################################

def omega(frequency):
    """
    Angular frequency

    .. math::

        \omega = 2 \pi f

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

        k = \sqrt{\omega**2 \mu \varepsilon - i \omega \mu \sigma}

    :param (float, numpy.ndarray) frequency: frequency (Hz)
    :param float sigma: electrical conductivity (S/m)
    :param float mu: magnetic permeability (H/m). Default: :math:`\mu_0 = 4\pi \times 10^{-7}` H/m
    :param float epsilon: dielectric permittivity (F/m). Default: :math:`\epsilon_0 = 8.85 \times 10^{-12}` F/m
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

        \sqrt{\\frac{2}{\omega \sigma \mu}}

    :param float frequency: frequency (Hz)
    :param float sigma: electrical conductivity (S/m)
    :param float mu: magnetic permeability (H/m). Default: :math:`\mu_0 = 4\pi \times 10^{-7}` H/m
    """
    w = omega(frequency)
    return np.sqrt(2./(w*sigma*mu))


def sigma_hat(frequency, sigma, epsilon=epsilon_0, quasistatic=False):
    """
    conductivity with displacement current contribution

    .. math::

        \hat{\sigma} = \sigma + i \omega \varepsilon

    :param (float, numpy.array) frequency: frequency (Hz)
    :param float sigma: electrical conductivity (S/m)
    :param float epsilon: dielectric permittivity. Default :math:`\varepsilon_0`
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
    frequency = properties.Float(
        "Source frequency (Hz)",
        default=1.,
        min=0.0
    )

    quasistatic = properties.Bool(
        "Use the quasi-static approximation and ignore displacement current?",
        default=False
    )

    @property
    def omega(self):
        """
        Angular frequency

        .. math::

            \omega = 2\pi f
        """
        return omega(self.frequency)

    @property
    def sigma_hat(self):
        """
        conductivity with displacement current contribution

        .. math::

            \hat{\sigma} = \sigma + i \omega \varepsilon

        """
        return sigma_hat(
            self.frequency, self.sigma, epsilon=self.epsilon,
            quasistatic=self.quasistatic
        )

    @property
    def wavenumber(self):
        """
        Wavenumber of an electromagnetic wave in a medium with constant
        physical properties

        .. math::

        k = \sqrt{\omega**2 \mu \varepsilon - i \omega \mu \sigma}
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

            \sqrt{\\frac{2}{\omega \sigma \mu}}
        """
        return skin_depth(self.frequency, self.sigma, mu=self.mu)


class ElectricDipoleWholeSpace(BaseElectricDipole, BaseFDEM):
    """
    Harmonic electric dipole in a whole space. The source is
    (c.f. Ward and Hohmann, 1988 page 173). The source current
    density for a dipole located at :math:`\mathbf{r}_s` with orientation
    :math:`\mathbf{\hat{u}}`

    .. math::

        \mathbf{J}(\mathbf{r}) = I ds \delta(\mathbf{r}
        - \mathbf{r}_s)\mathbf{\hat{u}}

    """
    def vector_potential(self, xyz):
        """
        Vector potential for an electric dipole in a wholespace

        .. math::

            \mathbf{A} = \frac{I ds}{4 \pi r} e^{-ikr}\mathbf{\hat{u}}

        """
        r = self.distance(xyz)
        a = (
            (self.current * self.length) / (4*np.pi*r) *
            np.exp(-i*self.wavenumber*r)
        )
        a = np.kron(np.ones(1, 3), np.atleast_2d(a).T)
        return self.dot_orientation(a)

    def electric_field(self, xyz):
        """
        Electric field from an electric dipole

        .. math::

            \mathbf{E} = \frac{1}{\hat{\sigma}} \nabla \nabla \cdot \mathbf{A}
            - i \omega \mu \mathbf{A}

        """
        dxyz = self.vector_distance(xyz)
        r = spatial.repeat_scalar(self.distance(xyz))
        kr = self.wavenumber * r
        ikr = 1j * kr

        front_term = (
            (self.current * self.length) / (4 * np.pi * self.sigma * r**3) *
            np.exp(-ikr)
        )
        symmetric_term = (
            spatial.repeat_scalar(self.dot_orientation(dxyz)) * dxyz *
            (-kr**2 + 3*ikr + 3) / r**2
        )
        oriented_term = (
            (kr**2 - ikr - 1) *
            np.kron(self.orientation, np.ones((dxyz.shape[0], 1)))
        )
        return front_term * (symmetric_term + oriented_term)

    def current_density(self, xyz):
        """
        Current density due to a harmonic electric dipole
        """
        return self.sigma * self.electric_field(xyz)

    def magnetic_field(self, xyz):
        """
        Magnetic field from an electric dipole

        .. math::

            \mathbf{H} = \nabla \times \mathbf{A}

        """
        dxyz = self.vector_distance(xyz)
        r = spatial.repeat_scalar(self.distance(xyz))
        kr = self.wavenumber * r
        ikr = 1j*kr

        front_term = (
            self.current * self.length / (4 * np.pi * r**2) * (ikr + 1) *
            np.exp(-ikr)
        )
        return -front_term * self.cross_orientation(dxyz) / r

    def magnetic_flux_density(self, xyz):
        """
        magnetic flux density from an electric dipole
        """
        return self.mu * self.magnetic_field(xyz)


class MagneticDipoleWholeSpace(BaseMagneticDipole, BaseFDEM):
    """
    Harmonic magnetic dipole in a whole space.
    """

    def vector_potential(self, xyz):
        """
        Vector potential for a magnetic dipole in a wholespace

        .. math::

            \mathbf{F} = \frac{i \omega \mu m}{4 \pi r} e^{-ikr}\mathbf{\hat{u}}

        """
        r = self.distance(xyz)
        f = (
            (1j * self.omega * self.mu * self.moment) / (4 * np.pi * r) *
            np.exp(-1j * self.wavenumber * r)
        )
        f = np.kron(np.ones(1, 3), np.atleast_2d(f).T)
        return self.dot_orientation(f)

    def electric_field(self, xyz):
        """
        Electric field from a magnetic dipole in a wholespace
        """
        dxyz = self.vector_distance(xyz)
        r = spatial.repeat_scalar(self.distance(xyz))
        kr = self.wavenumber*r
        ikr = 1j * kr

        front_term = (
            (1j * self.omega * self.mu * self.moment) / (4. * np.pi * r**2) *
            (ikr + 1) * np.exp(-ikr)
        )
        return front_term * self.cross_orientation(dxyz) / r

    def current_density(self, xyz):
        """
        Current density from a magnetic dipole in a wholespace
        """
        return self.sigma * self.electric_field(xyz)

    def magnetic_field(self, xyz):
        """
        Magnetic field due to a magnetic dipole in a wholespace
        """
        dxyz = self.vector_distance(xyz)
        r = spatial.repeat_scalar(self.distance(xyz))
        kr = self.wavenumber*r
        ikr = 1j*kr

        front_term = self.moment / (4. * np.pi * r**3) * np.exp(-ikr)
        symmetric_term = (
            spatial.repeat_scalar(self.dot_orientation(dxyz)) * dxyz *
            (-kr**2 + 3*ikr + 3) / r**2
        )
        oriented_term = (
            (kr**2 - ikr - 1) *
            np.kron(self.orientation, np.ones((dxyz.shape[0], 1)))
        )

        return front_term * (symmetric_term + oriented_term)

    def magnetic_flux_density(self, xyz):
        """
        Magnetic flux density due to a magnetic dipole in a wholespace
        """
        return self.mu * self.magnetic_field(xyz)
