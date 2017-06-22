from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import properties
from scipy.constants import mu_0, pi, epsilon_0

from .. import spatial


###############################################################################
#                                                                             #
#                                 Functions                                   #
#                                                                             #
###############################################################################

def omega(frequency):
    """
    Angular frequency

    :param frequency float: frequency (Hz)
    """
    return 2*np.pi*frequency


def wave_number(frequency, sigma, mu=mu_0, epsilon=epsilon_0):
    """
    Wavenumber of an electromagnetic wave in a medium with constant physical
    properties

    :param frequency float: frequency (Hz)
    :param sigma float: electrical conductivity (S/m)
    :param mu float: magnetic permeability (H/m). Default: :math:`\mu_0 = 4\pi \times 10^{-7}` H/m
    :param epsilon float: dielectric permittivity (F/m). Default: :math:`\epsilon_0 = 8.85 \times 10^{-12}` F/m
    """
    omega = omega(frequency)
    return np.sqrt(omega**2. * mu * epsilon - 1j * omega * mu * sigma)


def skin_depth(frequency, sigma, mu=mu_0):
    """
    Distance at which an em wave has decayed by a factor of 1/e in a medium
    with constant physical properties

    :param frequency float: frequency (Hz)
    :param sigma float: electrical conductivity (S/m)
    :param mu float: magnetic permeability (H/m). Default: :math:`\mu_0 = 4\pi \times 10^{-7}` H/m
    """
    omega = omega(frequency)
    return np.sqrt(2./(omega*sigma*mu))


def peak_time(z, sigma, mu=mu_0):
    """
    Time at which the maximum signal amplitude is observed at a particular
    location for a transient plane wave through a homogeneous medium.

    See: http://em.geosci.xyz/content/maxwell1_fundamentals/plane_waves_in_homogeneous_media/time/analytic_solution.html

    :param z float: distance from source (m)
    :param sigma float: electrical conductivity (S/m)
    :param mu float: magnetic permeability (H/m). Default: :math:`\mu_0 = 4\pi \times 10^{-7}` H/m
    """
    return (mu * sigma * z**2)/6.


def diffusion_distance(time, sigma, mu=mu_0):
    """
    Distance at which the signal amplitude is largest for a given time after
    shut off. Also referred to as the peak distance

    See: http://em.geosci.xyz/content/maxwell1_fundamentals/plane_waves_in_homogeneous_media/time/analytic_solution.html


    """
    return np.sqrt(2*time/(mu*sigma))


###############################################################################
#                                                                             #
#                              Base Classes                                   #
#                                                                             #
###############################################################################


class BaseEM(properties.HasProperties):
    """
    Base class for electromanetics. Contains physical properties that are
    relevant to all problems that use Maxwell's equations
    """

    mu = properties.Float(
        "Magnetic permeability (H/m)",
        default=mu_0,
        min=0.0
    )

    sigma = properties.Float(
        "Electrical conductivity (S/m)",
        default=1.0,
        min=0.0
    )

    epsilon = properties.Float(
        "Permitivity value (F/m)",
        default=epsilon_0,
        min=0.0
    )


class BaseDipole(BaseEM):
    """
    Base class for dipoles.
    """

    orientation = properties.Vector3(
        "orientation of dipole",
        default="X",
        length=1.0
    )

    location = properties.Vector3(
        "location of the electric dipole source",
        default="ZERO"
    )

    def vector_distance(self, xyz):
        return spatial.vector_distance(xyz, self.location)

    def distance(self, xyz):
        return spatial.distance(xyz, self.location)


class BaseFDEM(BaseEM):

    frequency = properties.Float(
        "Source frequency (Hz)",
        default=1e2,
        min=0.0
    )

    @property
    def omega(self):
        return omega(self.frequency)

    @property
    def sigma_hat(self):
        return self.sigma + 1j*self.omega*self.epsilon

    @property
    def wave_number(self):
        return wave_number(self.frequency, self.sigma, self.mu)

    @property
    def skin_depth(self):
        return skin_depth(self.frequency, self.sigma, self.mu)


class BaseTDEM(BaseEM):

    time = properties.Float(
        "time after shut-off at which we are evaluating the fields (s)",

        required=True
    )

    def peak_time(self, z):
        return peak_time(z, self.sigma, self.mu)

    @property
    def diffusion_distance(self):
        return diffusion_distance(self.time, self.sigma, self.mu)


class BaseElectricDipole(BaseDipole):
    """
    Base class for electric current dipoles
    """

    length = properties.Float(
        "length of the dipole (m)",
        default=1.0,
        min=0.0
    )

    current = properties.Float(
        "size of the injected current (A)",
        default=1.0,
        min=0.0
    )


class BaseMagneticDipole(BaseDipole):
    """
    Base class for magnetic dipoles
    """

    moment = properties.Float(
        "moment of the dipole (Am^2)",
        default=1.0,
        min=0.0
    )
