from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from scipy.constants import mu_0
from scipy.special import erf
import numpy as np
import properties

from .base import BaseElectricDipole, BaseEM
from .. import spatial


###############################################################################
#                                                                             #
#                           Utility Functions                                 #
#                                                                             #
###############################################################################

def peak_time(z, sigma, mu=mu_0):
    """
    `Peak time <https://em.geosci.xyz/content/maxwell1_fundamentals/transient_planewaves_homogeneous/peaktime.html>`_:
    Time at which the maximum signal amplitude is observed at a particular
    location for a transient plane wave through a homogeneous medium.


    **Required**

    :param float z: distance from source (m)
    :param float sigma: electrical conductivity (S/m)

    **Optional**

    :param float mu: magnetic permeability (H/m). Default: :math:`\mu_0 = 4\pi \\times 10^{-7}` H/m

    """
    return (mu * sigma * z**2)/6.


def diffusion_distance(time, sigma, mu=mu_0):
    """
    `Diffusion distance <https://em.geosci.xyz/content/maxwell1_fundamentals/transient_planewaves_homogeneous/peakdistance.html>`_:
    Distance at which the signal amplitude is largest for a given time after
    shut off. Also referred to as the peak distance
    """
    return np.sqrt(2*time/(mu*sigma))


def theta(time, sigma, mu=mu_0):
    """
    Analog to wavenumber in the frequency domain. See Ward and Hohmann, 1988
    pages 174-175
    """
    return np.sqrt(mu*sigma/(4.*time))


###############################################################################
#                                                                             #
#                                  Classes                                    #
#                                                                             #
###############################################################################

class BaseTDEM(BaseEM):

    time = properties.Float(
        "time after shut-off at which we are evaluating the fields (s)",
        required=True,
        default=1e-4
    )

    def peak_time(self, z):
        return peak_time(z, self.sigma, self.mu)

    @property
    def diffusion_distance(self):
        return diffusion_distance(self.time, self.sigma, self.mu)

    @property
    def theta(self):
        return theta(self.time, self.sigma, self.mu)


class ElectricDipoleWholeSpace(BaseElectricDipole, BaseTDEM):
    """
    Harmonic electric dipole in a whole space. The source is
    (c.f. Ward and Hohmann, 1988 page 173). The source current
    density for a dipole located at :math:`\mathbf{r}_s` with orientation
    :math:`\mathbf{\hat{u}}`

    .. math::

        \mathbf{J}(\mathbf{r}) = I ds \delta(\mathbf{r}
        - \mathbf{r}_s)\mathbf{\hat{u}}

    """

    def electric_field(self, xyz):
        """
        Electric field from an electric dipole

        .. math::

            \mathbf{E} = \\frac{1}{\hat{\sigma}} \\nabla \\nabla \cdot \mathbf{A}
            - i \omega \mu \mathbf{A}

        """
        dxyz = self.vector_distance(xyz)
        r = self.distance(xyz)
        r = spatial.repeat_scalar(r)
        thetar = self.theta * r
        root_pi = np.sqrt(np.pi)

        front = (
            (self.current * self.length) / (4 * np.pi * self.sigma * r**3)
        )

        symmetric_term = (
            (
                - (
                    4/root_pi * thetar ** 3 + 6/root_pi * thetar
                ) * np.exp(-thetar**2) +
                3 * erf(thetar)
            ) * (
                spatial.repeat_scalar(self.dot_orientation(dxyz)) * dxyz / r**2
            )
        )

        oriented_term = (
            (
                4./root_pi * thetar**3 + 2./root_pi * thetar
            ) * np.exp(-thetar**2) -
            erf(thetar)
        ) * np.kron(self.orientation, np.ones((dxyz.shape[0], 1)))

        return front * (symmetric_term + oriented_term)

    def current_density(self, xyz):
        """
        Current density due to a harmonic electric dipole
        """
        return self.sigma * self.electric_field(xyz)

    def magnetic_field(self, xyz):
        """
        Magnetic field from an electric dipole
        """
        dxyz = self.vector_distance(xyz)
        r = self.distance(dxyz)
        r = spatial.repeat_scalar(r)
        thetar = self.theta * r

        front = (
            self.current * self.length / (4 * np.pi * r**2) * (
                2/root_pi * thetar * np.exp(-thetar**2) + erf(thetar)
            )
        )

        return - front * self.cross_orientation(xyz) / r

    def magnetic_field_time_deriv(self, xyz):
        """
        Time derivative of the magnetic field,
        :math:`\\frac{\partial \mathbf{h}}{\partial t}`
        """

        dxyz = self.vector_distance(xyz)
        r = self.distance(xyz)
        r = spatial.repeat_scalar(r)

        front = (
            self.current * self.length * self.theta**3 * r /
            (2 * np.sqrt(np.pi)**3 * self.time)
        )

        return - front * self.cross_orientation(xyz) / r

    def magnetic_flux_density(self, xyz):
        """
        Magnetic flux density from an electric dipole
        """

        return self.mu * self.magnetic_field(xyz)

    def magnetic_flux_density_time_deriv(self, xyz):
        """
        Time derivative of the magnetic flux density from an electric dipole
        """

        return self.mu * self.magnetic_field_time_deriv(xyz)



