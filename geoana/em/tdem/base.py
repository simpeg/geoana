from geoana.em.base import BaseEM
import numpy as np
from scipy.constants import mu_0
import properties


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
