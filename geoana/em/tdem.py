from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .base import BaseElectricDipole, BaseTDEM

from scipy.constants import mu_0, pi, epsilon_0
from scipy.special import erf
import numpy as np
import warnings


class ElectricDipole(BaseTDEM, BaseElectricDipole):

    """
    .. todo: write this independent of source orientation with dot products
    """
    @property
    def theta(self):
        return np.sqrt(self.mu*self.sigma / (4.*self.time))

    def electric_field(self, xyz):
        dxyz = self.vector_distance(xyz)
        x = dxyz[:, 0]
        y = dxyz[:, 1]
        z = dxyz[:, 2]

        root_pi = np.sqrt(np.pi)
        r = self.distance(xyz)
        r2 = r**2
        r3 = r**3.

        theta_r = self.theta * r
        e_n_theta_r_2 = np.exp(-theta_r**2)

        erf_thetat_r = erf(theta_r)

        current_term = (
            (self.current * self.length) /
            (4. * np.pi * self.sigma * r3)
        )

        symmetric_term = (
            - (
                (4./root_pi) * theta_r**3 + (6./root_pi) * theta_r
            ) * e_n_theta_r_2 +
            3 * erf_thetat_r
        )

        src_orientation_term = (
            - (
                (4./root_pi) * theta_r**3 + (2./root_pi) * theta_r
            ) * e_n_theta_r_2 +
            erf_thetat_r
        )

        if np.all(self.orientation == np.r_[1., 0., 0.]):
            ex = current_term * (
                symmetric_term * (x**2/r2) - src_orientation_term
            )
            ey = current_term * ( symmetric_term * (x*y)/r2 )
            ez = current_term * ( symmetric_term * (x*z)/r2 )

        elif np.all(self.orientation == np.r_[0., 1., 0.]):
            ey = current_term * (
                symmetric_term * (y**2/r2) - src_orientation_term
            )
            ez = current_term * ( symmetric_term * (y*z)/r2 )
            ex = current_term * ( symmetric_term * (y*x)/r2 )

        elif np.all(self.orientation == np.r_[0., 0., 1.]):
            ez = current_term * (
                symmetric_term * (z**2/r2) - src_orientation_term
            )
            ex = current_term * ( symmetric_term * (z*x)/r2 )
            ey = current_term * ( symmetric_term * (z*y)/r2 )
        else:
            raise NotImplementedError

        return np.c_[ex, ey, ez]

    def current_density(self, xyz):
        return self.sigma * self.e(xyz)





