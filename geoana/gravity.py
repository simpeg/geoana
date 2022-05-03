from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from scipy.constants import G


class GravityPointMass:
    """
    Gravity due to a point mass
    """
    def gravitational_potential(self, m, r):
        """
        Gravitational potential for a point mass.  See Blakely, 1996
        equation 3.4

        .. math::

            U(P) = \\gamma \\frac{m}{r}

        """
        u_g = (G * m) / r
        return u_g

    def gravitational_field(self, m, r, xyz):
        """
        Gravitational field for a point mass.  See Blakely, 1996
        equation 3.3

        .. math::

            \\mathbf{g} = \\nabla U(P)

        """
        r_vec = self.distance(xyz)
        g_vec = (G * m * r_vec) / r
        return g_vec

    def gravitational_gradient(self, m, r, xyz):
        """
        Gravitational gradient for a point mass.

        .. math::

            in progress

        """
        r_vec = self.distance(xyz)
        gg_tens = (G * m * np.eye) / r ** 3 + (3 * np.outer(r_vec, r_vec)) / r ** 5
        return gg_tens
