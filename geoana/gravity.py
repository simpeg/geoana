import numpy as np
import properties

from scipy.constants import G


class PointMass:

    """
    Gravity due to a point mass
    """

    mass = properties.Float(
        "mass of the point particle (kg)", default=1.
    )

    location = properties.Float(
        "location of the point mass (m)", default=1
    )

    def gravitational_potential(self, xyz):
        """
        Gravitational potential for a point mass.  See Blakely, 1996
        equation 3.4

        .. math::

            U(P) = \\gamma \\frac{m}{r}

        """
        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)
        u_g = (G * self.mass) / r
        return u_g

    def gravitational_field(self, xyz):
        """
        Gravitational field for a point mass.  See Blakely, 1996
        equation 3.3

        .. math::

            \\mathbf{g} = \\nabla U(P)

        """
        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)
        g_vec = (G * self.mass * r_vec) / r
        return g_vec

    def gravitational_gradient(self, xyz):
        """
        Gravitational gradient for a point mass.

        .. math::

            \\Gamma = \\nabla \\mathbf{g}

        """
        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)
        gg_tens = (G * self.mass * np.eye) / r ** 3 + (3 * np.outer(r_vec, r_vec)) / r ** 5
        return gg_tens
