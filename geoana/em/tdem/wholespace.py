from scipy.special import erf
import numpy as np
from geoana.em.tdem.base import BaseTDEM
from geoana.spatial import repeat_scalar
from geoana.em.base import BaseElectricDipole

###############################################################################
#                                                                             #
#                                  Classes                                    #
#                                                                             #
###############################################################################


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
        r = repeat_scalar(r)
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
                repeat_scalar(self.dot_orientation(dxyz)) * dxyz / r**2
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
        r = repeat_scalar(r)
        thetar = self.theta * r

        front = (
            self.current * self.length / (4 * np.pi * r**2) * (
                2 / np.sqrt(np.pi) * thetar * np.exp(-thetar**2) + erf(thetar)
            )
        )

        return - front * self.cross_orientation(xyz) / r

    def magnetic_field_time_deriv(self, xyz):
        """
        Time derivative of the magnetic field,
        :math:`\\frac{\partial \mathbf{h}}{\partial t}`
        """

        dxyz = self.vector_distance(xyz)
        r = self.distance(dxyz)
        r = repeat_scalar(r)

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
