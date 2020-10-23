import numpy as np
from geoana.em.fdem.base import BaseFDEM
from geoana.spatial import repeat_scalar
from geoana.em.base import BaseElectricDipole, BaseMagneticDipole


class ElectricDipoleWholeSpace(BaseElectricDipole, BaseFDEM):
    """
    Harmonic electric dipole in a whole space. The source is
    (c.f. Ward and Hohmann, 1988 page 173). The source current
    density for a dipole located at :math:`\\mathbf{r}_s` with orientation
    :math:`\\mathbf{\\hat{u}}`

    .. math::

        \\mathbf{J}(\\mathbf{r}) = I ds \\delta(\\mathbf{r}
        - \\mathbf{r}_s)\\mathbf{\\hat{u}}

    """
    def vector_potential(self, xyz):
        """
        Vector potential for an electric dipole in a wholespace

        .. math::

            \\mathbf{A} = \\frac{I ds}{4 \\pi r} e^{-ikr}\\mathbf{\\hat{u}}

        """
        r = self.distance(xyz)
        a = (
            (self.current * self.length) / (4*np.pi*r) *
            np.exp(-1j*self.wavenumber*r)
        )
        a = np.kron(np.ones(1, 3), np.atleast_2d(a).T)
        return self.dot_orientation(a)

    def electric_field(self, xyz):
        """
        Electric field from an electric dipole

        .. math::

            \\mathbf{E} = \\frac{1}{\\hat{\\sigma}} \\nabla \\nabla \\cdot
             \\mathbf{A}
            - i \\omega \\mu \\mathbf{A}

        """
        dxyz = self.vector_distance(xyz)
        r = repeat_scalar(self.distance(xyz))
        kr = self.wavenumber * r
        ikr = 1j * kr

        front_term = (
            (self.current * self.length) / (4 * np.pi * self.sigma * r**3) *
            np.exp(-ikr)
        )
        symmetric_term = (
            repeat_scalar(self.dot_orientation(dxyz)) * dxyz *
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

            \\mathbf{H} = \\nabla \\times \\mathbf{A}

        """
        dxyz = self.vector_distance(xyz)
        r = repeat_scalar(self.distance(xyz))
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

            \\mathbf{F} = \\frac{i \\omega \\mu m}{4 \\pi r} e^{-ikr}
            \\mathbf{\\hat{u}}

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
        r = repeat_scalar(self.distance(xyz))
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
        r = repeat_scalar(self.distance(xyz))
        kr = self.wavenumber*r
        ikr = 1j*kr

        front_term = self.moment / (4. * np.pi * r**3) * np.exp(-ikr)
        symmetric_term = (
            repeat_scalar(self.dot_orientation(dxyz)) * dxyz *
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
