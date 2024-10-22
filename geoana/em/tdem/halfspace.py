import numpy as np

from geoana.em.base import BaseMagneticDipole
from geoana.em.tdem.base import BaseTDEM

from geoana.em.tdem.simple_functions import magnetic_field_vertical_magnetic_dipole, magnetic_field_time_deriv_magnetic_dipole
from geoana.utils import check_xyz_dim


class VerticalMagneticDipoleHalfSpace(BaseTDEM, BaseMagneticDipole):
    """Transient of a vertical magnetic dipole in a half space.

    Only valid for source and receivers at the surface. The surface is assumed
    to be at z=0.

    Waveform is assumed to be the step off.
    """

    def magnetic_field(self, xy):
        """Magnetic field due to a magnetic dipole over a half space

        The analytic expression is only valid for a source and receiver at the
        surface of the earth.

        Parameters
        ----------
        xy : (..., 2) numpy.ndarray
            receiver locations

        Returns
        -------
        (n_t, ..., 3) numpy.ndarray
            magnetic field for each xy location
        """
        try:
            xy = check_xyz_dim(xy, dim=2)
        except ValueError:
            xy = check_xyz_dim(xy, dim=3)[..., :2]
        dxy = xy - self.location[:2]
        h = magnetic_field_vertical_magnetic_dipole(
            self.time, dxy, self.sigma, self.mu, self.moment
        )
        return h

    def magnetic_flux_density(self, xy):
        """Magnetic flux due to a step off magnetic dipole over a half space

        The analytic expression is only valid for a source and receiver at the
        surface of the earth.

        Parameters
        ----------
        xy : (..., 2) numpy.ndarray
            receiver locations

        Returns
        -------
        (n_t, ..., 3) numpy.ndarray
            magnetic flux for each xy location
        """
        return self.mu * self.magnetic_field(xy)

    def magnetic_field_time_derivative(self, xy):
        """Magnetic flux time derivative due to a step off magnetic dipole over a half space

        The analytic expression is only valid for a source and receiver at the
        surface of the earth.

        Parameters
        ----------
        xy : (..., 2) numpy.ndarray
            receiver locations

        Returns
        -------
        (n_t, ..., 3) numpy.ndarray
            Magnetic flux time derivative for each xy location
        """
        try:
            xy = check_xyz_dim(xy, dim=2)
        except ValueError:
            xy = check_xyz_dim(xy, dim=3)[..., :2]
        dxy = xy - self.location[:2]
        dh_dt = magnetic_field_time_deriv_magnetic_dipole(
            self.time, dxy, self.sigma, self.mu, self.moment
        )
        return dh_dt

    def magnetic_flux_time_derivative(self, xy):
        """Magnetic flux due to a step off magnetic dipole over a half space

        The analytic expression is only valid for a source and receiver at the
        surface of the earth.

        Parameters
        ----------
        xy : (..., 2) numpy.ndarray
            receiver locations

        Returns
        -------
        (n_t, ..., 3) numpy.ndarray
            magnetic flux for each xy location
        """
        return self.mu * self.magnetic_field_time_derivative(xy)
