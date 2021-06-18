import numpy as np

from geoana.em.base import BaseMagneticDipole
from geoana.em.tdem.base import BaseTDEM

from geoana.em.tdem.simple_functions import magnetic_field_vertical_magnetic_dipole, magnetic_field_time_deriv_magnetic_dipole


class VerticalMagneticDipoleHalfSpace(BaseMagneticDipole, BaseTDEM):
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
        xy : numpy.ndarray
            receiver locations of shape (n_locations, 2)
        """
        dxy = xy - self.location
        h = magnetic_field_vertical_magnetic_dipole(
            np.r_[self.time], dxy[:, :2], self.sigma, self.mu, self.moment
        )
        return h[0]  # because time was a 1 element array

    def magnetic_flux_density(self, xy):
        return self.mu * self.magnetic_field(xy)

    def magnetic_field_time_derivative(self, xy):

        dxy = xy - self.location
        dh_dt = magnetic_field_time_deriv_magnetic_dipole(
            np.r_[self.time], dxy[:, :2], self.sigma, self.mu, self.moment
        )
        return dh_dt[0]

    def magnetic_flux_time_derivative(self, xy):
        return self.mu * self.magnetic_field_time_derivative(xy)
