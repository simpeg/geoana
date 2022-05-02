import numpy as np
from scipy.constants import mu_0
from scipy.special import iv, kv
from geoana.em.base import BaseMagneticDipole
from geoana.em.fdem.base import BaseFDEM


class MagneticDipoleHalfSpace(BaseFDEM, BaseMagneticDipole):
    r"""Class for a harmonic magnetic dipole in a wholespace.
    """

    def __init__(self, frequency, **kwargs):
        super().__init__(frequency=frequency, **kwargs)
        self._check_is_valid_location()

    def _check_is_valid_location(self):
        if self.location[2] != 0.0:
            raise ValueError("Source must be at the surface of the earth (i.e. z=0.0)")


    def magnetic_field(self, xy, field="secondary"):
        """Magnetic field due to a magnetic dipole over a half space

        The analytic expression is only valid for a source and receiver at the
        surface of the earth. For arbitrary source and receiver locations above
        the earth, use the layered solution.

        Parameters
        ----------
        xy : numpy.ndarray
            receiver locations of shape (n_locations, 2)
        field : ("secondary", "total")
            Flag for the type of field to return.

        Returns
        -------
        (n_freq, n_loc, 3) numpy.array of complex
            Magnetic field at all frequencies for the gridded
            locations provided. Output array is squeezed when n_freq and/or
            n_loc = 1.

        Examples
        --------
        Here, we define an z-oriented magnetic dipole at (0, 0, 0) and plot
        the secondary magnetic field at multiple frequencies at (5, 0, 0).

        >>> from geoana.em.fdem import MagneticDipoleHalfSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the electric current dipole.

        >>> frequency = np.logspace(2, 6, 41)
        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[0., 0., 1.]
        >>> moment = 1.
        >>> sigma = 1.0
        >>> simulation = MagneticDipoleHalfSpace(
        >>>     frequency, location=location, orientation=orientation,
        >>>     moment=moment, sigma=sigma
        >>> )

        Now we define the receiver location and plot the secondary field.

        >>> xyz = np.c_[5, 0, 0]
        >>> H = simulation.magnetic_field(xyz, field='secondary')

        Finally, we plot the real and imaginary components of the magnetic field.

        >>> fig = plt.figure(figsize=(6, 4))
        >>> ax1 = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        >>> ax1.semilogx(frequency, np.real(H[:, 2]), 'r', lw=2)
        >>> ax1.semilogx(frequency, np.imag(H[:, 2]), 'r--', lw=2)
        >>> ax1.set_xlabel('Frequency (Hz)')
        >>> ax1.set_ylabel('Secondary field (H/m)')
        >>> ax1.grid()
        >>> ax1.autoscale(tight=True)
        >>> ax1.legend(['real', 'imaginary'])


        """
        f = self.frequency
        n_freq = len(f)
        sig = self.sigma_hat
        w = 2*np.pi*f
        k = np.sqrt(-1j*w*mu_0*sig)[:, None]

        dxy = xy[:, :2] - self.location[:2]
        r = np.linalg.norm(dxy, axis=-1)
        n_loc = len(r)
        x = dxy[:, 0]
        y = dxy[:, 1]

        em_x = em_y = em_z = np.zeros((n_freq, n_loc), dtype=complex)
        src_x, src_y, src_z = self.orientation

        # tile such that (n_freq, n_loc)
        alpha = 1j * np.outer(k, r) / 2
        r = np.tile(r.reshape((1, n_loc)), (n_freq, 1))
        k = np.tile(k.reshape((n_freq, 1)), (1, n_loc))
        # alpha = 1j*k*r/2.
        IK1 = iv(1, alpha)*kv(1, alpha)
        IK2 = iv(2, alpha)*kv(2, alpha)

        if src_z != 0.0:
            # Z component of source
            em_z += src_z*2.0/(k**2*r**5)*(9-(9+9*1j*k*r-4*k**2*r**2-1j*k**3*r**3)*np.exp(-1j*k*r))
            Hr = (k**2/r)*(IK1 - IK2)
            angle = np.arctan2(y, x)
            angle = np.tile(angle.reshape((1, n_loc)), (n_freq, 1))
            em_x += src_z*np.cos(angle)*Hr
            em_y += src_z*np.sin(angle)*Hr

        x = np.tile(x.reshape((1, n_loc)), (n_freq, 1))
        y = np.tile(y.reshape((1, n_loc)), (n_freq, 1))

        if src_x != 0.0 or src_y != 0.0:
            # X component of source
            phi = 2/(k**2*r**4)*(3 + k**2*r**2 - (3 + 3j*k*r - k**2*r**2)*np.exp(-1j*k*r))
            dphi_dr = 2/(k**2*r**5)*(-2*k**2*r**2 - 12 + (-1j*k**3*r**3 - 5*k**2*r**2 + 12j*k*r + 12)*np.exp(-1j*k*r))
            if src_x != 0.0:
                em_x += src_x*(-1.0/r**3)*(y**2*phi + x**2*r*dphi_dr)
                em_y += src_x*(1.0/r**3)*x*y*(phi - r*dphi_dr)
                em_z -= src_x*(k**2*x/r**2)*(IK1 - IK2)

            # Y component of source
            if src_y != 0.0:
                em_x += src_y*(1.0/r**3)*x*y*(phi - r*dphi_dr)
                em_y += src_y*(-1.0/r**3)*(x**2*phi + y**2*r*dphi_dr)
                em_z -= src_y*(k**2*y/r**2)*(IK1 - IK2)

        if field == "secondary":
            # subtract out primary field from above
            mdotr = src_x*x + src_y*y# + m[2]*(z=0)

            em_x -= 3*x*mdotr/r**5 - src_x/r**3
            em_y -= 3*y*mdotr/r**5 - src_y/r**3
            em_z -= -src_z/r**3 # + 3*(z=0)*mdotr/r**5

        return self.moment/(4*np.pi)*np.stack((em_x, em_y, em_z), axis=-1).squeeze()
