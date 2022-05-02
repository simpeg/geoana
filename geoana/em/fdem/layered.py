import numpy as np
from geoana.em.base import BaseMagneticDipole
from geoana.em.fdem.base import BaseFDEM, sigma_hat
from scipy.constants import mu_0, epsilon_0
from empymod.utils import check_hankel
from empymod.transform import get_dlf_points
from geoana.kernels.tranverse_electric_reflections import rTE_forward


class MagneticDipoleLayeredHalfSpace(BaseFDEM, BaseMagneticDipole):
    """Simulation class for a harmonic magnetic dipole over a layered halfspace.

    This class is used to simulate the fields produced by a harmonic magnetic dipole
    source over a layered halfspace.

    Parameters
    ----------
    thickness : None or ('*') numpy.ndarray
        Layer thicknesses (m) from the top layer downwards. If ``None`` is entered, you are
        defining a homogeneous halfspace. The bottom layer extends to infinity, thus
        n_layer = n_thickness + 1.
    sigma : (n_layer) np.ndarray or (n_layer, n_frequency) np.ndarray
        Electrical conductivities for all layers (and at all frequencies). For non-dispersive
        conductivity (no chargeability) or for an instance of **MagneticDipoleLayeredHalfSpace**
        at a single frequency, *sigma* is assigned with a (n_layer) np.ndarray.
        For dispersive conductivity and multiple frequencies, *sigma* is assigned with
        a (n_layer, n_frequency) np.ndarray.
    mu : (n_layer) np.ndarray or (n_layer, n_frequency) np.ndarray
        Magnetic permeability for all layers (and at all frequencies). For non-dispersive
        permeability (no viscous remanent magnetization) or for an instance of
        **MagneticDipoleLayeredHalfSpace** at a single frequency, *mu* is assigned with a
        (n_layer) np.ndarray. For dispersive permeability and multiple frequencies, *mu*
        is assigned with a (n_layer, n_frequency) np.ndarray.
    epsilon : (n_layer) np.ndarray or (n_layer, n_frequency) np.ndarray
        Dielectric permittivity for all layers (and at all frequencies). Only applicable when
        *quasistatic* == ``True``. For non-dispersive permittivity or for an instance of
        **MagneticDipoleLayeredHalfSpace** at a single frequency, *epsilon* is assigned with
        a (n_layer) np.ndarray. For dispersive permittivity and multiple frequencies,
        *epsilon* is assigned with a (n_layer, n_frequency) np.ndarray.

    """

    def __init__(self, frequency, thickness, **kwargs):

        self.thickness = thickness
        super().__init__(frequency=frequency, **kwargs)
        self._check_is_valid_location()


    def _check_is_valid_location(self):
        if self.location[2] < 0.0:
            raise ValueError("Source must be above the surface of the earth (i.e. z >= 0.0)")

    @property
    def frequency(self):
        """Frequency (Hz) used for all computations

        Returns
        -------
        numpy.ndarray
            Frequency (or frequencies) in Hz used for all computations
        """
        return self._frequency

    @frequency.setter
    def frequency(self, value):

        # Ensure float or numpy array of float
        try:
            value = np.atleast_1d(value).astype(float)
        except:
            raise TypeError(f"frequencies are not a valid type")

        # Enforce positivity and dimensions
        if (value < 0.).any():
            raise ValueError("All frequencies must be greater than 0")
        if value.ndim > 1:
            raise TypeError(f"frequencies must be ('*') array")

        self._frequency = value

    @property
    def thickness(self):
        """Thicknesses (m) for all layers from top to bottom

        Returns
        -------
        numpy.ndarray
            Thicknesses (m) for all layers from top to bottom
        """
        return self._thickness

    @thickness.setter
    def thickness(self, value):

        # Ensure float or numpy array of float
        try:
            if value is None:
                value = []
            else:
                value = np.atleast_1d(value).astype(float)
        except:
            raise TypeError(f"thickness are not a valid type")

        # Enforce positivity and dimensions
        if (value < 0.).any():
            raise ValueError("Thicknesses must be greater than 0")
        if value.ndim > 1:
            raise TypeError(f"Thicknesses must be ('*') array")

        self._thickness = value


    @property
    def sigma(self):
        """Electrical conductivity for all layers (and frequencies)

        Returns
        -------
        numpy.ndarray (n_layer) or (n_layer, n_frequency)
            Electrical conductivity  for all layers (and frequencies)
        """
        return self._sigma

    @sigma.setter
    def sigma(self, value):

        n_layer = len(self.thickness) + 1
        n_frequency = len(self.frequency)

        # Ensure numpy array of complex
        try:
            if isinstance(value, (list, tuple, np.ndarray)):
                value = np.atleast_1d(value).astype(complex)
            else:
                value = complex(value) * np.ones(n_layer)
        except:
            raise TypeError(f"sigma array is not a valid type")

        # Error if values are non-physical
        if (value.imag < 0.).any():
            raise ValueError("Imaginary components must be >= 0.0")
        if (value.real <= 0.0).any():
            raise ValueError("Real components must be > 0.0")

        # Enforce dimensions
        if (value.ndim == 1) & (len(value) != n_layer):
            raise TypeError(f"sigma must be (n_layer) or (n_layer, n_frequency) np.ndarray")
        elif (value.ndim == 2) & (np.shape(value) != (n_layer, n_frequency)):
            raise TypeError(f"sigma must be (n_layer) or (n_layer, n_frequency) np.ndarray")
        elif value.ndim > 2:
            raise TypeError(f"sigma must be (n_layer) or (n_layer, n_frequency) np.ndarray")

        self._sigma = value

    @property
    def mu(self):
        """Magnetic permeability for all layers (and frequencies)

        Returns
        -------
        numpy.ndarray (n_layer) or (n_layer, n_frequency)
            Magnetic permeability for all layers (and frequencies)
        """
        return self._mu

    @mu.setter
    def mu(self, value):

        n_layer = len(self.thickness) + 1
        n_frequency = len(self.frequency)

        # Ensure float or numpy array of complex
        try:
            if isinstance(value, (list, tuple, np.ndarray)):
                value = np.atleast_1d(value).astype(complex)
            else:
                value = complex(value) * np.ones(n_layer)
        except:
            raise TypeError(f"mu array is not a valid type")

        # Error if values are non-physical
        if (value.imag < 0.).any():
            raise ValueError("Imaginary components must be >= 0")
        if (value.real < mu_0).any():
            raise ValueError("Real components must be >= mu_0")

        # Enforce dimensions
        if (value.ndim == 1) & (len(value) != n_layer):
            raise TypeError(f"mu must be (n_layer) or (n_layer, n_frequency) np.ndarray")
        elif (value.ndim == 2) & (np.shape(value) != (n_layer, n_frequency)):
            raise TypeError(f"mu must be (n_layer) or (n_layer, n_frequency) np.ndarray")
        elif value.ndim > 2:
            raise TypeError(f"mu must be (n_layer) or (n_layer, n_frequency) np.ndarray")

        self._mu = value


    @property
    def epsilon(self):
        """Dielectric permittivity for all layers (and frequencies)

        Returns
        -------
        numpy.ndarray (n_layer) or (n_layer, n_frequency)
            Magnetic permeability for all layers (and frequencies)
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):

        n_layer = len(self.thickness) + 1
        n_frequency = len(self.frequency)

        # Ensure float or numpy array of complex
        try:
            if isinstance(value, (list, tuple, np.ndarray)):
                value = np.atleast_1d(value).astype(complex)
            else:
                value = complex(value) * np.ones(n_layer)
        except:
            raise TypeError(f"epsilon array is not a valid type")

        # Error if values are non-physical
        if (value.imag < 0.).any():
            raise ValueError("Imaginary components must be >= 0.0")
        if (value.real < epsilon_0).any():
            raise ValueError("Real components must be >= epsilon_0")

        # Enforce dimensions
        if (value.ndim == 1) & (len(value) != n_layer):
            raise TypeError(f"epsilon must be (n_layer) or (n_layer, n_frequency) np.ndarray")
        elif (value.ndim == 2) & (np.shape(value) != (n_layer, n_frequency)):
            raise TypeError(f"epsilon must be (n_layer) or (n_layer, n_frequency) np.ndarray")
        elif value.ndim > 2:
            raise TypeError(f"epsilon must be (n_layer) or (n_layer, n_frequency) np.ndarray")

        self._epsilon = value

    # def _get_valid_properties(self):
    #     thick = self.thickness
    #     n_layer = len(thick)+1
    #     sigma = self.sigma
    #     epsilon = self.epsilon
    #     mu = self.mu
    #     if n_layer != 1:
    #         sigma = self.sigma
    #         if len(sigma) == 1:
    #             sigma = np.ones(n_layer)*sigma
    #         epsilon = self.epsilon
    #         if len(epsilon) == 1:
    #             epsilon = np.ones(n_layer)*epsilon
    #         mu = self.mu
    #         if len(mu) == 1:
    #             mu = np.ones(n_layer)*mu
    #     return thick, sigma, epsilon, mu

    def _get_valid_properties_array(self):

        n_layer = len(self.thickness)+1
        n_frequency = len(self.frequency)

        sigma = self.sigma
        if sigma.ndim == 1:
            sigma = np.tile(sigma.reshape((n_layer, 1)), (1, n_frequency))
        mu = self.mu
        if mu.ndim == 1:
            mu = np.tile(mu.reshape((n_layer, 1)), (1, n_frequency))
        epsilon = self.epsilon
        if epsilon.ndim == 1:
            epsilon = np.tile(epsilon.reshape((n_layer, 1)), (1, n_frequency))

        return self.thickness, sigma, epsilon, mu

    @property
    def sigma_hat(self):
        _, sigma, epsilon, _ = self._get_valid_properties()
        return sigma_hat(
            self.frequency[:, None], sigma, epsilon,
            quasistatic=self.quasistatic
        ).T

    @property
    def wavenumber(self):
        raise NotImplementedError()

    @property
    def skin_depth(self):
        raise NotImplementedError()

    def magnetic_field(self, xyz, field="secondary"):
        r"""
        Compute the magnetic field produced by a magnetic dipole over a layered halfspace.

        Parameters
        ----------
        xyz : numpy.ndarray
            receiver locations of shape (n_locations, 3).
            The z component cannot be below the surface (z >= 0.0).
        field : ("secondary", "total")
            Flag for the type of field to return.

        Returns
        -------
        (n_freq, n_loc, 3) numpy.array of complex
            Magnetic field at all frequencies for the gridded
            locations provided. Output array is squeezed when n_freq and/or
            n_loc = 1.

        Notes
        -----
        We compute the magnetic using the Hankel transform solutions from Ward and Hohmann.
        For the vertical component of the magnetic dipole, the vertical and horizontal fields
        are given by equations 4.45 and 4.46:

        .. math::
            H_\rho = \frac{m_z}{4\pi} \int_0^\infty \bigg [ e^{-u_0 (z - h)} - r_{te} e^{u_0 (z + h)} \bigg ] \lambda^2 J_1 (\lambda \rho) \, d\lambda

        .. math::
            H_z = \frac{m_z}{4\pi} \int_0^\infty \bigg [ e^{-u_0 (z - h)} + r_{te} e^{u_0 (z + h)} \bigg ] \frac{\lambda^3}{u_0} J_0 (\lambda \rho) \, d\lambda

        For the horizontal component of the magnetic dipole, we compute the contribution by adapting
        Ward and Hohmann equations 4.119-4.121; which is for an x-oriented magnetic dipole:

        .. math::
            H_x = & -\frac{m_x}{4\pi} \bigg ( \frac{1}{\rho} - \frac{2x^2}{\rho^3} \bigg ) \int_0^\infty \bigg [ e^{-u_0 (z - h)} - r_{te} e^{u_0 (z + h)} \bigg ] \lambda J_1 (\lambda \rho) \, d\lambda \\
            & -\frac{m_x}{4\pi} \frac{x^2}{\rho^2} \int_0^\infty \bigg [ e^{-u_0 (z - h)} - r_{te} e^{u_0 (z + h)} \bigg ] \lambda^2 J_0 (\lambda \rho) \, d\lambda

        .. math::
            H_y = & \frac{m_x}{2\pi} \frac{xy}{\rho^3} \int_0^\infty \bigg [ e^{-u_0 (z - h)} - r_{te} e^{u_0 (z + h)} \bigg ] \lambda J_1 (\lambda \rho) \, d\lambda \\
            & -\frac{m_x}{4\pi} \frac{xy}{\rho^2} \int_0^\infty \bigg [ e^{-u_0 (z - h)} - r_{te} e^{u_0 (z + h)} \bigg ] \lambda^2 J_0 (\lambda \rho) \, d\lambda

        .. math::
            H_z = \frac{m_x}{4\pi} \frac{x}{\rho}  \int_0^\infty \bigg [ e^{-u_0 (z - h)} + r_{te} e^{u_0 (z + h)} \bigg ] \lambda^2 J_1 (\lambda \rho) \, d\lambda

        Examples
        --------
        Here, we define an z-oriented magnetic dipole at (0, 0, 0) and plot
        the secondary magnetic field at multiple frequencies at (5, 0, 0).
        We compare the secondary fields for a halfspace and for a layered Earth.

        >>> from geoana.em.fdem import (
        >>>     MagneticDipoleHalfSpace, MagneticDipoleLayeredHalfSpace
        >>> )
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the electric current dipole.

        >>> frequency = np.logspace(2, 6, 41)
        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[0., 0., 1.]
        >>> moment = 1.

        We now define the halfspace simulation.

        >>> sigma = 1.0
        >>> simulation_halfspace = MagneticDipoleHalfSpace(
        >>>     frequency, location=location, orientation=orientation,
        >>>     moment=moment, sigma=sigma
        >>> )

        And the layered Earth simulation.

        >>> sigma_top = 0.1
        >>> sigma_middle = 1.0
        >>> sigma_bottom = 0.01
        >>> thickness = np.r_[5., 2.]
        >>> sigma_layers = np.r_[sigma_top, sigma_middle, sigma_bottom]
        >>> simulation_layered = MagneticDipoleLayeredHalfSpace(
        >>>     frequency, thickness, location=location, orientation=orientation,
        >>>     moment=moment, sigma=sigma_layers
        >>> )

        Now we define the receiver location and plot the seconary field.

        >>> xyz = np.c_[5, 0, 0]
        >>> H_halfspace = simulation_halfspace.magnetic_field(xyz, field='secondary')
        >>> H_layered = simulation_layered.magnetic_field(xyz, field='secondary')

        Finally, we plot the real and imaginary components of the magnetic field.

        >>> fig = plt.figure(figsize=(6, 4))
        >>> ax1 = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        >>> ax1.semilogx(frequency, np.real(H_halfspace[:, 2]), 'r', lw=2)
        >>> ax1.semilogx(frequency, np.imag(H_halfspace[:, 2]), 'r--', lw=2)
        >>> ax1.semilogx(frequency, np.real(H_layered[:, 2]), 'b', lw=2)
        >>> ax1.semilogx(frequency, np.imag(H_layered[:, 2]), 'b--', lw=2)
        >>> ax1.set_xlabel('Frequency (Hz)')
        >>> ax1.set_ylabel('Secondary field (H/m)')
        >>> ax1.grid()
        >>> ax1.autoscale(tight=True)
        >>> ax1.legend(['Halfspace: real', 'Halfspace: imag', 'Layered: real', 'Layered: imag'])

        """

        if (xyz[:, 2] < 0.0).any():
            raise ValueError("Cannot compute fields below the surface")
        h = self.location[2]
        dxyz = xyz - self.location
        offsets = np.linalg.norm(dxyz[:, :-1], axis=-1)

        # Comput transform operations
        # -1 gives lagged convolution in dlf
        ht, htarg = check_hankel('dlf', {'dlf': 'key_101_2009', 'pts_per_dec': 0}, 1)
        fhtfilt = htarg['dlf']
        pts_per_dec = htarg['pts_per_dec']

        f = self.frequency
        n_frequency = len(f)

        lambd, int_points = get_dlf_points(fhtfilt, offsets, pts_per_dec)

        thick = self.thickness
        n_layer = len(thick) + 1

        thick, sigma, epsilon, mu = self._get_valid_properties_array()

        # sigh = sigma_hat(
        #     self.frequency[:, None], sigma, epsilon,
        #     quasistatic=self.quasistatic
        # ).T  # this gets sigh with proper shape (n_layer x n_freq) and fortran ordering.
        # mu = np.tile(mu, (n_frequency, 1)).T  # shape(n_layer x n_freq)

        sigh = sigma_hat(
            np.tile(self.frequency.reshape((1, n_frequency)), (n_layer, 1)),
            sigma, epsilon,
            quasistatic=self.quasistatic
        )

        rTE = rTE_forward(f, lambd.reshape(-1), sigh, mu, thick)
        rTE = rTE.reshape((n_frequency, *lambd.shape))

        # secondary is height of receiver plus height of source
        rTE *= np.exp(-lambd*(xyz[:, -1] + h)[:, None])
        # works for variable xyz because each point has it's own lambdas

        src_x, src_y, src_z = self.orientation
        C0x = C0y = C0z = 0.0
        C1x = C1y = C1z = 0.0
        if src_x != 0.0:
            C0x += src_x*(dxyz[:, 0]**2/offsets**2)[:, None]*lambd**2
            C1x += src_x*(1/offsets - 2*dxyz[:, 0]**2/offsets**3)[:, None]*lambd
            C0y += src_x*(dxyz[:, 0]*dxyz[:, 1]/offsets**2)[:, None]*lambd**2
            C1y -= src_x*(2*dxyz[:, 0]*dxyz[:, 1]/offsets**3)[:, None]*lambd
            # C0z += 0.0
            C1z -= (src_x*dxyz[:, 0]/offsets)[:, None]*lambd**2

        if src_y != 0.0:
            C0x += src_y*(dxyz[:, 0]*dxyz[:, 1]/offsets**2)[:, None]*lambd**2
            C1x -= src_y*(2*dxyz[:, 0]*dxyz[:, 1]/offsets**3)[:, None]*lambd
            C0y += src_y*(dxyz[:, 1]**2/offsets**2)[:, None]*lambd**2
            C1y += src_y*(1/offsets - 2*dxyz[:, 1]**2/offsets**3)[:, None]*lambd
            # C0z += 0.0
            C1z -= (src_y*dxyz[:, 1]/offsets)[:, None]*lambd**2

        if src_z != 0.0:
            # C0x += 0.0
            C1x += (src_z*dxyz[:, 0]/offsets)[:, None]*lambd**2
            # C0y += 0.0
            C1y += (src_z*dxyz[:, 1]/offsets)[:, None]*lambd**2
            C0z += src_z*lambd**2
            # C1z += 0.0

        # Do the hankel transform on each component
        em_x = ((C0x*rTE)@fhtfilt.j0 + (C1x*rTE)@fhtfilt.j1)/offsets
        em_y = ((C0y*rTE)@fhtfilt.j0 + (C1y*rTE)@fhtfilt.j1)/offsets
        em_z = ((C0z*rTE)@fhtfilt.j0 + (C1z*rTE)@fhtfilt.j1)/offsets

        if field == "total":
            # add in the primary field
            r = np.linalg.norm(dxyz, axis=-1)
            mdotr = src_x*dxyz[:, 0] + src_y*dxyz[:, 1] + src_z*dxyz[:, 2]

            em_x += 3*dxyz[:, 0]*mdotr/r**5 - src_x/r**3
            em_y += 3*dxyz[:, 1]*mdotr/r**5 - src_y/r**3
            em_z += 3*dxyz[:, 2]*mdotr/r**5 - src_z/r**3

        return self.moment/(4*np.pi)*np.stack((em_x, em_y, em_z), axis=-1).squeeze()
