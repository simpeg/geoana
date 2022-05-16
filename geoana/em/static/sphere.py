import numpy as np
from scipy.constants import epsilon_0
from scipy.constants import mu_0


class ElectrostaticSphere:
    """Class for electrostatic solutions for a sphere in a wholespace.

    The ``ElectrostaticSphere`` class is used to analytically compute the electric
    potentials, fields, currents and charge densities for a sphere in a wholespace.
    For this class, we assume a homogeneous primary electric field along the
    :math:`\\hat{x}` direction.

    Parameters
    ----------
    radius : float
        radius of sphere (m).
    sigma_sphere : float
        conductivity of target sphere (S/m)
    sigma_background : float
        background conductivity (S/m)
    amplitude : float, optional
        amplitude of primary electric field along the :math:`\\hat{x}` direction (V/m).
        Default is 1.
    location : (3) array_like, optional
        Center of the sphere. Defaults is (0, 0, 0).
    """

    def __init__(
        self, radius, sigma_sphere, sigma_background, amplitude=1.0, location=np.r_[0.,0.,0.]
    ):

        self.radius = radius
        self.sigma_sphere = sigma_sphere
        self.sigma_background = sigma_background
        self.amplitude = amplitude
        self.location = location

    @property
    def sigma_sphere(self):
        """Electrical conductivity of the sphere in S/m

        Returns
        -------
        float
            Electrical conductivity of the sphere in S/m
        """
        return self._sigma_sphere

    @sigma_sphere.setter
    def sigma_sphere(self, item):
        item = float(item)
        if item <= 0.0:
            raise ValueError('Conductiviy must be positive')
        self._sigma_sphere = item

    @property
    def sigma_background(self):
        """Electrical conductivity of the background in S/m

        Returns
        -------
        float
            Electrical conductivity of the background in S/m
        """
        return self._sigma_background

    @sigma_background.setter
    def sigma_background(self, item):
        item = float(item)
        if item <= 0.0:
            raise ValueError('Conductiviy must be positive')
        self._sigma_background = item

    @property
    def radius(self):
        """Radius of the sphere in meters

        Returns
        -------
        float
            Radius of the sphere in meters
        """
        return self._radius

    @radius.setter
    def radius(self, item):
        item = float(item)
        if item < 0.0:
            raise ValueError('radius must be non-negative')
        self._radius = item

    @property
    def amplitude(self):
        """Amplitude of the primary current density along the x-direction.

        Returns
        -------
        float
            Amplitude of the primary current density along the x-direction in :math:`A/m^2`.
        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, item):
        self._amplitude = float(item)

    @property
    def location(self):
        """Center of the sphere

        Returns
        -------
        (3) numpy.ndarray of float
            Center of the sphere. Default = np.r_[0,0,0]
        """
        return self._location

    @location.setter
    def location(self, vec):

        try:
            vec = np.atleast_1d(vec).astype(float)
        except:
            raise TypeError(f"location must be array_like, got {type(vec)}")

        if len(vec) != 3:
            raise ValueError(
                f"location must be array_like with shape (3,), got {len(vec)}"
            )

        self._location = vec

    def _check_XYZ(self, XYZ):
        if len(XYZ) == 3:
            x, y, z = XYZ
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            z = np.asarray(z, dtype=float)
        elif isinstance(XYZ, np.ndarray) and XYZ.shape[-1] == 3:
            x, y, z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
        else:
            raise TypeError(
                "XYZ must be either a length three tuple of each dimension, "
                "or a numpy.ndarray of shape (..., 3)."
                )
        if not (x.shape == y.shape and x.shape == z.shape):
            raise ValueError(
                "x, y, z must all have the same shape"
            )
        return x, y, z

    def potential(self, XYZ, field='all'):
        """Compute the electric potential.

        Parameters
        ----------
        XYZ : (3, ) tuple of np.ndarray or (..., 3) np.ndarray
            locations to evaluate at. If a tuple, all
            the numpy arrays must be the same shape.
        field : {'all', 'total', 'primary', 'secondary'}

        Returns
        -------
        Vt, Vp, Vs : (..., ) np.ndarray
            If field == "all"
        V : (..., ) np.ndarray
            If only requesting a single field.
        """
        sig0 = self.sigma_background
        sig1 = self.sigma_sphere
        E0 = self.amplitude
        sig_cur = (sig1 - sig0) / (sig1 + 2 * sig0)
        x0, y0, z0 = self.location
        x, y, z = self._check_XYZ(XYZ)
        x = x-x0
        y = y-y0
        z = z-z0
        r = np.sqrt(x**2 + y**2 + z**2)

        if field != 'total':
            Vp = -E0 * x
            if field == 'primary':
                return Vp

        Vt = np.zeros_like(r)
        ind0 = r > self.radius
        # total potential outside the sphere
        Vt[ind0] = -E0*x[ind0]*(1.-sig_cur*self.radius**3./r[ind0]**3.)
        # inside the sphere
        Vt[~ind0] = -E0*x[~ind0]*3.*sig0/(sig1+2.*sig0)

        if field == 'total':
            return Vt
        # field was not primary or total
        Vs = Vt - Vp
        if field == 'secondary':
            return Vs
        return Vt, Vp, Vs

    def electric_field(self, XYZ, field='all'):
        """Electric field for a sphere in a uniform wholespace

        Parameters
        ----------
        XYZ : (3, ) tuple of np.ndarray or (..., 3) np.ndarray
            locations to evaluate at. If a tuple, all
            the numpy arrays must be the same shape.
        field : {'all', 'total', 'primary', 'secondary'}

        Returns
        -------
        Et, Ep, Es : (..., 3) np.ndarray
            If field == "all"
        E : (..., 3) np.ndarray
            If only requesting a single field.
        """
        sig0 = self.sigma_background
        sig1 = self.sigma_sphere
        E0 = self.amplitude
        sig_cur = (sig1 - sig0) / (sig1 + 2 * sig0)

        x, y, z = self._check_XYZ(XYZ)
        x0, y0, z0 = self.location
        x = x-x0
        y = y-y0
        z = z-z0
        r = np.sqrt(x**2 + y**2 + z**2)

        if field != 'total':
            Ep = np.zeros((*x.shape, 3))
            Ep[..., 0] = E0
            if field == 'primary':
                return Ep

        Et = np.zeros((*x.shape, 3))
        ind0 = r > self.radius
        # total field outside the sphere
        Et[ind0, 0] = E0 + E0*self.radius**3./(r[ind0]**5.)*sig_cur*(2.*x[ind0]**2.-y[ind0]**2.-z[ind0]**2.)
        Et[ind0, 1] = E0*self.radius**3./(r[ind0]**5.)*3.*x[ind0]*y[ind0]*sig_cur
        Et[ind0, 2] = E0*self.radius**3./(r[ind0]**5.)*3.*x[ind0]*z[ind0]*sig_cur
        # inside the sphere
        Et[~ind0, 0] = 3.*sig0/(sig1+2.*sig0)*E0

        if field == 'total':
            return Et
        # field was not primary or total
        Es = Et - Ep
        if field == 'secondary':
            return Es
        return Et, Ep, Es

    def current_density(self, XYZ, field='all'):
        """Current density for a sphere in a uniform wholespace

        Parameters
        ----------
        XYZ : (3, ) tuple of np.ndarray or (..., 3) np.ndarray
            locations to evaluate at. If a tuple, all
            the numpy arrays must be the same shape.
        field : {'all', 'total', 'primary', 'secondary'}

        Returns
        -------
        Jt, Jp, Js : (..., 3) np.ndarray
            If field == "all"
        J : (..., 3) np.ndarray
            If only requesting a single field.
        """

        Et, Ep, Es = self.electric_field(XYZ, field='all')
        if field != 'total':
            Jp = self.sigma_background * Ep
            if field == 'primary':
                return Jp

        x, y, z = self._check_XYZ(XYZ)
        x0, y0, z0 = self.location
        x = x-x0
        y = y-y0
        z = z-z0
        r = np.sqrt(x**2 + y**2 + z**2)

        sigma = np.full(r.shape, self.sigma_background)
        sigma[r <= self.radius] = self.sigma_sphere

        Jt = sigma[..., None] * Et
        if field == 'total':
            return Jt

        Js = Jt - Jp
        if field == 'secondary':
            return Js
        return Jt, Jp, Js

    def charge_density(self, XYZ, dr=None):
        """charge density on the surface of a sphere in a uniform wholespace

        Parameters
        ----------
        XYZ : (3, ) tuple of np.ndarray or (..., 3) np.ndarray
            locations to evaluate at. If a tuple, all
            the numpy arrays must be the same shape.
        dr : float, optional
            Buffer around the edge of the sphere to calculate
            current density. Defaults to 5 % of the sphere radius

        Returns
        -------
        rho: (..., ) np.ndarray
        """

        sig0 = self.sigma_background
        sig1 = self.sigma_sphere
        sig_cur = (sig1 - sig0) / (sig1 + 2 * sig0)
        Ep = self.electric_field(XYZ, field='primary')

        x, y, z = self._check_XYZ(XYZ)
        x0, y0, z0 = self.location
        x = x-x0
        y = y-y0
        z = z-z0
        r = np.sqrt(x**2 + y**2 + z**2)

        if dr is None:
            dr = 0.05 * self.radius

        ind = (r < self.radius + 0.5*dr) & (r > self.radius - 0.5*dr)

        rho = np.zeros_like(r)
        rho[ind] = epsilon_0*3.*Ep[ind, 0]*sig_cur*x[ind]/(np.sqrt(x[ind]**2.+y[ind]**2.))

        return rho


class MagnetostaticSphere:
    """Class for magnetostatic solutions for a permeable sphere in a uniform magnetostatic field.

        The ``MagnetostaticSphere`` class is used to analytically compute the magnetic
        potentials, fields, and magnetic flux densities for a permeable sphere in a uniform magnetostatic field.
        For this class, we assume a homogeneous primary magnetic field along the
        :math:`\\hat{x}` direction.

        Parameters
        ----------
        radius : float
            radius of sphere (m).
        mu_sphere : float
            permeability of target sphere (H/m).
        mu_background : float
            background permeability (H/m).
        amplitude : float, optional
            amplitude of primary magnetic field along the :math:`\\hat{x}` direction (A/m).
            Default is 1.
        location : (3) array_like, optional
            Center of the sphere. Defaults is (0, 0, 0).
        """

    def __init__(
        self, radius, mu_sphere, mu_background, amplitude=1.0, location=None
    ):

        self.radius = radius
        self.mu_sphere = mu_sphere
        self.mu_background = mu_background
        self.amplitude = amplitude
        if location is None:
            location = np.r_[0, 0, 0]
        self.location = location

    @property
    def mu_sphere(self):
        """Magnetic permeability of the sphere in H/m.

        Returns
        -------
        float
            Magnetic permeability of the sphere in H/m.
        """
        return self._mu_sphere

    @mu_sphere.setter
    def mu_sphere(self, item):
        item = float(item)
        if item <= 0.0:
            raise ValueError('Permeability must be positive')
        self._mu_sphere = item

    @property
    def mu_background(self):
        """Magnetic permeability of the background in H/m.

        Returns
        -------
        float
            Magnetic permeability of the background in H/m.
        """
        return self._mu_background

    @mu_background.setter
    def mu_background(self, item):
        item = float(item)
        if item <= 0.0:
            raise ValueError('Permeability must be positive')
        self._mu_background = item

    @property
    def radius(self):
        """Radius of the sphere in meters.

        Returns
        -------
        float
            Radius of the sphere in meters.
        """
        return self._radius

    @radius.setter
    def radius(self, item):
        item = float(item)
        if item < 0.0:
            raise ValueError('radius must be non-negative')
        self._radius = item

    @property
    def amplitude(self):
        """Amplitude of the primary current density along the x-direction.

        Returns
        -------
        float
            Amplitude of the primary current density along the x-direction in :math:`A/m^2`.
        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, item):
        self._amplitude = float(item)

    @property
    def location(self):
        """Center of the sphere

        Returns
        -------
        (3) numpy.ndarray of float
            Center of the sphere. Default = np.r_[0,0,0]
        """
        return self._location

    @location.setter
    def location(self, vec):

        try:
            vec = np.atleast_1d(vec).astype(float)
        except:
            raise TypeError(f"location must be array_like, got {type(vec)}")

        if len(vec) != 3:
            raise ValueError(
                f"location must be array_like with shape (3,), got {len(vec)}"
            )

        self._location = vec

    def _check_xyz(self, xyz):
        if len(xyz) == 3:
            x, y, z = xyz
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            z = np.asarray(z, dtype=float)
        elif isinstance(xyz, np.ndarray) and xyz.shape[-1] == 3:
            x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        else:
            raise TypeError(
                "xyz must be either a length three tuple of each dimension, "
                "or a numpy.ndarray of shape (..., 3)."
                )
        if not (x.shape == y.shape and x.shape == z.shape):
            raise ValueError(
                "x, y, z must all have the same shape"
            )
        return x, y, z

    def potential(self, xyz, field='all'):
        """Magnetic potential for a permeable sphere in a uniform magnetostatic field.

        .. math::



        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Locations to evaluate at in units m.
        field : {'all', 'total', 'primary', 'secondary'}

        Returns
        -------
        Vt, Vp, Vs : (..., ) np.ndarray
            If field == "all"
        V : (..., ) np.ndarray
            Potential for permeable sphere in a uniform magnetostatic field in units T m if requesting single field.

        Examples
        --------
        Here, we define a sphere with permeability mu_sphere in a uniform magnetostatic field with permeability
        mu_background and plot the magnetic potential as a function of distance.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from geoana.em.static import MagnetostaticSphere

        Define the sphere.

        >>> location = np.r_[0., 0., 0.]
        >>> mu_sphere = 1.0
        >>> mu_background = 1.0
        >>> radius = 1.0
        >>> amplitude = 1.0
        >>> simulation = MagnetostaticSphere(
        >>>     location=location, mu_sphere=mu_sphere, mu_background=mu_background, radius=radius, amplitude=amplitude
        >>> )

        Now we create a set of gridded locations, take the distances and compute the magnetic potential.

        >>> X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> r = np.linalg.norm(xyz, axis=-1)
        >>> vt = simulation.potential(xyz, field='total')
        >>> vp = simulation.potential(xyz, field='primary')
        >>> vs = simulation.potential(xyz, field='secondary')

        Finally, we plot the magnetic potential as a function of distance.

        >>> fig = plt.figure(figsize=(10, 10))
        >>> gs = fig.add_gridspec(3, 1, hspace=0, wspace=0)
        >>> (ax1, ax2, ax3) = gs.subplots(sharex='col', sharey='row')
        >>> fig.suptitle('Magnetic Potential for a Sphere as a function of distance')
        >>> ax1.plot(r, vt)
        >>> ax2.plot(r, vp)
        >>> ax3.plot(r, vs)
        >>> plt.show()
        """

        mu0 = self.mu_background
        mu1 = self.mu_sphere
        H0 = self.amplitude
        mu_cur = (mu1 - mu0) / (mu1 + 2 * mu0)
        x0, y0, z0 = self.location
        x, y, z = self._check_xyz(xyz)
        x = x-x0
        y = y-y0
        z = z-z0
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        Vt = np.zeros_like(r)
        ind0 = r > self.radius

        # total potential outside the sphere
        Vt[ind0] = -H0 * x[ind0] * (1. - mu_cur * self.radius ** 3. / r[ind0] ** 3.)

        # inside the sphere
        Vt[~ind0] = -H0 * x[~ind0] * 3. * mu0 / (mu1 + 2. * mu0)

        if field == 'total':
            return Vt

        if field != 'total':
            Vp = np.zeros_like(r)
            Vp[..., 0] = H0
            if field == 'primary':
                return Vp

        # field was not primary or total
        Vs = Vt - Vp
        if field == 'secondary':
            return Vs
        return Vt, Vp, Vs

    def magnetic_field(self, xyz, field='all'):
        """Magnetic field for a permeable sphere in a uniform magnetostatic field.  See Ward and Hohmann,
        1988 Equation 6.69.

        .. math::

            \\mathbf{H} = - \\nabla U

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Locations to evaluate at in units m.
        field : {'all', 'total', 'primary', 'secondary'}

        Returns
        -------
        Ht, Hp, Hs : (..., 3) np.ndarray
            If field == "all"
        H : (..., 3) np.ndarray
            Magnetic field for permeable sphere in a uniform magnetostatic field in units T if requesting single field.

        Examples
        --------
        Here, we define a sphere with permeability mu_sphere in a uniform magnetostatic field with permeability
        mu_background and plot the magnetic field lines for total, primary and secondary field.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from geoana.em.static import MagnetostaticSphere

        Define the sphere.

        >>> location = np.r_[0., 0., 0.]
        >>> mu_sphere = 1.0
        >>> mu_background = 1.0
        >>> radius = 1.0
        >>> amplitude = 1.0
        >>> simulation = MagnetostaticSphere(
        >>>     location=location, mu_sphere=mu_sphere, mu_background=mu_background, radius=radius, amplitude=amplitude
        >>> )

        Now we create a set of gridded locations, take the distances and compute the magnetic fields.

        >>> X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> vt = simulation.magnetic_field(xyz, field='total')
        >>> vp = simulation.magnetic_field(xyz, field='primary')
        >>> vs = simulation.magnetic_field(xyz, field='secondary')

        Finally, we plot the magnetic field lines.

        >>> fig = plt.figure(figsize=(10, 10))
        >>> gs = fig.add_gridspec(3, 1, hspace=0, wspace=0)
        >>> (ax1, ax2, ax3) = gs.subplots(sharex='col', sharey='row')
        >>> fig.suptitle('Magnetic Field Lines for a Sphere')
        >>> ax1.quiver(X, Y, vt[:,:,0], vt[:,:,1])
        >>> ax2.quiver(X, Y, vp[:,:,0], vp[:,:,1])
        >>> ax3.quiver(X, Y, vs[:,:,0], vs[:,:,1])
        >>> plt.show()
        """

        mu0 = self.mu_background
        mu1 = self.mu_sphere
        H0 = self.amplitude
        mu_cur = (mu1 - mu0) / (mu1 + 2 * mu0)

        x, y, z = self._check_xyz(xyz)
        x0, y0, z0 = self.location
        x = x-x0
        y = y-y0
        z = z-z0
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        Ht = np.zeros((*x.shape, 3))
        ind0 = r > self.radius

        # total field outside the sphere
        Ht[ind0, 0] = H0 + H0 * self.radius ** 3. * mu_cur *\
            (2. * x[ind0] ** 2. - y[ind0] ** 2. - z[ind0] ** 2.) / (r[ind0] ** 5.)
        Ht[ind0, 1] = H0 * self.radius ** 3. * mu_cur * 3. * x[ind0] * y[ind0] / (r[ind0] ** 5.)
        Ht[ind0, 2] = H0 * self.radius ** 3. * mu_cur * 3. * x[ind0] * z[ind0] / (r[ind0] ** 5.)

        # inside the sphere
        Ht[~ind0, 0] = 3. * mu0 / (mu1 + 2. * mu0) * H0

        if field == 'total':
            return Ht

        if field != 'total':
            Hp = np.zeros((*x.shape, 3))
            Hp[..., 0] = H0
            if field == 'primary':
                return Hp

        # field was not primary or total
        Hs = Ht - Hp
        if field == 'secondary':
            return Hs
        return Ht, Hp, Hs

    def magnetic_flux_density(self, xyz, field='all'):
        """Magnetic flux density for a permeable sphere in a uniform magnetostatic field.

        .. math::

            \\mathbf{B} = \\mu \\mathbf{H}

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Locations to evaluate at in units m.
        field : {'all', 'total', 'primary', 'secondary'}

        Returns
        -------
        Bt, Bp, Bs : (..., 3) np.ndarray
            If field == "all"
        B : (..., 3) np.ndarray
            Magnetic flux density for permeable sphere in a uniform magnetostatic field in units T if requesting
            single field.

        Examples
        --------
        Here, we define a sphere with permeability mu_sphere in a uniform magnetostatic field with permeability
        mu_background and plot the magnetic flux densities for total, primary and secondary field.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from geoana.em.static import MagnetostaticSphere

        Define the sphere.

        >>> location = np.r_[0., 0., 0.]
        >>> mu_sphere = 1.0
        >>> mu_background = 1.0
        >>> radius = 1.0
        >>> amplitude = 1.0
        >>> simulation = MagnetostaticSphere(
        >>>     location=location, mu_sphere=mu_sphere, mu_background=mu_background, radius=radius, amplitude=amplitude
        >>> )

        Now we create a set of gridded locations, take the distances and compute the magnetic flux densities.

        >>> X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> bt = simulation.magnetic_flux_density(xyz, field='total')
        >>> bp = simulation.magnetic_flux_density(xyz, field='primary')
        >>> bs = simulation.magnetic_flux_density(xyz, field='secondary')

        Finally, we plot the magnetic flux densities for total, primary and secondary field.

        >>> fig = plt.figure(figsize=(10, 10))
        >>> gs = fig.add_gridspec(3, 2, hspace=0, wspace=0)
        >>> (ax1, ax2, ax3), (ax4, ax5, ax6) = gs.subplots(sharex='col', sharey='row')
        >>> fig.suptitle('Magnetic Flux Densities for a Sphere')
        >>> ax1.contourf(X, Y, bt[:,:,0])
        >>> ax2.contourf(X, Y, bp[:,:,0])
        >>> ax3.contourf(X, Y, bs[:,:,0])
        >>> ax4.contourf(X, Y, bt[:,:,1])
        >>> ax5.contourf(X, Y, bp[:,:,1])
        >>> ax6.contourf(X, Y, bs[:,:,1])
        >>> plt.show()
        """

        if field == 'total':
            return self.magnetic_field(xyz, field='total') * mu_0
        if field == 'primary':
            return self.magnetic_field(xyz, field='primary') * mu_0
        if field == 'secondary':
            return self.magnetic_field(xyz, field='secondary') * mu_0




