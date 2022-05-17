import numpy as np
from scipy.constants import epsilon_0
from scipy.constants import mu_0


class ElectrostaticSphere:
    """Class for electrostatic solutions for a sphere in a wholespace.

    The ``ElectrostaticSphere`` class is used to analytically compute the electric
    potentials, fields, currents and charge densities for a sphere in a wholespace.

    Parameters
    ----------
    radius : float
        radius of sphere (m).
    sigma_sphere : float
        conductivity of target sphere (S/m)
    sigma_background : float
        background conductivity (S/m)
    primary_field : (3) array_like, optional
        amplitude of primary electric field.
    location : (3) array_like, optional
        Center of the sphere. Defaults is (0, 0, 0).
    """

    def __init__(
        self, radius, sigma_sphere, sigma_background, primary_field, location=None
    ):

        self.radius = radius
        self.sigma_sphere = sigma_sphere
        self.sigma_background = sigma_background
        self.primary_field = primary_field
        if location is None:
            location = np.r_[0., 0., 0.]
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
    def primary_field(self):
        """Amplitude of the primary current density.

        Returns
        -------
        (3) numpy.ndarray of float
            Amplitude of the primary current density.
        """
        return self._primary_field

    @primary_field.setter
    def primary_field(self, vec):
        try:
            vec = np.atleast_1d(vec).astype(float)
        except:
            raise TypeError(f"primary_field must be array_like, got {type(vec)}")

        if len(vec) != 3:
            raise ValueError(
                f"primary_field must be array_like with shape (3,), got {len(vec)}"
            )

        self._primary_field = vec

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

    def potential(self, xyz, field='all'):
        """Electric potential for a conductive sphere in a uniform electrostatic field.

        Parameters
        ----------
        xyz : (3, ) tuple of np.ndarray or (..., 3) np.ndarray
            locations to evaluate at. If a tuple, all
            the numpy arrays must be the same shape.
        field : {'all', 'total', 'primary', 'secondary'}

        Returns
        -------
        Vt, Vp, Vs : (..., ) np.ndarray
            If field == "all"
        V : (..., ) np.ndarray
            If only requesting a single field.

        Examples
        --------
        Here, we define a sphere with conductivity sigma_sphere in a uniform electrostatic field with conductivity
        sigma_background and plot the electric potentials for total and secondary field.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib import patches
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> from geoana.em.static import ElectrostaticSphere

        Define the sphere.

        >>> location = np.r_[0., 0., 0.]
        >>> sigma_sphere = 10. ** -1
        >>> sigma_background = 10. ** -3
        >>> radius = 1.0
        >>> primary_field = np.r_[1., 1., 1.]
        >>> simulation = ElectrostaticSphere(
        >>>     location=location, sigma_sphere=sigma_sphere, sigma_background=sigma_background, radius=radius, primary_field=primary_field
        >>> )

        Now we create a set of gridded locations, take the distances and compute the magnetic potential.

        >>> X, Y = np.meshgrid(np.linspace(-5, 5, 20), np.linspace(-5, 5, 20))
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> r = np.linalg.norm(xyz, axis=-1)
        >>> vt = simulation.potential(xyz, field='total')
        >>> vs = simulation.potential(xyz, field='secondary')

        Finally, we plot the electric potential for total and secondary fields.

        >>> fig, axs = plt.subplots(1, 2, figsize=(18,12))
        >>> titles = ['Total Potential', 'Secondary Potential']
        >>> for ax, V, title in zip(axs.flatten(), [vt, vs], titles):
        >>>     im = ax.pcolor(X, Y, V, shading='auto')
        >>>     cb1 = plt.colorbar(im, ax=ax)
        >>>     cb1.set_label(label= 'Amplitude')
        >>>     ax.add_patch(patches.Circle((0, 0), radius, fill=False, linestyle='--'))
        >>>     ax.set_ylabel('Y coordinate ($m$)')
        >>>     ax.set_xlabel('X coordinate ($m$)')
        >>>     ax.set_title(title)
        >>>     ax.set_aspect('equal')
        >>> plt.tight_layout()
        >>> plt.show()
        """

        sig0 = self.sigma_background
        sig1 = self.sigma_sphere
        E0 = self.primary_field
        sig_cur = (sig1 - sig0) / (sig1 + 2 * sig0)
        x0, y0, z0 = self.location
        x, y, z = self._check_XYZ(xyz)
        x = x-x0
        y = y-y0
        z = z-z0
        r_vec = xyz - self.location
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        Vt = np.zeros_like(r)
        ind0 = r > self.radius

        # total potential outside the sphere
        Vt[ind0] = r_vec[ind0] @ -E0 * (1. - sig_cur * self.radius ** 3. / r[ind0] ** 3.)

        # inside the sphere
        Vt[~ind0] = r_vec[~ind0] @ -E0 * 3. * sig0 / (sig1 + 2. * sig0)

        # total field
        if field == 'total':
            return Vt

        # primary field
        if field != 'total':
            Vp = r_vec @ -E0
            if field == 'primary':
                return Vp

        # secondary field
        Vs = Vt - Vp
        if field == 'secondary':
            return Vs
        return Vt, Vp, Vs

    def electric_field(self, xyz, field='all'):
        """Electric field for a sphere in a uniform wholespace

        Parameters
        ----------
        xyz : (3, ) tuple of np.ndarray or (..., 3) np.ndarray
            locations to evaluate at. If a tuple, all
            the numpy arrays must be the same shape.
        field : {'all', 'total', 'primary', 'secondary'}

        Returns
        -------
        Et, Ep, Es : (..., 3) np.ndarray
            If field == "all"
        E : (..., 3) np.ndarray
            If only requesting a single field.

        Examples
        --------
        Here, we define a sphere with conductivity sigma_sphere in a uniform electrostatic field with conductivity
        sigma_background and plot the electric field lines for total and secondary field.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib import patches
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> from geoana.em.static import ElectrostaticSphere

        Define the sphere.

        >>> location = np.r_[0., 0., 0.]
        >>> sigma_sphere = 10. ** -1
        >>> sigma_background = 10. ** -3
        >>> radius = 1.0
        >>> primary_field = np.r_[1., 1., 1.]
        >>> simulation = ElectrostaticSphere(
        >>>     location=location, sigma_sphere=sigma_sphere, sigma_background=sigma_background, radius=radius, primary_field=primary_field
        >>> )

        Now we create a set of gridded locations, take the distances and compute the electric fields.

        >>> X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> et = simulation.electric_field(xyz, field='total')
        >>> es = simulation.electric_field(xyz, field='secondary')

        Finally, we plot the magnetic field lines.

        >>> fig, axs = plt.subplots(1, 2, figsize=(18,12))
        >>> titles = ['Total Electric Field', 'Secondary Electric Field']
        >>> for ax, E, title in zip(axs.flatten(), [et, es], titles):
        >>>     E_amp = np.linalg.norm(E, axis=-1)
        >>>     im = ax.pcolor(X, Y, E_amp, shading='auto')
        >>>     divider = make_axes_locatable(ax)
        >>>     cax = divider.append_axes("right", size="5%", pad=0.05)
        >>>     cb = plt.colorbar(im, cax=cax)
        >>>     cb.set_label(label= 'Amplitude ($V/m$)')
        >>>     ax.streamplot(X, Y, E[..., 0], E[..., 1], density=0.75)
        >>>     ax.add_patch(patches.Circle((0, 0), radius, fill=False, linestyle='--'))
        >>>     ax.set_ylabel('Y coordinate ($m$)')
        >>>     ax.set_xlabel('X coordinate ($m$)')
        >>>     ax.set_aspect('equal')
        >>>     ax.set_title(title)
        >>> plt.tight_layout()
        >>> plt.show()
        """

        sig0 = self.sigma_background
        sig1 = self.sigma_sphere
        E0 = self.primary_field
        sig_cur = (sig1 - sig0) / (sig1 + 2 * sig0)

        x, y, z = self._check_XYZ(xyz)
        x0, y0, z0 = self.location
        x = x-x0
        y = y-y0
        z = z-z0
        r_vec = xyz - self.location
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        Et = np.zeros((*r.shape, 3))
        ind0 = r > self.radius

        # total field outside the sphere
        Et[ind0] = E0 * (1. - sig_cur * self.radius ** 3. / r[ind0, None] ** 3) +\
            3. * (r_vec[ind0, None] @ E0) * sig_cur * self.radius * r_vec[ind0] / r[ind0, None] ** 4

        # inside the sphere
        Et[~ind0] = 3. * sig0 / (sig1 + 2. * sig0) * E0

        # total field
        if field == 'total':
            return Et

        # primary field
        if field != 'total':
            Ep = E0
            if field == 'primary':
                return Ep

        # secondary field
        Es = Et - Ep
        if field == 'secondary':
            return Es
        return Et, Ep, Es

    def current_density(self, xyz, field='all'):
        """Current density for a sphere in a uniform wholespace

        Parameters
        ----------
        xyz : (3, ) tuple of np.ndarray or (..., 3) np.ndarray
            locations to evaluate at. If a tuple, all
            the numpy arrays must be the same shape.
        field : {'all', 'total', 'primary', 'secondary'}

        Returns
        -------
        Jt, Jp, Js : (..., 3) np.ndarray
            If field == "all"
        J : (..., 3) np.ndarray
            If only requesting a single field.

        Examples
        --------
        Here, we define a sphere with conductivity sigma_sphere in a uniform electrostatic field with conductivity
        sigma_background and plot the current density for total and secondary field.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib import patches
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> from geoana.em.static import ElectrostaticSphere

        Define the sphere.

        >>> location = np.r_[0., 0., 0.]
        >>> sigma_sphere = 10. ** -1
        >>> sigma_background = 10. ** -3
        >>> radius = 1.0
        >>> primary_field = np.r_[1., 1., 1.]
        >>> simulation = ElectrostaticSphere(
        >>>     location=location, sigma_sphere=sigma_sphere, sigma_background=sigma_background, radius=radius, primary_field=primary_field
        >>> )

        Now we create a set of gridded locations, take the distances and compute the electric fields.

        >>> X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> jt = simulation.current_density(xyz, field='total')
        >>> js = simulation.current_density(xyz, field='secondary')

        Finally, we plot the current densities for total and secondary fields.

        >>> fig, axs = plt.subplots(1, 2, figsize=(18,12))
        >>> titles = ['Total Current Density', 'Secondary Current Density']
        >>> for ax, J, title in zip(axs.flatten(), [jt, js], titles):
        >>>     J_amp = np.linalg.norm(J, axis=-1)
        >>>     im = ax.pcolor(X, Y, J_amp, shading='auto')
        >>>     divider = make_axes_locatable(ax)
        >>>     cax = divider.append_axes("right", size="5%", pad=0.05)
        >>>     cb = plt.colorbar(im, cax=cax)
        >>>     cb.set_label(label= 'Amplitude ($V/m$)')
        >>>     ax.streamplot(X, Y, J[..., 0], J[..., 1], density=0.75)
        >>>     ax.add_patch(patches.Circle((0, 0), radius, fill=False, linestyle='--'))
        >>>     ax.set_ylabel('Y coordinate ($m$)')
        >>>     ax.set_xlabel('X coordinate ($m$)')
        >>>     ax.set_aspect('equal')
        >>>     ax.set_title(title)
        >>> plt.tight_layout()
        >>> plt.show()
        """

        Et = self.electric_field(xyz, field='total')
        Ep = self.electric_field(xyz, field='primary')

        x, y, z = self._check_XYZ(xyz)
        x0, y0, z0 = self.location
        x = x-x0
        y = y-y0
        z = z-z0
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        sigma = np.full(r.shape, self.sigma_background)
        sigma[r <= self.radius] = self.sigma_sphere

        Jt = sigma[..., None] * Et
        if field == 'total':
            return Jt

        if field != 'total':
            Jp = self.sigma_background * Ep
            if field == 'primary':
                return Jp

        Js = Jt - Jp
        if field == 'secondary':
            return Js
        return Jt, Jp, Js

    def charge_density(self, xyz, dr=None):
        """charge density on the surface of a sphere in a uniform wholespace

        Parameters
        ----------
        xyz : (3, ) tuple of np.ndarray or (..., 3) np.ndarray
            locations to evaluate at. If a tuple, all
            the numpy arrays must be the same shape.
        dr : float, optional
            Buffer around the edge of the sphere to calculate
            current density. Defaults to 5 % of the sphere radius

        Returns
        -------
        rho: (..., ) np.ndarray

        Examples
        --------
        Here, we define a sphere with conductivity sigma_sphere in a uniform electrostatic field with conductivity
        sigma_background and plot the current density.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib import patches
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> from geoana.em.static import ElectrostaticSphere

        Define the sphere.

        >>> location = np.r_[0., 0., 0.]
        >>> sigma_sphere = 10. ** -1
        >>> sigma_background = 10. ** -3
        >>> radius = 1.0
        >>> primary_field = [1., 1., 1.]
        >>> simulation = ElectrostaticSphere(
        >>>     location=location, sigma_sphere=sigma_sphere, sigma_background=sigma_background, radius=radius, primary_field=primary_field
        >>> )

        Now we create a set of gridded locations, take the distances and compute the charge density.

        >>> X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> q = simulation.charge_density(xyz)

        Finally, we plot the charge density.

        >>> fig, ax = plt.subplots(figsize=(18,6))
        >>> im = ax.pcolor(X, Y, q, shading='auto')
        >>> cb1 = plt.colorbar(im, ax=ax)
        >>> cb1.set_label(label= 'Charge Density ($C/m^2$)')
        >>> ax.add_patch(patches.Circle((0, 0), radius, fill=False, linestyle='--'))
        >>> ax.set_ylabel('Y coordinate ($m$)')
        >>> ax.set_xlabel('X coordinate ($m$)')
        >>> ax.set_title('Charge Accumulation')
        >>> ax.set_aspect('equal')
        >>> plt.show()
        """

        sig0 = self.sigma_background
        sig1 = self.sigma_sphere
        sig_cur = (sig1 - sig0) / (sig1 + 2 * sig0)
        Ep = self.electric_field(xyz, field='primary')

        x, y, z = self._check_XYZ(xyz)
        x0, y0, z0 = self.location
        x = x-x0
        y = y-y0
        z = z-z0
        r_vec = xyz - self.location
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        if dr is None:
            dr = 0.05 * self.radius

        ind = (r < self.radius + 0.5 * dr) & (r > self.radius - 0.5 * dr)

        rho = np.zeros((*r.shape, 3))
        rho[ind] = epsilon_0 * 3. * Ep * sig_cur * r_vec[ind] / r[ind, None]

        return rho


class MagnetostaticSphere:
    """Class for magnetostatic solutions for a permeable sphere in a uniform magnetostatic field.

        The ``MagnetostaticSphere`` class is used to analytically compute the magnetic
        potentials, fields, and magnetic flux densities for a permeable sphere in a uniform magnetostatic field.

        Parameters
        ----------
        radius : float
            radius of sphere (m).
        mu_sphere : float
            permeability of target sphere (H/m).
        mu_background : float
            background permeability (H/m).
        primary_field : (3) array_like, optional
            amplitude of primary magnetic field
        location : (3) array_like, optional
            Center of the sphere. Defaults is (0, 0, 0).
        """

    def __init__(
        self, radius, mu_sphere, mu_background, primary_field, location=None
    ):

        self.radius = radius
        self.mu_sphere = mu_sphere
        self.mu_background = mu_background
        self.primary_field = primary_field
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
    def primary_field(self):
        """Amplitude of the primary current density.

        Returns
        -------
        (3) numpy.ndarray of float
            Amplitude of the primary current density.
        """
        return self._primary_field

    @primary_field.setter
    def primary_field(self, vec):

        try:
            vec = np.atleast_1d(vec).astype(float)
        except:
            raise TypeError(f"primary_field must be array_like, got {type(vec)}")

        if len(vec) != 3:
            raise ValueError(
                f"primary_field must be array_like with shape (3,), got {len(vec)}"
            )

        self._primary_field = vec

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

    def potential(self, xyz, field='all'):
        """Magnetic potential for a permeable sphere in a uniform magnetostatic field.

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
        mu_background and plot the magnetic potentials for total and secondary field.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib import patches
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> from geoana.em.static import MagnetostaticSphere

        Define the sphere.

        >>> location = np.r_[0., 0., 0.]
        >>> mu_sphere = 10. ** -1
        >>> mu_background = 10. ** -3
        >>> radius = 1.0
        >>> primary_field = [1., 1., 1.]
        >>> simulation = MagnetostaticSphere(
        >>>     location=location, mu_sphere=mu_sphere, mu_background=mu_background, radius=radius, primary_field=primary_field
        >>> )

        Now we create a set of gridded locations, take the distances and compute the magnetic potential.

        >>> X, Y = np.meshgrid(np.linspace(-5, 5, 20), np.linspace(-5, 5, 20))
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> r = np.linalg.norm(xyz, axis=-1)
        >>> vt = simulation.potential(xyz, field='total')
        >>> vs = simulation.potential(xyz, field='secondary')

        Finally, we plot the total and secondary potentials.

        >>> fig, axs = plt.subplots(1, 2, figsize=(18,12))
        >>> titles = ['Total Potential', 'Secondary Potential']
        >>> for ax, V, title in zip(axs.flatten(), [vt, vs], titles):
        >>>     im = ax.pcolor(X, Y, V, shading='auto')
        >>>     cb1 = plt.colorbar(im, ax=ax)
        >>>     cb1.set_label(label= 'Amplitude')
        >>>     ax.add_patch(patches.Circle((0, 0), radius, fill=False, linestyle='--'))
        >>>     ax.set_ylabel('Y coordinate ($m$)')
        >>>     ax.set_xlabel('X coordinate ($m$)')
        >>>     ax.set_title(title)
        >>>     ax.set_aspect('equal')
        >>> plt.tight_layout()
        >>> plt.show()
        """

        mu0 = self.mu_background
        mu1 = self.mu_sphere
        H0 = self.primary_field
        mu_cur = (mu1 - mu0) / (mu1 + 2 * mu0)
        x0, y0, z0 = self.location
        x, y, z = self._check_XYZ(xyz)
        x = x - x0
        y = y - y0
        z = z - z0
        r_vec = xyz - self.location
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        Vt = np.zeros_like(r)
        ind0 = r > self.radius

        # total potential outside the sphere
        Vt[ind0] = r_vec[ind0] @ -H0 * (1. - mu_cur * self.radius ** 3. / r[ind0] ** 3.)

        # inside the sphere
        Vt[~ind0] = r_vec[~ind0] @ -H0 * 3. * mu0 / (mu1 + 2. * mu0)

        # total field
        if field == 'total':
            return Vt

        # primary field
        if field != 'total':
            Vp = r_vec @ -H0
            if field == 'primary':
                return Vp

        # secondary field
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
        mu_background and plot the magnetic field lines for total and secondary field.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib import patches
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> from geoana.em.static import MagnetostaticSphere

        Define the sphere.

        >>> location = np.r_[0., 0., 0.]
        >>> mu_sphere = 10. ** -1
        >>> mu_background = 10. ** -3
        >>> radius = 1.0
        >>> primary_field = [1., 1., 1.]
        >>> simulation = MagnetostaticSphere(
        >>>     location=location, mu_sphere=mu_sphere, mu_background=mu_background, radius=radius, primary_field=primary_field
        >>> )

        Now we create a set of gridded locations, take the distances and compute the magnetic fields.

        >>> X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> ht = simulation.magnetic_field(xyz, field='total')
        >>> hs = simulation.magnetic_field(xyz, field='secondary')

        Finally, we plot the total and secondary magnetic fields.

        >>> fig, axs = plt.subplots(1, 2, figsize=(18,12))
        >>> titles = ['Total Magnetic Field', 'Secondary Magnetic Field']
        >>> for ax, H, title in zip(axs.flatten(), [ht, hs], titles):
        >>>     H_amp = np.linalg.norm(H, axis=-1)
        >>>     im = ax.pcolor(X, Y, H_amp, shading='auto')
        >>>     divider = make_axes_locatable(ax)
        >>>     cax = divider.append_axes("right", size="5%", pad=0.05)
        >>>     cb = plt.colorbar(im, cax=cax)
        >>>     cb.set_label(label= 'Amplitude ($V/m$)')
        >>>     ax.streamplot(X, Y, H[..., 0], H[..., 1], density=0.75)
        >>>     ax.add_patch(patches.Circle((0, 0), radius, fill=False, linestyle='--'))
        >>>     ax.set_ylabel('Y coordinate ($m$)')
        >>>     ax.set_xlabel('X coordinate ($m$)')
        >>>     ax.set_aspect('equal')
        >>>     ax.set_title(title)
        >>> plt.tight_layout()
        >>> plt.show()
        """

        mu0 = self.mu_background
        mu1 = self.mu_sphere
        H0 = self.primary_field
        mu_cur = (mu1 - mu0) / (mu1 + 2 * mu0)

        x, y, z = self._check_XYZ(xyz)
        x0, y0, z0 = self.location
        x = x - x0
        y = y - y0
        z = z - z0
        r_vec = xyz - self.location
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        Ht = np.zeros((*x.shape, 3))
        ind0 = r > self.radius

        # total field outside the sphere
        Ht[ind0] = H0 * (1. - mu_cur * self.radius ** 3. / r[ind0, None] ** 3) +\
            3. * r_vec[ind0, None] @ H0 * mu_cur * self.radius * r_vec[ind0] / r[ind0, None] ** 4

        # inside the sphere
        Ht[~ind0] = 3. * mu0 / (mu1 + 2. * mu0) * H0

        if field == 'total':
            return Ht

        if field != 'total':
            Hp = H0
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
        mu_background and plot the magnetic flux densities for total and secondary fields.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib import patches
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> from geoana.em.static import MagnetostaticSphere

        Define the sphere.

        >>> location = np.r_[0., 0., 0.]
        >>> mu_sphere = 10. ** -1
        >>> mu_background = 10. ** -3
        >>> radius = 1.0
        >>> primary_field = [1., 1., 1.]
        >>> simulation = MagnetostaticSphere(
        >>>     location=location, mu_sphere=mu_sphere, mu_background=mu_background, radius=radius, primary_field=primary_field
        >>> )

        Now we create a set of gridded locations, take the distances and compute the magnetic flux densities.

        >>> X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> bt = simulation.magnetic_flux_density(xyz, field='total')
        >>> bs = simulation.magnetic_flux_density(xyz, field='secondary')

        Finally, we plot the total and secondary magnetic flux densities.

        >>> fig, axs = plt.subplots(1, 2, figsize=(18,12))
        >>> titles = ['Total Magnetic Flux Density', 'Secondary Magnetic Flux Density']
        >>> for ax, B, title in zip(axs.flatten(), [bt, bs], titles):
        >>>     B_amp = np.linalg.norm(B, axis=-1)
        >>>     im = ax.pcolor(X, Y, B_amp, shading='auto')
        >>>     divider = make_axes_locatable(ax)
        >>>     cax = divider.append_axes("right", size="5%", pad=0.05)
        >>>     cb = plt.colorbar(im, cax=cax)
        >>>     cb.set_label(label= 'Amplitude ($V/m$)')
        >>>     ax.streamplot(X, Y, B[..., 0], B[..., 1], density=0.75)
        >>>     ax.add_patch(patches.Circle((0, 0), radius, fill=False, linestyle='--'))
        >>>     ax.set_ylabel('Y coordinate ($m$)')
        >>>     ax.set_xlabel('X coordinate ($m$)')
        >>>     ax.set_aspect('equal')
        >>>     ax.set_title(title)
        >>> plt.tight_layout()
        >>> plt.show()
        """

        if field == 'total':
            return self.magnetic_field(xyz, field='total') * mu_0
        if field == 'primary':
            return self.magnetic_field(xyz, field='primary') * mu_0
        if field == 'secondary':
            return self.magnetic_field(xyz, field='secondary') * mu_0




