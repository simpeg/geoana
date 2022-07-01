import numpy as np
from scipy.constants import epsilon_0
from geoana.utils import check_xyz_dim


class ElectrostaticSphere:
    """Class for electrostatic solutions for a sphere in a wholespace.

    The ``ElectrostaticSphere`` class is used to analytically compute the electric
    potentials, fields, currents and charge densities for a sphere in a wholespace.

    Parameters
    ----------
    radius : float
        radius of sphere (m).
    sigma_sphere : float
        conductivity of target sphere (S/m).
    sigma_background : float
        background conductivity (S/m).
    primary_field : (3) array_like, optional
        amplitude of primary electric field.  Default is (1, 0, 0).
    location : (3) array_like, optional
        center of the sphere. Default is (0, 0, 0).
    """

    def __init__(
        self, radius, sigma_sphere, sigma_background, primary_field=None, location=None
    ):

        self.radius = radius
        self.sigma_sphere = sigma_sphere
        self.sigma_background = sigma_background
        if primary_field is None:
            primary_field = np.r_[1., 0., 0.]
        self.primary_field = primary_field
        if location is None:
            location = np.r_[0., 0., 0.]
        self.location = location

    @property
    def sigma_sphere(self):
        """Electrical conductivity of the sphere in S/m.

        Returns
        -------
        float
            Electrical conductivity of the sphere in S/m.
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
        """Electrical conductivity of the background in S/m.

        Returns
        -------
        float
            Electrical conductivity of the background in S/m.
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
            Amplitude of the primary current density. Default = np.r_[1,0,0]
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
        """Center of the sphere.

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

    def potential(self, xyz, field='all'):
        """Electric potential for a conductive sphere in a uniform electrostatic field.

       .. math::

            V_p(\\mathbf{r}) = -\\mathbf{E_0} \\dot \\mathbf{r}

            r > R

            V_T(\\mathbf{r}) = -\\mathbf{E_0} \\dot \\mathbf{r} (1 - \\frac{\\sigma_s - \\sigma_0}{\\sigma_s + 2 \\sigma_0} \\frac{R}{r^3}

            r < R

            V_T(\\mathbf{r}) = -3 \\mathbf{E_0} \\dot \\mathbf{r} \\frac{\\sigma_0}{\\sigma_s + 2 \\sigma_0}

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
            If only requesting a single field.

        Examples
        --------
        Here, we define a sphere with conductivity sigma_sphere in a uniform electrostatic field with conductivity
        sigma_background and plot the total and secondary electric potentials.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib import patches
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> from geoana.em.static import ElectrostaticSphere

        Define the sphere.

        >>> sigma_sphere = 10. ** -1
        >>> sigma_background = 10. ** -3
        >>> radius = 1.0
        >>> simulation = ElectrostaticSphere(
        >>>     location=None, sigma_sphere=sigma_sphere, sigma_background=sigma_background, radius=radius, primary_field=None
        >>> )

        Now we create a set of gridded locations and compute the magnetic potential.

        >>> X, Y = np.meshgrid(np.linspace(-2*radius, 2*radius, 20), np.linspace(-2*radius, 2*radius, 20))
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> vt = simulation.potential(xyz, field='total')
        >>> vs = simulation.potential(xyz, field='secondary')

        Finally, we plot the total and secondary electric potentials.

        >>> fig, axs = plt.subplots(1, 2, figsize=(18,12))
        >>> titles = ['Total Potential', 'Secondary Potential']
        >>> for ax, V, title in zip(axs.flatten(), [vt, vs], titles):
        >>>     im = ax.pcolor(X, Y, V, shading='auto')
        >>>     divider = make_axes_locatable(ax)
        >>>     cax = divider.append_axes("right", size="5%", pad=0.05)
        >>>     cb = plt.colorbar(im, cax=cax)
        >>>     cb.set_label(label= 'Potential (V)')
        >>>     ax.add_patch(patches.Circle((0, 0), radius, fill=False, linestyle='--'))
        >>>     ax.set_ylabel('Y coordinate ($m$)')
        >>>     ax.set_xlabel('X coordinate ($m$)')
        >>>     ax.set_title(title)
        >>>     ax.set_aspect('equal')
        >>> plt.tight_layout()
        >>> plt.show()
        """
        xyz = check_xyz_dim(xyz)
        sig0 = self.sigma_background
        sig1 = self.sigma_sphere
        E0 = self.primary_field
        sig_cur = (sig1 - sig0) / (sig1 + 2 * sig0)
        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)

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

        .. math::

            E_p(\\mathbf{r}) = - \\nabla V_p = \\mathbf{E_0}

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Locations to evaluate at in units m.
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
        sigma_background and plot the total and secondary electric fields.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib import patches
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> from geoana.em.static import ElectrostaticSphere

        Define the sphere.

        >>> sigma_sphere = 10. ** -1
        >>> sigma_background = 10. ** -3
        >>> radius = 1.0
        >>> simulation = ElectrostaticSphere(
        >>>     location=None, sigma_sphere=sigma_sphere, sigma_background=sigma_background, radius=radius, primary_field=None
        >>> )

        Now we create a set of gridded locations and compute the electric fields.

        >>> X, Y = np.meshgrid(np.linspace(-2*radius, 2*radius, 20), np.linspace(-2*radius, 2*radius, 20))
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> et = simulation.electric_field(xyz, field='total')
        >>> es = simulation.electric_field(xyz, field='secondary')

        Finally, we plot the total and secondary electric fields.

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
        xyz = check_xyz_dim(xyz)
        sig0 = self.sigma_background
        sig1 = self.sigma_sphere
        E0 = self.primary_field
        sig_cur = (sig1 - sig0) / (sig1 + 2 * sig0)
        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)

        Et = np.zeros((*r.shape, 3))
        ind0 = r > self.radius

        # total field outside the sphere
        Et[ind0] = E0 * (1. - sig_cur * self.radius ** 3. / r[ind0, None] ** 3) +\
            3. * (r_vec[ind0] @ E0)[..., None] * sig_cur * self.radius * r_vec[ind0] / r[ind0, None] ** 4

        # inside the sphere
        Et[~ind0] = 3. * sig0 / (sig1 + 2. * sig0) * E0

        # total field
        if field == 'total':
            return Et

        # primary field
        if field != 'total':
            Ep = np.zeros((*r.shape, 3))
            Ep = Ep + E0
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
        xyz : (..., 3) numpy.ndarray
            Locations to evaluate at in units m.
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
        sigma_background and plot the total and secondary current densities.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib import patches
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> from geoana.em.static import ElectrostaticSphere

        Define the sphere.

        >>> sigma_sphere = 10. ** -1
        >>> sigma_background = 10. ** -3
        >>> radius = 1.0
        >>> simulation = ElectrostaticSphere(
        >>>     location=None, sigma_sphere=sigma_sphere, sigma_background=sigma_background, radius=radius, primary_field=None
        >>> )

        Now we create a set of gridded locations and compute the current densities.

        >>> X, Y = np.meshgrid(np.linspace(-2*radius, 2*radius, 20), np.linspace(-2*radius, 2*radius, 20))
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> jt = simulation.current_density(xyz, field='total')
        >>> js = simulation.current_density(xyz, field='secondary')

        Finally, we plot the total and secondary current densities.

        >>> fig, axs = plt.subplots(1, 2, figsize=(18,12))
        >>> titles = ['Total Current Density', 'Secondary Current Density']
        >>> for ax, J, title in zip(axs.flatten(), [jt, js], titles):
        >>>     J_amp = np.linalg.norm(J, axis=-1)
        >>>     im = ax.pcolor(X, Y, J_amp, shading='auto')
        >>>     divider = make_axes_locatable(ax)
        >>>     cax = divider.append_axes("right", size="5%", pad=0.05)
        >>>     cb = plt.colorbar(im, cax=cax)
        >>>     cb.set_label(label= 'Current Density ($A/m^2$)')
        >>>     ax.streamplot(X, Y, J[..., 0], J[..., 1], density=0.75)
        >>>     ax.add_patch(patches.Circle((0, 0), radius, fill=False, linestyle='--'))
        >>>     ax.set_ylabel('Y coordinate ($m$)')
        >>>     ax.set_xlabel('X coordinate ($m$)')
        >>>     ax.set_aspect('equal')
        >>>     ax.set_title(title)
        >>> plt.tight_layout()
        >>> plt.show()
        """
        xyz = check_xyz_dim(xyz)
        r = np.linalg.norm(xyz - self.location, axis=-1)

        sigma = np.full(r.shape, self.sigma_background)
        sigma[r <= self.radius] = self.sigma_sphere

        Et = self.electric_field(xyz, field='total')
        Jt = sigma[..., None] * Et
        if field == 'total':
            return Jt

        if field != 'total':
            Ep = self.electric_field(xyz, field='primary')
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
        xyz : (..., 3) numpy.ndarray
            Locations to evaluate at in units m.
        dr : float, optional
            Buffer around the edge of the sphere to calculate
            current density. Defaults to 5 % of the sphere radius

        Returns
        -------
        rho: (..., ) np.ndarray

        Examples
        --------
        Here, we define a sphere with conductivity sigma_sphere in a uniform electrostatic field with conductivity
        sigma_background and plot the charge density.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib import patches
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> from geoana.em.static import ElectrostaticSphere

        Define the sphere.

        >>> sigma_sphere = 10. ** -1
        >>> sigma_background = 10. ** -3
        >>> radius = 1.0
        >>> simulation = ElectrostaticSphere(
        >>>     location=None, sigma_sphere=sigma_sphere, sigma_background=sigma_background, radius=radius, primary_field=None
        >>> )

        Now we create a set of gridded locations and compute the charge density.

        >>> X, Y = np.meshgrid(np.linspace(-2*radius, 2*radius, 20), np.linspace(-2*radius, 2*radius, 20))
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> q = simulation.charge_density(xyz, 0.5)

        Finally, we plot the charge density.

        >>> plt.pcolor(X, Y, q, shading='auto')
        >>> cb1 = plt.colorbar()
        >>> cb1.set_label(label= 'Charge Density ($C/m^2$)')
        >>> plt.ylabel('Y coordinate ($m$)')
        >>> plt.xlabel('X coordinate ($m$)')
        >>> plt.title('Charge Accumulation')
        >>> plt.show()
        """
        xyz = check_xyz_dim(xyz)

        sig0 = self.sigma_background
        sig1 = self.sigma_sphere
        sig_cur = (sig1 - sig0) / (sig1 + 2 * sig0)
        E0 = self.primary_field
        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)

        if dr is None:
            dr = 0.05 * self.radius

        ind = (r < self.radius + 0.5 * dr) & (r > self.radius - 0.5 * dr)

        rho = np.zeros_like(r)
        rho[ind] = epsilon_0 * 3. * (r_vec[ind] @ E0) * sig_cur / r[ind]

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
            amplitude of primary magnetic field.  Default is (1, 0, 0)
        location : (3) array_like, optional
            Center of the sphere. Default is (0, 0, 0).
        """

    def __init__(
        self, radius, mu_sphere, mu_background, primary_field=None, location=None
    ):

        self.radius = radius
        self.mu_sphere = mu_sphere
        self.mu_background = mu_background
        if primary_field is None:
            primary_field = np.r_[1., 0., 0.]
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
            Amplitude of the primary current density.  Default = np.r_[1,0,0]
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
            If only requesting a single field.

        Examples
        --------
        Here, we define a sphere with permeability mu_sphere in a uniform magnetostatic field with permeability
        mu_background and plot the total and secondary magnetic potentials.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib import patches
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> from geoana.em.static import MagnetostaticSphere

        Define the sphere.

        >>> mu_sphere = 10. ** -1
        >>> mu_background = 10. ** -3
        >>> radius = 1.0
        >>> simulation = MagnetostaticSphere(
        >>>     location=None, mu_sphere=mu_sphere, mu_background=mu_background, radius=radius, primary_field=None
        >>> )

        Now we create a set of gridded locations and compute the magnetic potential.

        >>> X, Y = np.meshgrid(np.linspace(-2*radius, 2*radius, 20), np.linspace(-2*radius, 2*radius, 20))
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> vt = simulation.potential(xyz, field='total')
        >>> vs = simulation.potential(xyz, field='secondary')

        Finally, we plot the total and secondary magnetic potentials.

        >>> fig, axs = plt.subplots(1, 2, figsize=(18,12))
        >>> titles = ['Total Potential', 'Secondary Potential']
        >>> for ax, V, title in zip(axs.flatten(), [vt, vs], titles):
        >>>     im = ax.pcolor(X, Y, V, shading='auto')
        >>>     divider = make_axes_locatable(ax)
        >>>     cax = divider.append_axes("right", size="5%", pad=0.05)
        >>>     cb = plt.colorbar(im, cax=cax)
        >>>     cb.set_label(label= 'Potential (A)')
        >>>     ax.add_patch(patches.Circle((0, 0), radius, fill=False, linestyle='--'))
        >>>     ax.set_ylabel('Y coordinate ($m$)')
        >>>     ax.set_xlabel('X coordinate ($m$)')
        >>>     ax.set_title(title)
        >>>     ax.set_aspect('equal')
        >>> plt.tight_layout()
        >>> plt.show()
        """
        xyz = check_xyz_dim(xyz)
        mu0 = self.mu_background
        mu1 = self.mu_sphere
        H0 = self.primary_field
        mu_cur = (mu1 - mu0) / (mu1 + 2 * mu0)
        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)

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
            If only requesting a single field.

        Examples
        --------
        Here, we define a sphere with permeability mu_sphere in a uniform magnetostatic field with permeability
        mu_background and plot the total and secondary magnetic fields.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib import patches
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> from geoana.em.static import MagnetostaticSphere

        Define the sphere.

        >>> mu_sphere = 10. ** -1
        >>> mu_background = 10. ** -3
        >>> radius = 1.0
        >>> simulation = MagnetostaticSphere(
        >>>     location=None, mu_sphere=mu_sphere, mu_background=mu_background, radius=radius, primary_field=None
        >>> )

        Now we create a set of gridded locations and compute the magnetic fields.

        >>> X, Y = np.meshgrid(np.linspace(-2*radius, 2*radius, 20), np.linspace(-2*radius, 2*radius, 20))
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
        >>>     cb.set_label(label= 'Amplitude ($A/m$)')
        >>>     ax.streamplot(X, Y, H[..., 0], H[..., 1], density=0.75)
        >>>     ax.add_patch(patches.Circle((0, 0), radius, fill=False, linestyle='--'))
        >>>     ax.set_ylabel('Y coordinate ($m$)')
        >>>     ax.set_xlabel('X coordinate ($m$)')
        >>>     ax.set_aspect('equal')
        >>>     ax.set_title(title)
        >>> plt.tight_layout()
        >>> plt.show()
        """
        xyz = check_xyz_dim(xyz)
        mu0 = self.mu_background
        mu1 = self.mu_sphere
        H0 = self.primary_field
        mu_cur = (mu1 - mu0) / (mu1 + 2 * mu0)
        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)

        Ht = np.zeros((*r.shape, 3))
        ind0 = r > self.radius

        # total field outside the sphere
        Ht[ind0] = H0 * (1. - mu_cur * self.radius ** 3. / r[ind0, None] ** 3) +\
            3. * (r_vec[ind0] @ H0)[..., None] * mu_cur * self.radius * r_vec[ind0] / r[ind0, None] ** 4

        # inside the sphere
        Ht[~ind0] = 3. * mu0 / (mu1 + 2. * mu0) * H0

        if field == 'total':
            return Ht

        if field != 'total':
            Hp = np.zeros((*r.shape, 3))
            Hp = Hp + H0
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
            If only requesting a single field.

        Examples
        --------
        Here, we define a sphere with permeability mu_sphere in a uniform magnetostatic field with permeability
        mu_background and plot the total and secondary magnetic flux densities.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib import patches
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> from geoana.em.static import MagnetostaticSphere

        Define the sphere.

        >>> mu_sphere = 10. ** -1
        >>> mu_background = 10. ** -3
        >>> radius = 1.0
        >>> simulation = MagnetostaticSphere(
        >>>     location=None, mu_sphere=mu_sphere, mu_background=mu_background, radius=radius, primary_field=None
        >>> )

        Now we create a set of gridded locations and compute the magnetic flux densities.

        >>> X, Y = np.meshgrid(np.linspace(-2*radius, 2*radius, 20), np.linspace(-2*radius, 2*radius, 20))
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
        >>>     cb.set_label(label= 'Amplitude (T)')
        >>>     ax.streamplot(X, Y, B[..., 0], B[..., 1], density=0.75)
        >>>     ax.add_patch(patches.Circle((0, 0), radius, fill=False, linestyle='--'))
        >>>     ax.set_ylabel('Y coordinate ($m$)')
        >>>     ax.set_xlabel('X coordinate ($m$)')
        >>>     ax.set_aspect('equal')
        >>>     ax.set_title(title)
        >>> plt.tight_layout()
        >>> plt.show()
        """

        xyz = check_xyz_dim(xyz)
        r = np.linalg.norm(xyz - self.location, axis=-1)

        mu = np.full(r.shape, self.mu_background)
        mu[r <= self.radius] = self.mu_sphere

        Ht = self.magnetic_field(xyz, field='total')
        Bt = mu[..., None] * Ht
        if field == 'total':
            return Bt

        if field != 'total':
            Hp = self.magnetic_field(xyz, field='primary')
            Bp = self.mu_background * Hp
            if field == 'primary':
                return Bp

        Bs = Bt - Bp
        if field == 'secondary':
            return Bs

        return Bt, Bp, Bs
