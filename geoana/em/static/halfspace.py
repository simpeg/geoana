import numpy as np

from geoana.em.static import PointCurrentWholeSpace
from geoana.utils import check_xyz_dim

__all__ = [
    "PointCurrentHalfSpace",
    "DipoleHalfSpace"
]


class PointCurrentHalfSpace:
    """Class for a point current in a halfspace.

    The ``PointCurrentHalfSpace`` class is used to analytically compute the
    potentials, current densities and electric fields within a halfspace due to a point current.
    Surface is assumed to be at z=0.

    Parameters
    ----------
    current : float
        Electrical current in the point current (A). Default is 1A.
    rho : float
        Resistivity in the point current (:math:`\\Omega \\cdot m`).
    location : array_like, optional
        Location at which we are observing in 3D space (m). Default is (0, 0, 0).
    """

    def __init__(self, rho, current=1.0, location=None):

        _primary = PointCurrentWholeSpace(rho, current=1.0, location=None)
        _image = PointCurrentWholeSpace(rho, current=1.0, location=None)
        self._primary = _primary
        self._image = _image

        self.current = current
        self.rho = rho
        if location is None:
            location = np.r_[0, 0, 0]
        self.location = location

    @property
    def current(self):
        """Current in the point current in Amps.

        Returns
        -------
        float
            Current in the point current in Amps.
        """
        return self._current

    @current.setter
    def current(self, value):

        try:
            value = float(value)
        except:
            raise TypeError(f"current must be a number, got {type(value)}")

        self._current = value
        self._primary.current = value
        self._image.current = value

    @property
    def rho(self):
        """Resistivity in the point current in :math:`\\Omega \\cdot m`.

        Returns
        -------
        float
            Resistivity in the point current in :math:`\\Omega \\cdot m`.
        """
        return self._rho

    @rho.setter
    def rho(self, value):

        try:
            value = float(value)
        except:
            raise TypeError(f"current must be a number, got {type(value)}")

        if value <= 0.0:
            raise ValueError("current must be greater than 0")

        self._rho = value
        self._primary.rho = value
        self._image.rho = value

    @property
    def location(self):
        """Location of observer in 3D space.

        Returns
        -------
        (3) numpy.ndarray of float
            Location of observer in 3D space. Default = np.r_[0,0,0].
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

        if np.any(vec[..., -1] > 0):
            raise ValueError(
                f"z value must be less than or equal to 0 in a halfspace, got {(vec[..., -1])}"
            )

        self._location = vec
        self._primary.location = vec

        vec = np.copy(vec)
        vec[-1] *= -1
        self._image.location = vec

    def potential(self, xyz):
        """Electric potential for a point current in a halfspace.

        This method computes the potential for the point current in a halfspace at
        the set of gridded xyz locations provided. Where :math:`\\rho` is the
        electric resistivity, I is the current and R is the distance between
        the location we want to evaluate at and the point current.
        The potential V is:

        .. math::

            V = \\frac{\\rho I}{2 \\pi R}

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Locations to evaluate at in units m.

        Returns
        -------
        V : (..., ) np.ndarray
            Electric potential of point current in units V.

        Examples
        --------
        Here, we define a point current with current=1A in a halfspace and plot the electric
        potential.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from geoana.em.static import PointCurrentHalfSpace

        Define the point current.

        >>> rho = 1.0
        >>> current = 1.0
        >>> simulation = PointCurrentHalfSpace(
        >>>     current=current, rho=rho, location=None
        >>> )

        Now we create a set of gridded locations and compute the electric potential.

        >>> X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        >>> Z = np.zeros_like(X)
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> v = simulation.potential(xyz)

        Finally, we plot the electric potential.

        >>> plt.pcolor(X, Y, v)
        >>> cb1 = plt.colorbar()
        >>> cb1.set_label(label= 'Potential (V)')
        >>> plt.xlabel('x')
        >>> plt.ylabel('y')
        >>> plt.title('Electric Potential from Point Current in a Halfspace')
        >>> plt.show()
        """

        xyz = check_xyz_dim(xyz)
        if np.any(xyz[..., -1] > 0):
            raise ValueError(
                f"z value must be less than or equal to 0 in a halfspace, got {(xyz[..., -1])}"
            )

        v = self._primary.potential(xyz) + self._image.potential(xyz)
        return v

    def electric_field(self, xyz):
        """Electric field for a point current in a halfspace.

        This method computes the electric field for the point current in a halfspace at
        the set of gridded xyz locations provided. Where :math:`- \\nabla V`
        is the negative gradient of the electric potential for the point current.
        The electric field :math:`\\mathbf{E}` is:

       .. math::

            \\mathbf{E} = -\\nabla V

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Locations to evaluate at in units m.

        Returns
        -------
        E : (..., 3) np.ndarray
            Electric field of point current in units :math:`\\frac{V}{m}`.

        Examples
        --------
        Here, we define a point current with current=1A in a halfspace and plot the electric
        field lines in the xy-plane.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from geoana.em.static import PointCurrentHalfSpace

        Define the point current.

        >>> rho = 1.0
        >>> current = 1.0
        >>> simulation = PointCurrentHalfSpace(
        >>>     current=current, rho=rho, location=None
        >>> )

        Now we create a set of gridded locations and compute the electric field.

        >>> X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        >>> Z = np.zeros_like(X)
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> e = simulation.electric_field(xyz)

        Finally, we plot the electric field lines.

        >>> e_amp = np.linalg.norm(e, axis=-1)
        >>> plt.pcolor(X, Y, e_amp)
        >>> cb = plt.colorbar()
        >>> cb.set_label(label= 'Amplitude ($V/m$)')
        >>> plt.streamplot(X, Y, e[..., 0], e[..., 1], density=0.50)
        >>> plt.xlabel('x')
        >>> plt.ylabel('y')
        >>> plt.title('Electric Field Lines for a Point Current in a Halfspace')
        >>> plt.show()
        """

        xyz = check_xyz_dim(xyz)
        if np.any(xyz[..., -1] > 0):
            raise ValueError(
                f"z value must be less than or equal to 0 in a halfspace, got {(xyz[..., -1])}"
            )

        e = self._primary.electric_field(xyz) + self._image.electric_field(xyz)
        return e

    def current_density(self, xyz):
        """Current density for a point current in a halfspace.

       This method computes the current density for the point current in a halfspace at
        the set of gridded xyz locations provided. Where :math:`\\rho`
        is the electric resistivity and :math:`\\mathbf{E}` is the electric field
        for the point current.
        The current density :math:`\\mathbf{J}` is:

       .. math::

            \\mathbf{J} = \\frac{\\mathbf{E}}{\\rho}

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Locations to evaluate at in units m.

        Returns
        -------
        J : (..., 3) np.ndarray
            Current density of point current in units :math:`\\frac{A}{m^2}`.

        Examples
        --------
        Here, we define a point current with current=1A in a halfspace and plot the current density.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from geoana.em.static import PointCurrentHalfSpace

        Define the point current.

        >>> rho = 1.0
        >>> current = 1.0
        >>> simulation = PointCurrentHalfSpace(
        >>>     current=current, rho=rho, location=None
        >>> )

        Now we create a set of gridded locations and compute the current density.

        >>> X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        >>> Z = np.zeros_like(X)
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> j = simulation.current_density(xyz)

        Finally, we plot the curent density.

        >>> j_amp = np.linalg.norm(j, axis=-1)
        >>> plt.pcolor(X, Y, j_amp, shading='auto')
        >>> cb1 = plt.colorbar()
        >>> cb1.set_label(label= 'Current Density ($A/m^2$)')
        >>> plt.ylabel('Y coordinate ($m$)')
        >>> plt.xlabel('X coordinate ($m$)')
        >>> plt.title('Current Density for a Point Current in a Halfspace')
        >>> plt.show()
        """

        j = self.electric_field(xyz) / self.rho
        return j


class DipoleHalfSpace:
    """Class for a dipole in a halfspace.

        The ``DipoleHalfSpace`` class is used to analytically compute the
        potentials, current densities and electric fields within a halfspace due to a dipole source.
        Surface is assumed to be at z=0.

        Parameters
        ----------
        current : float
            Electrical current in the point current (A). Default is 1A.
        rho : float
            Resistivity in the point current (:math:`\\Omega \\cdot m`).
        location_a : array_like
            Location of the A current source electrode (m). Default is (-1, 0, 0).
        location_b : array_like
            Location of the B current sink electrode (m). Default is (1, 0, 0).
        """

    def __init__(self, rho, location_a=None, location_b=None, current=1.0):

        _a = PointCurrentHalfSpace(rho, current=1.0, location=location_a)
        _b = PointCurrentHalfSpace(rho, current=1.0, location=location_b)
        self._a = _a
        self._b = _b

        self.current = current
        self.rho = rho
        if location_a is None:
            location_a = np.r_[-1, 0, 0]
        self.location_a = location_a
        if location_b is None:
            location_b = np.r_[1, 0, 0]
        self.location_b = location_b

    @property
    def current(self):
        """Current in the dipole in Amps.

        Returns
        -------
        float
            Current in the dipole in Amps.
        """
        return self._current

    @current.setter
    def current(self, value):

        try:
            value = float(value)
        except:
            raise TypeError(f"current must be a number, got {type(value)}")

        self._current = value
        self._a.current = value
        self._b.current = value

    @property
    def rho(self):
        """Resistivity in the dipole in :math:`\\Omega \\cdot m`.

        Returns
        -------
        float
            Resistivity in the dipole in :math:`\\Omega \\cdot m`.
        """
        return self._rho

    @rho.setter
    def rho(self, value):

        try:
            value = float(value)
        except:
            raise TypeError(f"current must be a number, got {type(value)}")

        if value <= 0.0:
            raise ValueError("current must be greater than 0")

        self._rho = value
        self._a.rho = value
        self._b.rho = value

    @property
    def location_a(self):
        """Location of the A current source electrode.

        Returns
        -------
        (3) numpy.ndarray of float
            Location of the A current source electrode. Default = np.r_[-1,0,0].
        """
        return self._location_a

    @location_a.setter
    def location_a(self, vec):

        try:
            vec = np.atleast_1d(vec).astype(float)
        except:
            raise TypeError(f"location must be array_like, got {type(vec)}")

        if len(vec) != 3:
            raise ValueError(
                f"location must be array_like with shape (3,), got {len(vec)}"
            )

        if np.any(vec[..., -1] > 0):
            raise ValueError(
                f"z value must be less than or equal to 0 in a halfspace, got {(vec[..., -1])}"
            )

        self._location_a = vec
        self._a.location = vec

    @property
    def location_b(self):
        """Location of the B current sink electrode.

        Returns
        -------
        (3) numpy.ndarray of float
            Location of the B current sink electrode. Default = np.r_[1,0,0].
        """
        return self._location_b

    @location_b.setter
    def location_b(self, vec):

        try:
            vec = np.atleast_1d(vec).astype(float)
        except:
            raise TypeError(f"location must be array_like, got {type(vec)}")

        if len(vec) != 3:
            raise ValueError(
                f"location must be array_like with shape (3,), got {len(vec)}"
            )

        if np.any(vec[..., -1] > 0):
            raise ValueError(
                f"z value must be less than or equal to 0 in a halfspace, got {(vec[..., -1])}"
            )

        self._location_b = vec
        self._b.location = vec

    def potential(self, xyz_m, xyz_n=None):
        """Electric potential for a dipole in a halfspace.

        This method computes the potential for a dipole in a halfspace at
        the set of gridded xyz locations provided. Where :math:`\\rho` is the
        electric resistivity, I is the current and R is the distance between
        the locations we want to evaluate at and the dipole source.

        Parameters
        ----------
        xyz_m : (..., 3) numpy.ndarray
            Location of the M voltage electrode.
        xyz_n : (..., 3) numpy.ndarray, optional
            Location of the N voltage electrode.

        Returns
        -------
        V : (..., ) np.ndarray
            Electric potential of dipole source in units V.

        Examples
        --------
        Here, we define a dipole source in a halfspace to compute potential.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> from geoana.em.static import DipoleHalfSpace

        Define the dipole source.

        >>> rho = 1.0
        >>> current = 1.0
        >>> location_a = np.r_[-1, 0, 0]
        >>> location_b = np.r_[1, 0, 0]
        >>> simulation = DipoleHalfSpace(
        >>>     current=current, rho=rho, location_a=location_a, location_b=location_b
        >>> )

        Now we create a set of gridded locations and compute the electric potential.

        >>> X, Y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
        >>> Z = np.zeros_like(X)
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> v1 = simulation.potential(xyz)
        >>> v2 = simulation.potential(xyz - np.r_[2, 0, 0], xyz + np.r_[2, 0, 0])

        Finally, we plot the electric potential.

        >>> fig, axs = plt.subplots(1, 2, figsize=(18,12))
        >>> titles = ['3 Electrodes', '4 Electrodes']
        >>> for ax, V, title in zip(axs.flatten(), [v1, v2], titles):
        >>>     im = ax.pcolor(X, Y, V, shading='auto')
        >>>     divider = make_axes_locatable(ax)
        >>>     cax = divider.append_axes("right", size="5%", pad=0.05)
        >>>     cb = plt.colorbar(im, cax=cax)
        >>>     cb.set_label(label= 'Potential (V)')
        >>>     ax.set_ylabel('Y coordinate ($m$)')
        >>>     ax.set_xlabel('X coordinate ($m$)')
        >>>     ax.set_aspect('equal')
        >>>     ax.set_title(title)
        >>> plt.tight_layout()
        >>> plt.show()
        """

        xyz_m = check_xyz_dim(xyz_m)
        if np.any(xyz_m[..., -1] > 0):
            raise ValueError(
                f"z value must be less than or equal to 0 in a halfspace, got {(xyz_m[..., -1])}"
            )

        if xyz_n is not None:
            xyz_n = check_xyz_dim(xyz_n)
            if np.any(xyz_n[..., -1] > 0):
                raise ValueError(
                    f"z value must be less than or equal to 0 in a halfspace, got {(xyz_n[..., -1])}"
                )

        vm = self._a.potential(xyz_m) - self._b.potential(xyz_m)

        if xyz_n is not None:
            vn = self._a.potential(xyz_n) - self._b.potential(xyz_n)
            v = vm - vn
            return v
        else:
            return vm

    def electric_field(self, xyz_m, xyz_n=None):
        """Electric field for a dipole source in a halfspace.

        This method computes the electric field for a dipole source in a halfspace at
        the set of gridded xyz locations provided. Where :math:`-\\nabla V`
        is the negative gradient of the electric potential for a dipole source.
        The electric field :math:`\\mathbf{E}` is:

       .. math::

            \\mathbf{E} = -\\nabla V

        Parameters
        ----------
        xyz_m : (..., 3) numpy.ndarray
            Location of the M voltage electrode.
        xyz_n : (..., 3) numpy.ndarray, optional
            Location of the N voltage electrode.

        Returns
        -------
        E : (..., 3) np.ndarray
            Electric field of point current in units :math:`\\frac{V}{m}`.

        Examples
        --------
        Here, we define a dipole source in a halfspace to compute electric field.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> from geoana.em.static import DipoleHalfSpace

        Define the dipole source.

        >>> rho = 1.0
        >>> current = 1.0
        >>> location_a = np.r_[-1, 0, 0]
        >>> location_b = np.r_[1, 0, 0]
        >>> simulation = DipoleHalfSpace(
        >>>     current=current, rho=rho, location_a=location_a, location_b=location_b
        >>> )

        Now we create a set of gridded locations and compute the electric field.

        >>> X, Y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
        >>> Z = np.zeros_like(X)
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> e1 = simulation.electric_field(xyz)
        >>> e2 = simulation.electric_field(xyz - np.r_[2, 0, 0], xyz + np.r_[2, 0, 0])

        Finally, we plot the electric field.

        >>> fig, axs = plt.subplots(1, 2, figsize=(18,12))
        >>> titles = ['3 Electrodes', '4 Electrodes']
        >>> for ax, E, title in zip(axs.flatten(), [e1, e2], titles):
        >>>     E_amp = np.linalg.norm(E, axis=-1)
        >>>     im = ax.pcolor(X, Y, E_amp, shading='auto')
        >>>     divider = make_axes_locatable(ax)
        >>>     cax = divider.append_axes("right", size="5%", pad=0.05)
        >>>     cb = plt.colorbar(im, cax=cax)
        >>>     cb.set_label(label= 'Amplitude ($V/m$)')
        >>>     ax.streamplot(X, Y, E[..., 0], E[..., 1], density=0.75)
        >>>     ax.set_ylabel('Y coordinate ($m$)')
        >>>     ax.set_xlabel('X coordinate ($m$)')
        >>>     ax.set_aspect('equal')
        >>>     ax.set_title(title)


        Finally, we plot the electric field.

        >>> E_amp = np.linalg.norm(e1, axis=-1)
        >>> plt.pcolor(X, Y, E_amp, shading='auto')
        >>> cb = plt.colorbar()
        >>> cb.set_label(label= 'Electric Field ($V/m$)')
        >>> plt.streamplot(X, Y, e1[..., 0], e1[..., 1], density=0.75)
        >>> plt.ylabel('Y coordinate ($m$)')
        >>> plt.xlabel('X coordinate ($m$)')
        >>> plt.title('Electric Field from Dipole using 3 Electrodes')

        >>> plt.tight_layout()
        >>> plt.show()
        """

        xyz_m = check_xyz_dim(xyz_m)
        if np.any(xyz_m[..., -1] > 0):
            raise ValueError(
                f"z value must be less than or equal to 0 in a halfspace, got {(xyz_m[..., -1])}"
            )

        if xyz_n is not None:
            xyz_n = check_xyz_dim(xyz_n)
            if np.any(xyz_n[..., -1] > 0):
                raise ValueError(
                    f"z value must be less than or equal to 0 in a halfspace, got {(xyz_n[..., -1])}"
                )

        em = self._a.electric_field(xyz_m) - self._b.electric_field(xyz_m)

        if xyz_n is not None:
            en = self._a.electric_field(xyz_n) - self._b.electric_field(xyz_n)
            e = em - en
            return e
        else:
            return em

    def current_density(self, xyz_m, xyz_n=None):
        """Current density for a dipole source in a halfspace.

       This method computes the current density for a dipole source in a halfspace at
        the set of gridded xyz locations provided. Where :math:`\\rho`
        is the electric resistivity and :math:`\\mathbf{E}` is the electric field
        for the dipole source.
        The current density :math:`\\mathbf{J}` is:

       .. math::

            \\mathbf{J} = \\frac{\\mathbf{E}}{\\rho}

        Parameters
        ----------
        xyz_m : (..., 3) numpy.ndarray
            Location of the M voltage electrode.
        xyz_n : (..., 3) numpy.ndarray, optional
            Location of the N voltage electrode.

        Returns
        -------
        J : (..., 3) np.ndarray
            Current density of point current in units :math:`\\frac{A}{m^2}`.

        Examples
        --------
        Here, we define a dipole source in a halfspace to compute current density.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
        >>> from geoana.em.static import DipoleHalfSpace

        Define the dipole source.

        >>> rho = 1.0
        >>> current = 1.0
        >>> location_a = np.r_[-1, 0, 0]
        >>> location_b = np.r_[1, 0, 0]
        >>> simulation = DipoleHalfSpace(
        >>>     current=current, rho=rho, location_a=location_a, location_b=location_b
        >>> )

        Now we create a set of gridded locations and compute the current density.

        >>> X, Y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
        >>> Z = np.zeros_like(X)
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> j1 = simulation.current_density(xyz)
        >>> j2 = simulation.current_density(xyz - np.r_[2, 0, 0], xyz + np.r_[2, 0, 0])

        Finally, we plot the current density.

        >>> fig, axs = plt.subplots(1, 2, figsize=(18,12))
        >>> titles = ['3 Electrodes', '4 Electrodes']
        >>> for ax, J, title in zip(axs.flatten(), [j1, j2], titles):
        >>>     J_amp = np.linalg.norm(J, axis=-1)
        >>>     im = ax.pcolor(X, Y, J_amp, shading='auto')
        >>>     divider = make_axes_locatable(ax)
        >>>     cax = divider.append_axes("right", size="5%", pad=0.05)
        >>>     cb = plt.colorbar(im, cax=cax)
        >>>     cb.set_label(label= 'Current Density ($A/m^2$)')
        >>>     ax.streamplot(X, Y, J[..., 0], J[..., 1], density=0.75)
        >>>     ax.set_ylabel('Y coordinate ($m$)')
        >>>     ax.set_xlabel('X coordinate ($m$)')
        >>>     ax.set_aspect('equal')
        >>>     ax.set_title(title)

        Finally, we plot the current density.

        >>> J_amp = np.linalg.norm(j1, axis=-1)
        >>> plt.pcolor(X, Y, J_amp, shading='auto')
        >>> cb = plt.colorbar()
        >>> cb.set_label(label= 'Current Density ($A/m^2$)')
        >>> plt.streamplot(X, Y, j1[..., 0], j1[..., 1], density=0.75)
        >>> plt.ylabel('Y coordinate ($m$)')
        >>> plt.xlabel('X coordinate ($m$)')
        >>> plt.title('Current Density from Dipole using 3 Electrodes')
        >>> plt.tight_layout()
        >>> plt.show()
        """

        j = self.electric_field(xyz_m, xyz_n=xyz_n) / self.rho
        return j

