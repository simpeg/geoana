import numpy as np

__all__ = [
    "PointCurrentHalfSpace", "PointCurrentFourElectrodeArray"
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
                f"z value must be less than or equal to 0 in a halfspace, got {(vec[2])}"
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

    def potential(self, xyz):
        """Electric potential for a point current in a halfspace.

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
        potential as a function of distance.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from geoana.em.static import PointCurrentWholeSpace

        Define the point current.

        >>> rho = 1.0
        >>> current = 1.0
        >>> simulation = PointCurrentWholeSpace(
        >>>     current=current, rho=rho, location=None,
        >>> )

        Now we create a set of gridded locations, take the distances and compute the electric potential.

        >>> X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> r = np.linalg.norm(xyz, axis=-1)
        >>> v = simulation.potential(xyz)

        Finally, we plot the electric potential as a function of distance.

        >>> plt.plot(r, v)
        >>> plt.xlabel('Distance from point current')
        >>> plt.ylabel('Electric potential')
        >>> plt.title('Electric Potential as a function of distance from Point Current in a Halfspace')
        >>> plt.show()
        """

        x0, y0, z0 = self.location
        x, y, z = self._check_XYZ(xyz)
        x = x - x0
        y = y - y0
        z = z - z0
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if np.any(xyz[..., -1] > 0):
            raise ValueError(
                f"z value must be less than or equal to 0 in a halfspace, got {(xyz[2])}"
            )

        v = self.rho * self.current / (2 * np.pi * r)
        return v

    def electric_field(self, xyz):
        """Electric field for a point current in a halfspace.

       .. math::

            \\mathbf{E} = -\\nabla V

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Locations to evaluate at in units m.

        Returns
        -------
        E : (..., 3) np.ndarray
            Electric field of point current in units :math:`\\frac{V}{m^2}`.

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
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> e = simulation.electric_field(xyz)

        Finally, we plot the electric field lines.

        >>> plt.quiver(X, Y, e[:,:,0], e[:,:,1])
        >>> plt.xlabel('x')
        >>> plt.ylabel('y')
        >>> plt.title('Electric Field Lines for a Point Current in a Halfspace')
        >>> plt.show()
        """

        x0, y0, z0 = self.location
        x, y, z = self._check_XYZ(xyz)
        x = x - x0
        y = y - y0
        z = z - z0
        r_vec = xyz - self.location
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if np.any(xyz[..., -1] > 0):
            raise ValueError(
                f"z value must be less than or equal to 0 in a halfspace, got {(xyz[2])}"
            )

        e = self.rho * self.current * r_vec / (2 * np.pi * r[..., None] ** 3)
        return e

    def current_density(self, xyz):
        """Current density for a point current in a halfspace.

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
        >>> Z = np.zeros_like(X) + 0.25
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


class PointCurrentFourElectrodeArray:
    """Class for a point current in a four electrode array.

    The ``PointCurrentFourElectrodeArray`` class is used to analytically compute the
    potentials, current densities and electric fields within a four electrode array in a halfspace
    due to a point current.

    Parameters
    ----------
    current : float
        Electrical current in the point current (A). Default is 1A.
    rho : float
        Resistivity in the point current (:math:`\\Omega \\cdot m`).
    location : array_like, optional
        Location at which we are observing in 3D space (m). Default is (0, 0, 0).
    """