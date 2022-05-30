import numpy as np

from geoana.em.static import PointCurrentWholeSpace
from geoana.utils import check_xyz_dim

__all__ = [
    "PointCurrentHalfSpace",
    "electrode_array_potential"
]


def electrode_array_potential(xyz1, xyz2, rho, current, location1, location2):
    """Potential for a four electrode array.  Surface is assumed to be at z=0.

    Parameters
    ----------
    xyz1 : (..., 3) numpy.ndarray
        First location to evaluate at in units m.
    xyz2 : (..., 3) numpy.ndarray
        Second location to evaluate at in units m.
    rho : Resistivity in the electrode array in :math:`\\Omega \\cdot m`.
    current : Current in the electrode array in A.
    location1 : Location of first electrode in 3D space in m.
    location2 : Location of second electrode in 3D space in m.

    Returns
    -------
    V : (..., 3) np.ndarray
        Potential of four electrode array in units V.

    Examples
    --------
    Here, we define two point current with current=1A and current=-1A in a halfspace and plot the electric
    potential in the four electrode array.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from geoana.em.static import PointCurrentHalfSpace
    >>> from geoana.em.static import electrode_array_potential

    Define the point currents.

    >>> rho = 1.0
    >>> current = 1.0
    >>> location1 = np.r_[1., 1., -1.]
    >>> location2 = np.r_[2., 2., -1.]

    Now we create a set of gridded locations and compute the electric potential.

    >>> X1, Y1 = np.meshgrid(np.linspace(-1, 3, 20), np.linspace(-1, 3, 20))
    >>> Z1 = np.zeros_like(X1)
    >>> xyz1 = np.stack((X1, Y1, Z1), axis=-1)
    >>> X2, Y2 = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
    >>> Z2 = np.zeros_like(X1)
    >>> xyz2 = np.stack((X2, Y2, Z2), axis=-1)
    >>> v = electrode_array_potential(xyz1, xyz2, rho, current, location1, location2)

    Finally, we plot the electric potential.

    >>> plt.pcolor(X1, Y1, v)
    >>> cb1 = plt.colorbar()
    >>> cb1.set_label(label= 'Potential (V)')
    >>> plt.xlabel('x')
    >>> plt.ylabel('y')
    >>> plt.title('Electric Potential of an electrode array in a Halfspace')
    >>> plt.show()
    """

    xyz1 = check_xyz_dim(xyz1)
    xyz2 = check_xyz_dim(xyz2)
    if np.any(xyz1[..., -1] > 0):
        raise ValueError(
            f"z value must be less than or equal to 0 in a halfspace, got {(xyz1[..., -1])}"
        )
    if np.any(xyz2[..., -1] > 0):
        raise ValueError(
            f"z value must be less than or equal to 0 in a halfspace, got {(xyz2[..., -1])}"
        )
    if np.any(location1[..., -1] > 0):
        raise ValueError(
            f"z value must be less than or equal to 0 in a halfspace, got {(location1[..., -1])}"
        )
    if np.any(location2[..., -1] > 0):
        raise ValueError(
            f"z value must be less than or equal to 0 in a halfspace, got {(location2[..., -1])}"
        )

    a = PointCurrentHalfSpace(rho=rho, current=current, location=location1)
    b = PointCurrentHalfSpace(rho=rho, current=-current, location=location2)
    v = a.potential(xyz1) - a.potential(xyz2) - b.potential(xyz1) + b.potential(xyz2)
    return v


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
        >>>     current=current, rho=rho, location=np.r_[1, 1, -1]
        >>> )

        Now we create a set of gridded locations and compute the electric potential.

        >>> X, Y = np.meshgrid(np.linspace(-1, 3, 20), np.linspace(-1, 3, 20))
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
        >>>     current=current, rho=rho, location=np.r_[1, 1, -1]
        >>> )

        Now we create a set of gridded locations and compute the electric field.

        >>> X, Y = np.meshgrid(np.linspace(-1, 3, 20), np.linspace(-1, 3, 20))
        >>> Z = np.zeros_like(X)
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> e = simulation.electric_field(xyz)

        Finally, we plot the electric field lines.

        >>> plt.quiver(X, Y, e[:,:,0], e[:,:,1])
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

        v = self._primary.electric_field(xyz) + self._image.electric_field(xyz)
        return v

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
        >>>     current=current, rho=rho, location=np.r_[1, 1, -1]
        >>> )

        Now we create a set of gridded locations and compute the current density.

        >>> X, Y = np.meshgrid(np.linspace(-1, 3, 20), np.linspace(-1, 3, 20))
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
