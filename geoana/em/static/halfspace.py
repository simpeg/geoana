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
        locations : array_like
            Locations of the M and N electrodes (m).
        """

    def __init__(self, rho, locations, current=1.0):

        _primary = PointCurrentHalfSpace(rho, current=1.0, location=locations[0])
        _secondary = PointCurrentHalfSpace(rho, current=-1.0, location=locations[1])
        self._primary = _primary
        self._secondary = _secondary

        self.current = current
        self.rho = rho
        self.locations = locations

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
        self._secondary.current = -value

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
        self._secondary.rho = value

    @property
    def locations(self):
        """Locations of the two voltage electrodes.

        Returns
        -------
        (2, 3) numpy.ndarray of float
            Locations of the two voltage electrodes.
        """
        return self._locations

    @locations.setter
    def locations(self, vec):

        try:
            vec = np.atleast_1d(vec).astype(float)
        except:
            raise TypeError(f"location must be array_like, got {type(vec)}")

        if np.any(vec[..., -1] > 0):
            raise ValueError(
                f"z value must be less than or equal to 0 in a halfspace, got {(vec[..., -1])}"
            )

        self._locations = vec
        self._primary.location = vec[0]
        self._secondary.location = vec[1]

    def potential(self, xyz):
        """Electric potential for a dipole in a halfspace.

        This method computes the potential for a dipole in a halfspace at
        the set of gridded xyz locations provided. Where :math:`\\rho` is the
        electric resistivity, I is the current and R is the distance between
        the location we want to evaluate at and the point current.

        Parameters
        ----------
        xyz : (2, ..., 3) numpy.ndarray
            Locations of the A and B electrodes.

        Returns
        -------
        V : (..., ) np.ndarray
            Electric potential of dipole source in units V.

        Examples
        --------
        Here, we define a dipole source in a halfspace to compute potential.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from geoana.em.static import DipoleHalfSpace

        Define the dipole source.

        >>> rho = 1.0
        >>> current = 1.0
        >>> simulation = DipoleHalfSpace(
        >>>     current=current, rho=rho, locations=np.array([np.r_[1, 1, -1], np.r_[0, 0, -1]])
        >>> )

        Now we create a set of gridded locations and compute the electric potential.

        >>> X1, Y1 = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        >>> X2, Y2 = np.meshgrid(np.linspace(-2, 1, 20), np.linspace(-1, 2, 20))
        >>> Z = np.zeros_like(X1)
        >>> xyz1 = np.stack((X1, Y1, Z), axis=-1)
        >>> xyz2 = np.stack((X2, Y2, Z), axis=-1)
        >>> xyz = np.array([xyz1, xyz2])
        >>> v = simulation.potential(xyz)

        Finally, we plot the electric potential.

        >>> plt.pcolor(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20), v)
        >>> cb1 = plt.colorbar()
        >>> cb1.set_label(label= 'Potential (V)')
        >>> plt.xlabel('x')
        >>> plt.ylabel('y')
        >>> plt.title('Electric Potential from Dipole Source in a Halfspace')
        >>> plt.show()
        """

        xyz = check_xyz_dim(xyz)
        if np.any(xyz[..., -1] > 0):
            raise ValueError(
                f"z value must be less than or equal to 0 in a halfspace, got {(xyz[..., -1])}"
            )

        vm = self._primary.potential(xyz[0]) - self._secondary.potential(xyz[0])
        vn = self._primary.potential(xyz[1]) + self._secondary.potential(xyz[1])
        v = vm - vn
        return v

    def electric_field(self, xyz):
        """Electric field for a dipole source in a halfspace.

        This method computes the electric field for a dipole source in a halfspace at
        the set of gridded xyz locations provided. Where :math:`- \\nabla V`
        is the negative gradient of the electric potential for a dipole source.
        The electric field :math:`\\mathbf{E}` is:

       .. math::

            \\mathbf{E} = -\\nabla V

        Parameters
        ----------
        xyz : (2, ..., 3) numpy.ndarray
            Locations to evaluate at in units m.

        Returns
        -------
        E : (..., 3) np.ndarray
            Electric field of point current in units :math:`\\frac{V}{m}`.

        Examples
        --------
        Here, we define a dipole source in a halfspace to compute electric field.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from geoana.em.static import DipoleHalfSpace

        Define the dipole source.

        >>> rho = 1.0
        >>> current = 1.0
        >>> simulation = DipoleHalfSpace(
        >>>     current=current, rho=rho, locations=np.array([np.r_[1, 1, -1], np.r_[0, 0, -1]])
        >>> )

        Now we create a set of gridded locations and compute the electric field.

        >>> X1, Y1 = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        >>> X2, Y2 = np.meshgrid(np.linspace(-2, 1, 20), np.linspace(-1, 2, 20))
        >>> Z = np.zeros_like(X1)
        >>> xyz1 = np.stack((X1, Y1, Z), axis=-1)
        >>> xyz2 = np.stack((X2, Y2, Z), axis=-1)
        >>> xyz = np.array([xyz1, xyz2])
        >>> e = simulation.electric_field(xyz)

        Finally, we plot the electric field.

        >>> e_amp = np.linalg.norm(e, axis=-1)
        >>> plt.pcolor(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20), e_amp)
        >>> cb1 = plt.colorbar()
        >>> cb1.set_label(label= 'Amplitude ($V/m$)')
        >>> plt.xlabel('x')
        >>> plt.ylabel('y')
        >>> plt.title('Electric Field from Dipole Source in a Halfspace')
        >>> plt.show()
        """

        xyz = check_xyz_dim(xyz)
        if np.any(xyz[..., -1] > 0):
            raise ValueError(
                f"z value must be less than or equal to 0 in a halfspace, got {(xyz[..., -1])}"
            )

        em = self._primary.electric_field(xyz[0]) - self._secondary.electric_field(xyz[0])
        en = self._primary.electric_field(xyz[1]) + self._secondary.electric_field(xyz[1])
        e = em - en
        return e

    def current_density(self, xyz):
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
        xyz : (2, ..., 3) numpy.ndarray
            Locations to evaluate at in units m.

        Returns
        -------
        J : (..., 3) np.ndarray
            Current density of point current in units :math:`\\frac{A}{m^2}`.

        Examples
        --------
        Here, we define a dipole source in a halfspace to compute current density.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from geoana.em.static import DipoleHalfSpace

        Define the dipole source.

        >>> rho = 1.0
        >>> current = 1.0
        >>> simulation = DipoleHalfSpace(
        >>>     current=current, rho=rho, locations=np.array([np.r_[1, 1, -1], np.r_[0, 0, -1]])
        >>> )

        Now we create a set of gridded locations and compute the current density.

        >>> X1, Y1 = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        >>> X2, Y2 = np.meshgrid(np.linspace(-2, 1, 20), np.linspace(-1, 2, 20))
        >>> Z = np.zeros_like(X1)
        >>> xyz1 = np.stack((X1, Y1, Z), axis=-1)
        >>> xyz2 = np.stack((X2, Y2, Z), axis=-1)
        >>> xyz = np.array([xyz1, xyz2])
        >>> j = simulation.current_density(xyz)

        Finally, we plot the current density.

        >>> j_amp = np.linalg.norm(j, axis=-1)
        >>> plt.pcolor(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20), j_amp)
        >>> cb1 = plt.colorbar()
        >>> cb1.set_label(label= 'Current Density ($A/m^2$)')
        >>> plt.xlabel('x')
        >>> plt.ylabel('y')
        >>> plt.title('Current Density from Dipole Source in a Halfspace')
        >>> plt.show()
        """

        j = self.electric_field(xyz) / self.rho
        return j
