import numpy as np
from scipy.special import ellipk, ellipe

from ..base import BaseDipole, BaseMagneticDipole, BaseEM
from ... import spatial
from geoana.utils import check_xyz_dim

__all__ = [
    "MagneticDipoleWholeSpace", "CircularLoopWholeSpace",
    "MagneticPoleWholeSpace", "PointCurrentWholeSpace"
]


class MagneticDipoleWholeSpace(BaseEM, BaseMagneticDipole):
    """Class for a static magnetic dipole in a wholespace.

    The ``MagneticDipoleWholeSpace`` class is used to analytically compute the
    fields and potentials within a wholespace due to a static magnetic dipole.

    """

    def vector_potential(self, xyz, coordinates="cartesian"):
        r"""Compute the vector potential for the static magnetic dipole.

        This method computes the vector potential for the magnetic dipole at
        the set of gridded xyz locations provided. Where :math:`\mu` is the
        magnetic permeability, :math:`\mathbf{m}` is the dipole moment,
        :math:`\mathbf{r_0}` the dipole location and :math:`\mathbf{r}`
        is the location at which we want to evaluate
        the vector potential :math:`\mathbf{a}`:

        .. math::

            \mathbf{a}(\mathbf{r}) = \frac{\mu}{4\pi}
            \frac{\mathbf{m} \times \, \Delta \mathbf{r}}{| \Delta r |^3}

        where

        .. math::
            \mathbf{\Delta r} = \mathbf{r} - \mathbf{r_0}

        For reference, see equation 5.83 in Griffiths (1999).

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray xyz
            gridded locations at which we are calculating the vector potential
        coordinates: str {'cartesian', 'cylindrical'}
            coordinate system that the location (xyz) are provided.
            The solution is also returned in this coordinate system.
            Default: `"cartesian"`

        Returns
        -------
        (n, 3) numpy.ndarray
            The magnetic vector potential at each observation location in the
            coordinate system specified in units *Tm*.

        Examples
        --------
        Here, we define a z-oriented magnetic dipole and plot the vector
        potential on the xy-plane that intercepts at z=0.

        >>> from geoana.em.static import MagneticDipoleWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the magnetic dipole.

        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[0., 0., 1.]
        >>> moment = 1.
        >>> dipole_object = MagneticDipoleWholeSpace(
        >>>     location=location, orientation=orientation, moment=moment
        >>> )

        Now we create a set of gridded locations and compute the vector potential.

        >>> xyz = ndgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20), np.array([0]))
        >>> a = dipole_object.vector_potential(xyz)

        Finally, we plot the vector potential on the plane. Given the symmetry,
        there are only horizontal components.

        >>> fig = plt.figure(figsize=(4, 4))
        >>> ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        >>> plot2Ddata(xyz[:, 0:2], a[:, 0:2], ax=ax, vec=True, scale='log')
        >>> ax.set_xlabel('X')
        >>> ax.set_ylabel('Z')
        >>> ax.set_title('Vector potential at z=0')

        """
        supported_coordinates = ["cartesian", "cylindrical"]
        assert coordinates.lower() in supported_coordinates, (
            "coordinates must be in {}, the coordinate system "
            "you provided, {}, is not yet supported".format(
                supported_coordinates, coordinates
            )
        )
        xyz = check_xyz_dim(xyz)

        n_obs = xyz.shape[0]

        # orientation of the dipole
        if coordinates.lower() == "cylindrical":
            xyz = spatial.cylindrical_2_cartesian(xyz)

        dxyz = self.vector_distance(xyz)
        r = spatial.repeat_scalar(self.distance(xyz))
        m = self.moment * np.atleast_2d(self.orientation).repeat(n_obs, axis=0)

        m_cross_r = np.cross(m, dxyz)
        a = (self.mu / (4 * np.pi)) * m_cross_r / (r**3)

        if coordinates.lower() == "cylindrical":
            a = spatial.cartesian_2_cylindrical(xyz, a)

        return a

    def magnetic_flux_density(self, xyz, coordinates="cartesian"):
        r"""Compute magnetic flux density produced by the static magnetic dipole.

        This method computes the magnetic flux density produced by the static magnetic
        dipole at gridded xyz locations provided. Where :math:`\mu` is the magnetic
        permeability of the wholespace, :math:`\mathbf{m}` is the dipole moment,
        :math:`\mathbf{r_0}` the dipole location and :math:`\mathbf{r}` is the location
        at which we want to evaluate the magnetic flux density :math:`\mathbf{B}`:

        .. math::

            \mathbf{B}(\mathbf{r}) = \frac{\mu}{4\pi} \Bigg [
            \frac{3 \Delta \mathbf{r} \big ( \mathbf{m} \cdot \, \Delta \mathbf{r} \big ) }{| \Delta \mathbf{r} |^5}
            - \frac{\mathbf{m}}{| \Delta \mathbf{r} |^3} \Bigg ]

        where

        .. math::
            \mathbf{\Delta r} = \mathbf{r} - \mathbf{r_0}

        For reference, see equation Griffiths (1999).

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            gridded locations at which we calculate the magnetic flux density
        coordinates: str {'cartesian', 'cylindrical'}
            coordinate system that the location (xyz) are provided.
            The solution is also returned in this coordinate system.
            Default: `"cartesian"`

        Returns
        -------
        (n, 3) numpy.ndarray
            The magnetic flux density at each observation location in the
            coordinate system specified in Teslas.


        Examples
        --------
        Here, we define a z-oriented magnetic dipole and plot the magnetic
        flux density on the xy-plane that intercepts y=0.

        >>> from geoana.em.static import MagneticDipoleWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the magnetic dipole.

        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[0., 0., 1.]
        >>> moment = 1.
        >>> dipole_object = MagneticDipoleWholeSpace(
        >>>     location=location, orientation=orientation, moment=moment
        >>> )

        Now we create a set of gridded locations and compute the vector potential.

        >>> xyz = ndgrid(np.linspace(-1, 1, 20), np.array([0]), np.linspace(-1, 1, 20))
        >>> B = dipole_object.magnetic_flux_density(xyz)

        Finally, we plot the vector potential on the plane. Given the symmetry,
        there are only horizontal components.

        >>> fig = plt.figure(figsize=(4, 4))
        >>> ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        >>> plot2Ddata(xyz[:, 0::2], B[:, 0::2], ax=ax, vec=True, scale='log')
        >>> ax.set_xlabel('X')
        >>> ax.set_ylabel('Z')
        >>> ax.set_title('Magnetic flux density at y=0')
        """

        supported_coordinates = ["cartesian", "cylindrical"]
        assert coordinates.lower() in supported_coordinates, (
            "coordinates must be in {}, the coordinate system "
            "you provided, {}, is not yet supported".format(
                supported_coordinates, coordinates
            )
        )
        xyz = check_xyz_dim(xyz)

        if coordinates.lower() == "cylindrical":
            xyz = spatial.cylindrical_2_cartesian(xyz)

        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)[..., None]
        m_vec = self.moment * self.orientation

        m_dot_r = np.einsum('...i,i->...', r_vec, m_vec)[..., None]

        b = (self.mu / (4 * np.pi)) * (
            (3.0 * r_vec * m_dot_r / (r ** 5)) -
            m_vec / (r ** 3)
        )

        if coordinates.lower() == "cylindrical":
            b = spatial.cartesian_2_cylindrical(xyz, b)

        return b

    def magnetic_field(self, xyz, coordinates="cartesian"):
        r"""Compute the magnetic field produced by a static magnetic dipole.

        This method computes the magnetic field produced by the static magnetic dipole at
        the set of gridded xyz locations provided. Where :math:`\mathbf{m}` is the dipole
        moment, :math:`\mathbf{r_0}` is the dipole location and :math:`\mathbf{r}` is the
        location at which we want to evaluate the magnetic field :math:`\mathbf{H}`:

        .. math::

            \mathbf{H}(\mathbf{r}) = \frac{1}{4\pi} \Bigg [
            \frac{3 \Delta \mathbf{r} \big ( \mathbf{m} \cdot \, \Delta \mathbf{r} \big ) }{| \Delta \mathbf{r} |^5}
            - \frac{\mathbf{m}}{| \Delta \mathbf{r} |^3} \Bigg ]

        where

        .. math::
            \mathbf{\Delta r} = \mathbf{r} - \mathbf{r_0}

        For reference, see equation Griffiths (1999).

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray xyz
            gridded locations at which we calculate the magnetic field
        coordinates: str {'cartesian', 'cylindrical'}
            coordinate system that the location (xyz) are provided.
            The solution is also returned in this coordinate system.
            Default: `"cartesian"`

        Returns
        -------
        (n, 3) numpy.ndarray
            The magnetic field at each observation location in the
            coordinate system specified in units A/m.

        Examples
        --------
        Here, we define a z-oriented magnetic dipole and plot the magnetic
        field on the xz-plane that intercepts y=0.

        >>> from geoana.em.static import MagneticDipoleWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the magnetic dipole.

        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[0., 0., 1.]
        >>> moment = 1.
        >>> dipole_object = MagneticDipoleWholeSpace(
        >>>     location=location, orientation=orientation, moment=moment
        >>> )

        Now we create a set of gridded locations and compute the vector potential.

        >>> xyz = ndgrid(np.linspace(-1, 1, 20), np.array([0]), np.linspace(-1, 1, 20))
        >>> H = dipole_object.magnetic_field(xyz)

        Finally, we plot the vector potential on the plane. Given the symmetry,
        there are only horizontal components.

        >>> fig = plt.figure(figsize=(4, 4))
        >>> ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        >>> plot2Ddata(xyz[:, 0::2], H[:, 0::2], ax=ax, vec=True, scale='log')
        >>> ax.set_xlabel('X')
        >>> ax.set_ylabel('Z')
        >>> ax.set_title('Magnetic field at y=0')

        """
        return self.magnetic_flux_density(xyz, coordinates=coordinates) / self.mu


class MagneticPoleWholeSpace(BaseEM, BaseMagneticDipole):
    """Class for a static magnetic pole in a wholespace.

    The ``MagneticPoleWholeSpace`` class is used to analytically compute the
    fields and potentials within a wholespace due to a static magnetic pole.
    """

    def magnetic_flux_density(self, xyz, coordinates="cartesian"):
        r"""Compute the magnetic flux density produced by the static magnetic pole.

        This method computes the magnetic flux density produced by the static magnetic pole
        at the set of gridded xyz locations provided. Where :math:`\mu` is the magnetic
        permeability of the wholespace, :math:`m` is the moment amplitude,
        :math:`\mathbf{r_0}` the pole's location and :math:`\mathbf{r}` is the location
        at which we want to evaluate the magnetic flux density :math:`\mathbf{B}`:

        .. math::

            \mathbf{B}(\mathbf{r}) = \frac{\mu m}{4\pi} \frac{\Delta \mathbf{r}}{| \Delta \mathbf{r}|^3}

        where

        .. math::
            \mathbf{\Delta r} = \mathbf{r} - \mathbf{r_0}

        For reference, see equation Griffiths (1999).

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray xyz
            gridded xyz locations at which we calculate the magnetic flux density
        coordinates: str {'cartesian', 'cylindrical'}
            coordinate system that the location (xyz) are provided.
            The solution is also returned in this coordinate system.
            Default: `"cartesian"`

        Returns
        -------
        (n, 3) numpy.ndarray
            The magnetic flux density at each observation location in the
            coordinate system specified in units T.

        """

        supported_coordinates = ["cartesian", "cylindrical"]
        assert coordinates.lower() in supported_coordinates, (
            "coordinates must be in {}, the coordinate system "
            "you provided, {}, is not yet supported".format(
                supported_coordinates, coordinates
            )
        )
        xyz = check_xyz_dim(xyz)

        if coordinates.lower() == "cylindrical":
            xyz = spatial.cylindrical_2_cartesian(xyz)

        r = self.vector_distance(xyz)
        dxyz = spatial.repeat_scalar(self.distance(xyz))

        b = self.moment * self.mu / (4 * np.pi * (dxyz ** 3)) * r

        if coordinates.lower() == "cylindrical":
            b = spatial.cartesian_2_cylindrical(xyz, b)

        return b

    def magnetic_field(self, xyz, coordinates="cartesian"):
        r"""Compute the magnetic field produced by the static magnetic pole.

        This method computes the magnetic field produced by the static magnetic pole at
        the set of gridded xyz locations provided. Where :math:`\mu` is the magnetic
        permeability of the wholespace, :math:`m` is the moment amplitude,
        :math:`\mathbf{r_0}` the pole's location and :math:`\mathbf{r}` is the location
        at which we want to evaluate the magnetic field :math:`\mathbf{H}`:

        .. math::

            \mathbf{G}(\mathbf{r}) = \frac{m}{4\pi} \frac{\Delta \mathbf{r}}{| \Delta \mathbf{r}|^3}

        where

        .. math::
            \mathbf{\Delta r} = \mathbf{r} - \mathbf{r_0}

        For reference, see equation Griffiths (1999).

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray xyz
            gridded locations at which we calculate the magnetic field
        coordinates: str {'cartesian', 'cylindrical'}
            coordinate system that the location (xyz) are provided.
            The solution is also returned in this coordinate system.
            Default: `"cartesian"`

        Returns
        -------
        (n, 3) numpy.ndarray
            The magnetic field at each observation location in the
            coordinate system specified in units A/m.

        """
        return self.magnetic_flux_density(xyz, coordinates=coordinates) / self.mu


class CircularLoopWholeSpace(BaseEM, BaseDipole):
    """Class for a circular loop of static current in a wholespace.

    The ``CircularLoopWholeSpace`` class is used to analytically compute the
    fields and potentials within a wholespace due to a circular loop carrying
    static current.

    Parameters
    ----------
    current : float
        Electrical current in the loop (A). Default is 1.
    radius : float
        Radius of the loop (m). Default is :math:`\\pi^{-1/2}` so that the loop
        has a default dipole moment of 1 :math:`A/m^2`.
    """

    def __init__(self, radius=np.sqrt(1.0/np.pi), current=1.0, **kwargs):

        self.current = current
        self.radius = radius
        super().__init__(**kwargs)


    @property
    def current(self):
        """Current in the loop in Amps

        Returns
        -------
        float
            Current in the loop Amps
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
    def radius(self):
        """Radius of the loop in meters

        Returns
        -------
        float
            Radius of the loop in meters
        """
        return self._radius

    @radius.setter
    def radius(self, value):

        try:
            value = float(value)
        except:
            raise TypeError(f"radius must be a number, got {type(value)}")

        if value <= 0.0:
            raise ValueError("radius must be greater than 0")

        self._radius = value

    def vector_potential(self, xyz, coordinates="cartesian"):
        r"""Compute the vector potential for the static loop in a wholespace.

        This method computes the vector potential for the cirular current loop
        at the set of gridded xyz locations provided. Where :math:`\mu` is the magnetic
        permeability, :math:`I d\mathbf{s}` represents an infinitessimal segment
        of current at location :math:`\mathbf{r_s}` and :math:`\mathbf{r}` is the location
        at which we want to evaluate the vector potential :math:`\mathbf{a}`:

        .. math::

            \mathbf{a}(\mathbf{r}) = \frac{\mu I}{4\pi} \oint
            \frac{1}{|\mathbf{r} - \mathbf{r_s}|} d\mathbf{s}


        The above expression can be solve analytically by using the appropriate
        change of coordinate transforms and the solution for a horizontal current
        loop. For a horizontal current loop centered at (0,0,0), the solution in
        radial coordinates is given by:

        .. math::

            a_\theta (\rho, z) = \frac{\mu_0 I}{\pi k}
            \sqrt{ \frac{R}{\rho^2}} \bigg [ (1 - k^2/2) \, K(k^2) - K(k^2) \bigg ]

        where

        .. math::

            k^2 = \frac{4 R \rho}{(R + \rho)^2 + z^2}

        and

        - :math:`\rho = \sqrt{x^2 + y^2}` is the horizontal distance to the test point
        - :math:`I` is the current through the loop
        - :math:`R` is the radius of the loop
        - :math:`E(k^2)` and :math:`K(k^2)` are the complete elliptic integrals

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray xyz
            gridded locations at which we calculate the vector potential
        coordinates: str {'cartesian', 'cylindrical'}
            coordinate system that the location (xyz) are provided.
            The solution is also returned in this coordinate system.
            Default: `"cartesian"`

        Returns
        -------
        (n, 3) numpy.ndarray
            The magnetic vector potential at each observation location in the
            coordinate system specified in units *Tm*.

        Examples
        --------
        Here, we define a horizontal loop and plot the vector
        potential on the xy-plane that intercepts at z=0.

        >>> from geoana.em.static import CircularLoopWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the loop.

        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[0., 0., 1.]
        >>> radius = 0.5
        >>> simulation = CircularLoopWholeSpace(
        >>>     location=location, orientation=orientation, radius=radius
        >>> )

        Now we create a set of gridded locations and compute the vector potential.

        >>> xyz = ndgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50), np.array([0]))
        >>> a = simulation.vector_potential(xyz)

        Finally, we plot the vector potential on the plane. Given the symmetry,
        there are only horizontal components.

        >>> fig = plt.figure(figsize=(4, 4))
        >>> ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        >>> plot2Ddata(xyz[:, 0:2], a[:, 0:2], ax=ax, vec=True, scale='log')
        >>> ax.set_xlabel('X')
        >>> ax.set_ylabel('Y')
        >>> ax.set_title('Vector potential at z=0')

        """

        eps = 1e-10
        supported_coordinates = ["cartesian", "cylindrical"]
        assert coordinates.lower() in supported_coordinates, (
            "coordinates must be in {}, the coordinate system "
            "you provided, {}, is not yet supported".format(
                supported_coordinates, coordinates
            )
        )
        xyz = check_xyz_dim(xyz)

        # convert coordinates if not cartesian
        if coordinates.lower() == "cylindrical":
            xyz = spatial.cylindrical_2_cartesian(xyz)

        xyz = spatial.rotate_points_from_normals(
            xyz, np.array(self.orientation),  # work around for a properties issue
            np.r_[0., 0., 1.], x0=np.array(self.location)
        )

        n_obs = xyz.shape[0]
        dxyz = self.vector_distance(xyz)
        r = self.distance(xyz)

        rho = np.sqrt((dxyz[:, :2]**2).sum(1))

        k2 = (4 * self.radius * rho) / ((self.radius + rho)**2 +dxyz[:, 2]**2)
        k2[k2 > 1.] = 1.  # if there are any rounding errors

        E = ellipe(k2)
        K = ellipk(k2)

        # singular if rho = 0, k2 = 1
        ind = (rho > eps) & (k2 < 1)

        Atheta = np.zeros_like(r)
        Atheta[ind] = (
            (self.mu * self.current) / (np.pi * np.sqrt(k2[ind])) *
            np.sqrt(self.radius / rho[ind]) *
            ((1. - k2[ind] / 2.)*K[ind] - E[ind])
        )

        # assume that the z-axis aligns with the polar axis
        A = np.zeros_like(xyz)
        A[ind, 0] = Atheta[ind] * (-dxyz[ind, 1] / rho[ind])
        A[ind, 1] = Atheta[ind] * (dxyz[ind, 0] / rho[ind])

        # rotate the points to aligned with the normal to the source
        A = spatial.rotate_points_from_normals(
            A, np.r_[0., 0., 1.], np.array(self.orientation),
            x0=np.array(self.location)
        )

        if coordinates.lower() == "cylindrical":
            A = spatial.cartesian_2_cylindrical(xyz, A)

        return A

    def magnetic_flux_density(self, xyz, coordinates="cartesian"):
        r"""Compute the magnetic flux density for the current loop in a wholespace.

        This method computes the magnetic flux density for the cirular current loop
        at the set of gridded xyz locations provided. Where :math:`\mu` is the magnetic
        permeability, :math:`I d\mathbf{s}` represents an infinitessimal segment
        of current at location :math:`\mathbf{r_s}` and :math:`\mathbf{r}` is the location
        at which we want to evaluate the magnetic flux density :math:`\mathbf{B}`:

        .. math::

            \mathbf{B}(\mathbf{r}) = - \frac{\mu I}{4\pi} \oint
            \frac{(\mathbf{r}-\mathbf{r_s}) \times d\mathbf{s}}{|\mathbf{r} - \mathbf{r_0}|^3}

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray xyz
            gridded locations at which we calculate the magnetic flux density
        coordinates: str {'cartesian', 'cylindrical'}
            coordinate system that the location (xyz) are provided.
            The solution is also returned in this coordinate system.
            Default: `"cartesian"`

        Returns
        -------
        (n, 3) numpy.ndarray
            The magnetic flux density at each observation location in the
            coordinate system specified in units *T*.

        Examples
        --------
        Here, we define a horizontal loop and plot the magnetic flux
        density on the xz-plane that intercepts at y=0.

        >>> from geoana.em.static import CircularLoopWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the loop.

        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[0., 0., 1.]
        >>> radius = 0.5
        >>> simulation = CircularLoopWholeSpace(
        >>>     location=location, orientation=orientation, radius=radius
        >>> )

        Now we create a set of gridded locations and compute the magnetic flux density.

        >>> xyz = ndgrid(np.linspace(-1, 1, 50), np.array([0]), np.linspace(-1, 1, 50))
        >>> B = simulation.magnetic_flux_density(xyz)

        Finally, we plot the magnetic flux density on the plane.

        >>> fig = plt.figure(figsize=(4, 4))
        >>> ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        >>> plot2Ddata(xyz[:, 0::2], B[:, 0::2], ax=ax, vec=True, scale='log')
        >>> ax.set_xlabel('X')
        >>> ax.set_ylabel('Y')
        >>> ax.set_title('Magnetic flux density at y=0')

        """
        xyz = check_xyz_dim(xyz)
        # convert coordinates if not cartesian
        if coordinates.lower() == "cylindrical":
            xyz = spatial.cylindrical_2_cartesian(xyz)
        elif coordinates.lower() != "cartesian":
            raise TypeError(
                f"coordinates must be 'cartesian' or 'cylindrical', the coordinate "
                f"system you provided, {coordinates}, is not yet supported."
            )

        xyz = spatial.rotate_points_from_normals(
            xyz, np.array(self.orientation),  # work around for a properties issue
            np.r_[0., 0., 1.], x0=np.array(self.location)
        )
        # rotate all the points such that the orientation is directly vertical

        dxyz = self.vector_distance(xyz)
        r = self.distance(xyz)

        rho = np.linalg.norm(dxyz[:, :2], axis=-1)

        B = np.zeros((len(rho), 3))

        # for On axis points
        inds_axial = rho==0.0

        B[inds_axial, -1] = self.mu * self.current * self.radius**2 / (
            2 * (self.radius**2 + dxyz[inds_axial, 2]**2)**(1.5)
        )

        # Off axis
        alpha = rho[~inds_axial]/self.radius
        beta = dxyz[~inds_axial, 2]/self.radius
        gamma = dxyz[~inds_axial, 2]/rho[~inds_axial]

        Q = ((1+alpha)**2 + beta**2)
        k2 =  4 * alpha/Q

        # axial part:
        B[~inds_axial, -1] = self.mu * self.current / (2 * self.radius * np.pi * np.sqrt(Q)) * (
            ellipe(k2)*(1 - alpha**2 - beta**2)/(Q  - 4 * alpha) + ellipk(k2)
        )

        # radial part:
        B_rad = self.mu * self.current * gamma / (2 * self.radius * np.pi * np.sqrt(Q)) * (
            ellipe(k2)*(1 + alpha**2 + beta**2)/(Q  - 4 * alpha) - ellipk(k2)
        )

        # convert radial component to x and y..
        B[~inds_axial, 0] = B_rad * (dxyz[~inds_axial, 0]/rho[~inds_axial])
        B[~inds_axial, 1] = B_rad * (dxyz[~inds_axial, 1]/rho[~inds_axial])

        # rotate the vectors to be aligned with the normal to the source
        B = spatial.rotate_points_from_normals(
           B, np.r_[0., 0., 1.], np.array(self.orientation),
        )

        if coordinates.lower() == "cylindrical":
            B = spatial.cartesian_2_cylindrical(xyz, B)

        return B

    def magnetic_field(self, xyz, coordinates="cartesian"):
        r"""Compute the magnetic field for the current loop in a wholespace.

        This method computes the magnetic field for the cirular current loop
        at the set of gridded xyz locations provided. Where :math:`\mu` is the magnetic
        permeability, :math:`I d\mathbf{s}` represents an infinitessimal segment
        of current at location :math:`\mathbf{r_s}` and :math:`\mathbf{r}` is the location
        at which we want to evaluate the magnetic field :math:`\mathbf{H}`:

        .. math::

            \mathbf{H}(\mathbf{r}) = - \frac{I}{4\pi} \oint
            \frac{(\mathbf{r}-\mathbf{r_s}) \times d\mathbf{s}}{|\mathbf{r} - \mathbf{r_0}|^3}

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray xyz
            gridded locations at which we calculate the magnetic field
        coordinates: str {'cartesian', 'cylindrical'}
            coordinate system that the location (xyz) are provided.
            The solution is also returned in this coordinate system.
            Default: `"cartesian"`

        Returns
        -------
        (n, 3) numpy.ndarray
            The magnetic field at each observation location in the
            coordinate system specified in units A/m.

        Examples
        --------
        Here, we define a horizontal loop and plot the magnetic field
        on the xz-plane that intercepts at y=0.

        >>> from geoana.em.static import CircularLoopWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the loop.

        >>> location = np.r_[0., 0., 0.]
        >>> orientation = np.r_[0., 0., 1.]
        >>> radius = 0.5
        >>> simulation = CircularLoopWholeSpace(
        >>>     location=location, orientation=orientation, radius=radius
        >>> )

        Now we create a set of gridded locations and compute the magnetic field.

        >>> xyz = ndgrid(np.linspace(-1, 1, 50), np.array([0]), np.linspace(-1, 1, 50))
        >>> H = simulation.magnetic_field(xyz)

        Finally, we plot the magnetic field on the plane.

        >>> fig = plt.figure(figsize=(4, 4))
        >>> ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        >>> plot2Ddata(xyz[:, 0::2], H[:, 0::2], ax=ax, vec=True, scale='log')
        >>> ax.set_xlabel('X')
        >>> ax.set_ylabel('Z')
        >>> ax.set_title('Magnetic field at y=0')

        """
        return self.magnetic_flux_density(xyz, coordinates=coordinates) / self.mu


class PointCurrentWholeSpace:
    """Class for a point current in a wholespace.

    The ``PointCurrentWholeSpace`` class is used to analytically compute the
    potentials, current densities and electric fields within a wholespace due to a point current.

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
        """Resistivity in the point current in :math:`\\Omega \\cdot m`

        Returns
        -------
        float
            Resistivity in the point current in :math:`\\Omega \\cdot m`
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

        self._location = vec

    def potential(self, xyz):
        """Electric potential for a point current in a wholespace.

        This method computes the potential for the point current in a wholespace at
        the set of gridded xyz locations provided. Where :math:`\\rho` is the
        electric resistivity, I is the current and R is the distance between
        the location we want to evaluate at and the point current.
        The potential V is:

        .. math::

            V = \\frac{\\rho I}{4 \\pi R}

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
        Here, we define a point current with current=1A and plot the electric
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
        >>> Z = np.zeros_like(X)
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> v = simulation.potential(xyz)

        Finally, we plot the electric potential as a function of distance.

        >>> plt.pcolor(X, Y, v)
        >>> cb1 = plt.colorbar()
        >>> cb1.set_label(label= 'Potential (V)')
        >>> plt.xlabel('x')
        >>> plt.ylabel('y')
        >>> plt.title('Electric Potential from Point Current in a Wholespace')
        >>> plt.show()
        """

        xyz = check_xyz_dim(xyz)
        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)

        v = self.rho * self.current / (4 * np.pi * r)
        return v

    def electric_field(self, xyz):
        """Electric field for a point current in a wholespace.

        This method computes the electric field for the point current in a wholespace at
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
        Here, we define a point current with current=1A in a wholespace and plot the electric
        field lines in the xy-plane.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from geoana.em.static import PointCurrentWholeSpace

        Define the point current.

        >>> rho = 1.0
        >>> current = 1.0
        >>> simulation = PointCurrentWholeSpace(
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
        >>> plt.title('Electric Field Lines for a Point Current in a Wholespace')
        >>> plt.show()
        """

        xyz = check_xyz_dim(xyz)
        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)

        e = self.rho * self.current * r_vec / (4 * np.pi * r[..., None] ** 3)
        return e

    def current_density(self, xyz):
        """Current density for a point current in a wholespace.

        This method computes the curent density for the point current in a wholespace at
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
        Here, we define a point current with current=1A in a wholespace and plot the current density.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from geoana.em.static import PointCurrentWholeSpace

        Define the point current.

        >>> rho = 1.0
        >>> current = 1.0
        >>> simulation = PointCurrentWholeSpace(
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
        >>> plt.title('Current Density for a Point Current in a Wholespace')
        >>> plt.show()
        """

        j = self.electric_field(xyz) / self.rho
        return j
