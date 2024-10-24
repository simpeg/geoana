import numpy as np
from scipy.special import ellipk, ellipe

from ..base import BaseDipole, BaseMagneticDipole, BaseEM, BaseLineCurrent
from ... import spatial
from geoana.utils import check_xyz_dim

__all__ = [
    "MagneticDipoleWholeSpace", "CircularLoopWholeSpace",
    "MagneticPoleWholeSpace", "PointCurrentWholeSpace"
]

from ...kernels import prism_fzy
from ...spatial import  cylindrical_to_cartesian, cartesian_to_cylindrical, rotation_matrix_from_normals


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
        xyz : (..., 3) numpy.ndarray xyz
            gridded locations at which we are calculating the vector potential
        coordinates: str {'cartesian', 'cylindrical'}
            coordinate system that the location (xyz) are provided.
            The solution is also returned in this coordinate system.
            Default: `"cartesian"`

        Returns
        -------
        (..., 3) numpy.ndarray
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

        # orientation of the dipole
        if coordinates.lower() == "cylindrical":
            xyz = spatial.cylindrical_to_cartesian(xyz)

        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
        r_hat = r_vec / r

        a = self.moment * self.mu / (4 * np.pi * r**2) * np.cross(self.orientation, r_hat)

        if coordinates.lower() == "cylindrical":
            a = spatial.cartesian_to_cylindrical(xyz, a)

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
        xyz : (..., 3) numpy.ndarray
            gridded locations at which we calculate the magnetic flux density
        coordinates: str {'cartesian', 'cylindrical'}
            coordinate system that the location (xyz) are provided.
            The solution is also returned in this coordinate system.
            Default: `"cartesian"`

        Returns
        -------
        (..., 3) numpy.ndarray
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
            xyz = spatial.cylindrical_to_cartesian(xyz)

        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
        r_hat = r_vec / r

        b = self.moment * self.mu / (4 * np.pi * r**3) * (
            3 * r_hat.dot(self.orientation)[..., None] * r_hat - self.orientation
        )

        if coordinates.lower() == "cylindrical":
            b = spatial.cartesian_to_cylindrical(xyz, b)

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
        xyz : (..., 3) numpy.ndarray xyz
            gridded locations at which we calculate the magnetic field
        coordinates: str {'cartesian', 'cylindrical'}
            coordinate system that the location (xyz) are provided.
            The solution is also returned in this coordinate system.
            Default: `"cartesian"`

        Returns
        -------
        (..., 3) numpy.ndarray
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
        xyz : (..., 3) numpy.ndarray xyz
            gridded xyz locations at which we calculate the magnetic flux density
        coordinates: str {'cartesian', 'cylindrical'}
            coordinate system that the location (xyz) are provided.
            The solution is also returned in this coordinate system.
            Default: `"cartesian"`

        Returns
        -------
        (..., 3) numpy.ndarray
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
            xyz = spatial.cylindrical_to_cartesian(xyz)

        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
        r_hat = r_vec / r

        b = self.moment * self.mu / (4 * np.pi * r ** 2) * r_hat

        if coordinates.lower() == "cylindrical":
            b = spatial.cartesian_to_cylindrical(xyz, b)

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
        xyz : (..., 3) numpy.ndarray xyz
            gridded locations at which we calculate the magnetic field
        coordinates: str {'cartesian', 'cylindrical'}
            coordinate system that the location (xyz) are provided.
            The solution is also returned in this coordinate system.
            Default: `"cartesian"`

        Returns
        -------
        (..., 3) numpy.ndarray
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
            \sqrt{ \frac{R}{\rho^2}} \bigg [ (1 - k^2/2) \, K(k^2) - E(k^2) \bigg ]

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
        xyz : (..., 3) numpy.ndarray xyz
            gridded locations at which we calculate the vector potential
        coordinates: str {'cartesian', 'cylindrical'}
            coordinate system that the location (xyz) are provided.
            The solution is also returned in this coordinate system.
            Default: `"cartesian"`

        Returns
        -------
        (..., 3) numpy.ndarray
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
            xyz = spatial.cylindrical_to_cartesian(xyz)

        # define a rotation matrix that rotates my orientation to z:
        rot = rotation_matrix_from_normals(self.orientation, [0, 0, 1], as_matrix=False)

        # rotate the points
        r_vec = rot.apply(xyz.reshape(-1, 3) - self.location)

        r_cyl = cartesian_to_cylindrical(r_vec)

        rho = r_cyl[..., 0]
        z = r_cyl[..., 2]

        a = self.radius
        C = self.mu * self.current / np.pi

        alpha_sq = (a - rho)**2 + z**2
        beta_sq = (a + rho)**2 + z**2
        beta = np.sqrt(beta_sq)

        k_sq = 1 - alpha_sq/beta_sq

        ek = ellipe(k_sq)
        kk = ellipk(k_sq)

        A_cyl = np.zeros_like(r_vec)

        # when rho is small relative to the radius and z
        small_rho = rho**2/(a**2 + z**2) < 1E-6

        temp = np.sqrt(a**2 + z[small_rho]**2)
        A_cyl[small_rho, 1] = np.pi * C * (
                # A(rho=0) = 0
                rho[small_rho] * a**2 / (4 * temp**3) +  # rho * A'(rho=0)
                # A''(rho=0) = 0
                rho[small_rho]**3 * 3 * a**2 / ( a ** 2 - 4 * z[small_rho]**2) / (
                    32 * np.sqrt(a**2 + z[small_rho]**2)**7
                )
        )
        z = z[~small_rho]
        beta = beta[~small_rho]
        beta_sq = beta_sq[~small_rho]
        rho = rho[~small_rho]
        ek = ek[~small_rho]
        kk = kk[~small_rho]

        # singular if alpha = 0
        # (a.k.a. the point was numerically on the loop)
        A_cyl[~small_rho, 1] = C / (2 * rho * beta) * (
            (a**2 + rho**2 + z**2) * kk - beta_sq * ek
        )

        A = cylindrical_to_cartesian(r_cyl, A_cyl)

        # un-do the rotation on the vector components.
        A = rot.apply(A, inverse=True).reshape(xyz.shape)

        if coordinates.lower() == "cylindrical":
            A = spatial.cartesian_to_cylindrical(xyz, A)

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
            xyz = spatial.cylindrical_to_cartesian(xyz)
        elif coordinates.lower() != "cartesian":
            raise TypeError(
                f"coordinates must be 'cartesian' or 'cylindrical', the coordinate "
                f"system you provided, {coordinates}, is not yet supported."
            )

        # define a rotation matrix that rotates my orientation to z:
        rot = rotation_matrix_from_normals(self.orientation, [0, 0, 1], as_matrix=False)

        # rotate the points
        r_vec = rot.apply(xyz.reshape(-1, 3) - self.location)

        r_cyl = cartesian_to_cylindrical(r_vec)
        rho = r_cyl[:, 0]
        z = r_cyl[:, 2]
        a = self.radius
        C = self.mu * self.current / np.pi

        alpha_sq = (a - rho)**2 + z**2
        beta_sq = (a + rho)**2 + z**2
        beta = np.sqrt(beta_sq)
        k_sq = 1 - alpha_sq / beta_sq

        ek = ellipe(k_sq)
        kk = ellipk(k_sq)

        B_cyl = np.zeros_like(r_cyl)

        # when rho is small relative to the radius and z
        small_rho = rho**2/(a**2 + z**2) < 1E-6

        temp = np.sqrt(a**2 + z[small_rho]**2)
        B_cyl[small_rho, 0] = 3 * C * np.pi * a**2 * z[small_rho] * (
            rho[small_rho] /(4 * temp ** 5)
            + rho[small_rho]**3 * (15 * a**2 - 20 * z[small_rho]**2)/(32 * temp**9)
        )

        # this expectedly blows up when rho = radius and z = 0
        # Meaning it is literally on the loop...
        B_cyl[:, 2] = C / (2 * alpha_sq * beta) * (
            (a**2 - rho**2 - z**2) * ek + alpha_sq * kk
        )

        z = z[~small_rho]
        alpha_sq = alpha_sq[~small_rho]
        beta = beta[~small_rho]
        rho = rho[~small_rho]
        ek = ek[~small_rho]
        kk = kk[~small_rho]

        B_cyl[~small_rho, 0] = C * z / (2 * alpha_sq * beta * rho) * (
                (a**2 + rho**2 + z**2) * ek - alpha_sq * kk
        )

        B = cylindrical_to_cartesian(r_cyl, B_cyl)

        B = rot.apply(B, inverse=True).reshape(xyz.shape)

        if coordinates.lower() == "cylindrical":
            B = cartesian_to_cylindrical(xyz, B)

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


class LineCurrentWholeSpace(BaseLineCurrent, BaseEM):
    """Class for a static line current in whole space.

    The ``LineCurrentFreeSpace`` class is used to analytically compute the
    fields and potentials within a wholespace due to a set of constant current-carrying
    wires.
    """

    def scalar_potential(self, xyz):
        xyz = check_xyz_dim(xyz)
        # If i had a single point, treat it as a source
        if self.n_segments == 0:
            r = np.linalg.norm(xyz - self.nodes[0], axis=-1)
            return self.current/(4 * np.pi * self.sigma * r)
        # if I was a closed loop, return 0
        if np.all(self.nodes[-1] == self.nodes[0]):
            return np.zeros_like(xyz[..., 0])

        r_A = np.linalg.norm(xyz - self.nodes[-1], axis=-1)
        r_B = np.linalg.norm(xyz - self.nodes[0], axis=-1)

        return self.current/(4 * np.pi * self.sigma) * (1/r_A - 1/r_B)

    def vector_potential(self, xyz):

        xyz = check_xyz_dim(xyz)

        out = np.zeros_like(xyz)
        temp_storage = np.zeros_like(xyz)
        for p0, p1 in zip(self.nodes[:-1], self.nodes[1:]):
            l_vec = p1 - p0
            l = np.linalg.norm(l_vec)
            l_hat = l_vec / l

            # find the rotation from the line segments orientation
            # to the x_hat direction.
            rot = rotation_matrix_from_normals(l_hat, [1, 0, 0], as_matrix=False)

            # shift and rotate the grid points
            r0_vec = rot.apply(xyz.reshape(-1, 3) - p0).reshape(xyz.shape)

            #p1 would've been shifted and rotated to [l, 0, 0]
            r1_vec = r0_vec - np.array([l, 0, 0])

            # hey these are just the 1/r kernel evaluations!
            v0_x = -prism_fzy(r0_vec[..., 0], r0_vec[..., 1], r0_vec[..., 2])
            v1_x = -prism_fzy(r1_vec[..., 0], r1_vec[..., 1], r1_vec[..., 2])

            temp_storage[..., 0] = v1_x - v0_x
            # the undo the local rotation...
            out += rot.apply(temp_storage.reshape(-1, 3), inverse=True).reshape(xyz.shape)

        out *= -self.mu * self.current / (4 * np.pi)

        # note because this is a whole space, we do not have to deal with the
        # magnetic fields due to the current flowing out of the ends of a grounded
        # wire.
        return out

    def magnetic_field(self, xyz):
        r"""Compute the magnetic field for the static current-carrying wire segments.

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray xyz
            gridded locations at which we are calculating the magnetic field

        Returns
        -------
        (..., 3) numpy.ndarray
            The magnetic field at each observation location in H/m.

        Examples
        --------
        Here, we define a horizontal square loop and plot the magnetic field
        on the xz-plane that intercepts at y=0.

        >>> from geoana.em.static import LineCurrentWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the loop. Note that to create an inductive
        source, we closed the loop.

        >>> x_nodes = np.array([-0.5, 0.5, 0.5, -0.5, -0.5])
        >>> y_nodes = np.array([-0.5, -0.5, 0.5, 0.5, -0.5])
        >>> z_nodes = np.zeros_like(x_nodes)
        >>> nodes = np.c_[x_nodes, y_nodes, z_nodes]
        >>> simulation = LineCurrentWholeSpace(nodes)

        Now we create a set of gridded locations and compute the magnetic field.

        >>> xyz = ndgrid(np.linspace(-1, 1, 50), np.array([0]), np.linspace(-1, 1, 50))
        >>> H = simulation.magnetic_field(xyz)

        Finally, we plot the magnetic field.

        >>> fig = plt.figure(figsize=(4, 4))
        >>> ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
        >>> plot2Ddata(xyz[:, [0, 2]], H[:, [0, 2]], ax=ax, vec=True, scale='log', ncontour=25)
        >>> ax.set_xlabel('X')
        >>> ax.set_ylabel('Z')
        >>> ax.set_title('Magnetic field')

        """
        return self.magnetic_flux_density(xyz) / self.mu

    def magnetic_flux_density(self, xyz):
        r"""Compute the magnetic flux density for the static current-carrying wire segments.

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray xyz
            gridded locations at which we are calculating the magnetic flux density

        Returns
        -------
        (..., 3) numpy.ndarray
            The magnetic flux density at each observation location in T.

        Examples
        --------
        Here, we define a horizontal square loop and plot the magnetic flux
        density on the XZ-plane that intercepts at Y=0.

        >>> from geoana.em.static import LineCurrentWholeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the loop. Note that to create an inductive
        source, we closed the loop

        >>> x_nodes = np.array([-0.5, 0.5, 0.5, -0.5, -0.5])
        >>> y_nodes = np.array([-0.5, -0.5, 0.5, 0.5, -0.5])
        >>> z_nodes = np.zeros_like(x_nodes)
        >>> nodes = np.c_[x_nodes, y_nodes, z_nodes]
        >>> simulation = LineCurrentWholeSpace(nodes)

        Now we create a set of gridded locations and compute the magnetic flux density.

        >>> xyz = ndgrid(np.linspace(-1, 1, 50), np.array([0]), np.linspace(-1, 1, 50))
        >>> B = simulation.magnetic_flux_density(xyz)

        Finally, we plot the magnetic flux density on the plane.

        >>> fig = plt.figure(figsize=(4, 4))
        >>> ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        >>> plot2Ddata(xyz[:, [0, 2]], B[:, [0, 2]], ax=ax, vec=True, scale='log', ncontour=25)
        >>> ax.set_xlabel('X')
        >>> ax.set_ylabel('Z')
        >>> ax.set_title('Magnetic flux density')

        """

        xyz = check_xyz_dim(xyz)

        out = np.zeros_like(xyz)
        temp_storage = np.zeros_like(xyz)
        for p0, p1 in zip(self.nodes[:-1], self.nodes[1:]):
            l_vec = p1 - p0
            l = np.linalg.norm(l_vec)
            l_hat = l_vec / l

            # find the rotation from the line segments orientation
            # to the x_hat direction.
            rot = rotation_matrix_from_normals(l_hat, [1, 0, 0], as_matrix=False)

            # shift and rotate the grid points
            r0_vec = rot.apply(xyz.reshape(-1, 3) - p0).reshape(xyz.shape)
            #p1 would've been shifted and rotated to [l, 0, 0]
            r1_vec = r0_vec - np.array([l, 0, 0])

            r0 = np.linalg.norm(r0_vec, axis=-1, keepdims=True)
            r1 = np.linalg.norm(r1_vec, axis=-1, keepdims=True)
            r0_hat = r0_vec / r0
            r1_hat = r1_vec / r1

            cyl_points = cartesian_to_cylindrical(r0_vec[..., 1:])

            temp_storage[..., 1] = (r1_hat[..., 0] - r0_hat[..., 0])/cyl_points[..., 0]
            temp_storage[..., 1:] = cylindrical_to_cartesian(cyl_points, temp_storage)

            # the undo the local rotation...
            out += rot.apply(temp_storage.reshape(-1, 3), inverse=True).reshape(xyz.shape)

        out *= -self.mu * self.current / (4 * np.pi)
        return out

    def electric_field(self, xyz):
        r"""Compute the electric for the static current-carrying wire segments.

        If the wire is closed, there is no electric field, but if it is an open
        wire,

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray xyz
            gridded locations at which we are calculating the electric field

        Returns
        -------
        (..., 3) numpy.ndarray
            The electric field y at each observation location in T.

        """
        xyz = check_xyz_dim(xyz)
        # If I had a single point, treat it as a source
        if self.n_segments == 0:
            r_vec = xyz - self.nodes[0]
            r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
            return self.current / (4 * np.pi * self.sigma) * (r_vec/r**3)
        # if I was a closed loop, return 0
        if np.all(self.nodes[-1] == self.nodes[0]):
            return np.zeros_like(xyz)
        # otherwise, current leaks out at the ends!
        r_vec_A = xyz - self.nodes[-1]
        r_vec_B = xyz - self.nodes[0]
        r_A = np.linalg.norm(r_vec_A, axis=-1, keepdims=True)
        r_B = np.linalg.norm(r_vec_B, axis=-1, keepdims=True)

        return self.current/(4 * np.pi * self.sigma) * (r_vec_A/r_A**3 - r_vec_B/r_B**3)

    def current_density(self, xyz):
        r"""Compute the current density for the static current-carrying wire segments.

        If the wire is closed, there is no current density, but if it is an open
        wire,

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray xyz
            gridded locations at which we are calculating the current density

        Returns
        -------
        (..., 3) numpy.ndarray
            The current density y at each observation location in T.

        """
        return self.sigma * self.electric_field(xyz)


class PointCurrentWholeSpace(LineCurrentWholeSpace):
    """Class for a point current in a wholespace.

    The ``PointCurrentWholeSpace`` class is used to analytically compute the
    potentials, current densities and electric fields within a wholespace due to a point current.

    Parameters
    ----------
    rho : float
        Resistivity in the point current (:math:`\\Omega \\cdot m`).
    current : float
        Electrical current in the point current (A). Default is 1A.
    location : array_like, optional
        Location at which we are observing in 3D space (m). Default is (0, 0, 0).
    """

    def __init__(self, rho, current=1.0, location=None):
        if location is None:
            location = np.r_[0, 0, 0]
        super().__init__(sigma=1/rho, nodes=location, current=current)

    @property
    def location(self):
        """Location of observer in 3D space.

        Returns
        -------
        (3) numpy.ndarray of float
            Location of observer in 3D space. Default = np.r_[0,0,0].
        """
        return self.nodes[0]

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
        self.nodes[0] = vec

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
        return super().scalar_potential(xyz)

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
        return super().electric_field(xyz)

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
        return super().current_density(xyz)
