from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import properties
from scipy.special import ellipk, ellipe
from scipy.constants import mu_0, epsilon_0

from .base import BaseEM, BaseDipole, BaseMagneticDipole
from .. import spatial

__all__ = [
    "MagneticDipoleWholeSpace", "CircularLoopWholeSpace",
    "MagneticPoleWholeSpace"
]


class MagneticDipoleWholeSpace(BaseMagneticDipole, BaseEM):
    """Class for a static magnetic dipole in a wholespace.

    The ``MagneticDipoleWholeSpace`` class is used to analytically compute the
    fields and potentials within a wholespace due to a static magnetic dipole.
    """

    def __init__(self, **kwargs):
        BaseMagneticDipole.__init__(self, **kwargs)
        BaseEM.__init__(self, **kwargs)


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

        """
        supported_coordinates = ["cartesian", "cylindrical"]
        assert coordinates.lower() in supported_coordinates, (
            "coordinates must be in {}, the coordinate system "
            "you provided, {}, is not yet supported".format(
                supported_coordinates, coordinates
            )
        )

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

            \mathbf{B}(\mathbf{r}) = \frac{\mu}{4\pi} \bigg [
            \frac{3 \Delta \mathbf{r} big ( \mathbf{m} \cdot \, \Delta \mathbf{r} \big ) }{| \Delta \mathbf{r} |^5}
            - \frac{\mathbf{m}}{| \Delta \mathbf{r} |^3}

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

        """

        supported_coordinates = ["cartesian", "cylindrical"]
        assert coordinates.lower() in supported_coordinates, (
            "coordinates must be in {}, the coordinate system "
            "you provided, {}, is not yet supported".format(
                supported_coordinates, coordinates
            )
        )

        n_obs = xyz.shape[0]

        if coordinates.lower() == "cylindrical":
            xyz = spatial.cylindrical_2_cartesian(xyz)

        r = self.vector_distance(xyz)
        dxyz = spatial.repeat_scalar(self.distance(xyz))
        m_vec = (
            self.moment * np.atleast_2d(self.orientation).repeat(n_obs, axis=0)
        )

        m_dot_r = (m_vec * r).sum(axis=1)

        # Repeat the scalars
        m_dot_r = np.atleast_2d(m_dot_r).T.repeat(3, axis=1)
        # dxyz = np.atleast_2d(dxyz).T.repeat(3, axis=1)

        b = (self.mu / (4 * np.pi)) * (
            (3.0 * r * m_dot_r / (dxyz ** 5)) -
            m_vec / (dxyz ** 3)
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

            \mathbf{H}(\mathbf{r}) = \frac{1}{4\pi} \bigg [
            \frac{3 \Delta \mathbf{r} big ( \mathbf{m} \cdot \, \Delta \mathbf{r} \big ) }{| \Delta \mathbf{r} |^5}
            - \frac{\mathbf{m}}{| \Delta \mathbf{r} |^3}

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
        (*, 3) numpy.ndarray
            The magnetic field at each observation location in the
            coordinate system specified in units A/m.

        """
        return self.magnetic_flux_density(xyz, coordinates=coordinates) / self.mu


class MagneticPoleWholeSpace(BaseMagneticDipole, BaseEM):
    """Class for a static magnetic pole in a wholespace.

    The ``MagneticPoleWholeSpace`` class is used to analytically compute the
    fields and potentials within a wholespace due to a static magnetic pole.
    """

    def __init__(self, **kwargs):
        BaseMagneticDipole.__init__(self, **kwargs)
        BaseEM.__init__(self, **kwargs)

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

        n_obs = xyz.shape[0]

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


class CircularLoopWholeSpace(BaseDipole, BaseEM):
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
        BaseDipole.__init__(self, **kwargs)
        BaseEM.__init__(self, **kwargs)


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
        
        if value <= 0.0:
            raise ValueError("current must be greater than 0")

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


    # current = properties.Float(
    #     "Electric current through the loop (A)", default=1.
    # )

    # radius = properties.Float(
    #     "radius of the loop (m)", default=1., min=0.
    # )

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
            \sqrt{R / \rho^2}[(1 - k^2/2) * K(k^2) - K(k^2)]

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

        """

        eps = 1e-10
        supported_coordinates = ["cartesian", "cylindrical"]
        assert coordinates.lower() in supported_coordinates, (
            "coordinates must be in {}, the coordinate system "
            "you provided, {}, is not yet supported".format(
                supported_coordinates, coordinates
            )
        )

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


        # The above expression can be solve analytically by using the appropriate
        # change of coordinates transforms and the solution for a horizontal current
        # loop. For a horizontal current loop centered at (0,0,0), the solution in
        # radial coordinates is given by:

        # .. math::

        #     a_\theta (\rho, z) = \frac{\mu_0 I}{\pi k}
        #     \sqrt{R / \rho^2}[(1 - k^2/2) * K(k^2) - K(k^2)]

        # where

        # .. math::

        #     k^2 = \frac{4 R \rho}{(R + \rho)^2 + z^2}

        # and

        # - :math:`\rho = \sqrt{x^2 + y^2}` is the horizontal distance to the test point
        # - :math:`I` is the current through the loop
        # - :math:`R` is the radius of the loop
        # - :math:`E(k^2)` and :math:`K(k^2)` are the complete elliptic integrals



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

        """
        xyz = np.atleast_2d(xyz)
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


        # The above expression can be solve analytically by using the appropriate
        # change of coordinates transforms and the solution for a horizontal current
        # loop. For a horizontal current loop centered at (0,0,0), the solution in
        # radial coordinates is given by:

        # .. math::

        #     a_\theta (\rho, z) = \frac{\mu_0 I}{\pi k}
        #     \sqrt{R / \rho^2}[(1 - k^2/2) * K(k^2) - K(k^2)]

        # where

        # .. math::

        #     k^2 = \frac{4 R \rho}{(R + \rho)^2 + z^2}

        # and

        # - :math:`\rho = \sqrt{x^2 + y^2}` is the horizontal distance to the test point
        # - :math:`I` is the current through the loop
        # - :math:`R` is the radius of the loop
        # - :math:`E(k^2)` and :math:`K(k^2)` are the complete elliptic integrals



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


class ElectrostaticSphere:
    """Class for electrostatic solutions for a sphere in a wholespace.

    The ``ElectrostaticSphere`` class is used to analytically compute the electric
    potentials, fields, currents and change densities for a sphere in a wholespace.
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
            vec = np.asarray(vec, dtype=np.float64)
            vec = np.atleast_1d(vec)
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
