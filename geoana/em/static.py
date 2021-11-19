from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import properties
from scipy.special import ellipk, ellipe
from scipy.constants import epsilon_0

from .base import BaseEM, BaseDipole, BaseMagneticDipole
from .. import spatial

__all__ = [
    "MagneticDipoleWholeSpace", "CircularLoopWholeSpace",
    "MagneticPoleWholeSpace"
]


class MagneticDipoleWholeSpace(BaseMagneticDipole, BaseEM):
    """
    Static magnetic dipole in a wholespace.
    """

    def vector_potential(self, xyz, coordinates="cartesian"):
        """Vector potential of a static magnetic dipole. See Griffiths, 1999
        equation 5.83

        .. math::

            \\vec{A}(\\vec{r}) = \\frac{\mu_0}{4\pi}
            \\frac{\\vec{m}\\times\\vec{r}}{r^3}

        **Required**

        :param numpy.ndarray xyz: Location at which we calculate the vector
                                potential

        **Optional**

        :param str coordinates: coordinate system that the xyz is provided
                                in and that the solution will be returned
                                in (cartesian or cylindrical).
                                Default: `"cartesian"`

        **Returns**

        :rtype: numpy.ndarray
        :return: The magnetic vector potential at each observation location

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
        """Magnetic flux (:math:`\\vec{b}`) of a static magnetic dipole

        **Required**

        :param numpy.ndarray xyz: Location of the receivers(s)

        **Optional**

        :param str coordinates: coordinate system that the xyz is provided
                                in and that the solution will be returned
                                in (cartesian or cylindrical).
                                Default: `"cartesian"`

        **Returns**

        :rtype: numpy.ndarray
        :return: The magnetic flux at each observation location
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
        """Magnetic field (:math:`\\vec{h}`) of a static magnetic dipole

        **Required**

        :param numpy.ndarray xyz: Location of the receivers(s)

        **Optional**

        :param str coordinates: coordinate system that the xyz is provided
                                in and that the solution will be returned
                                in (cartesian or cylindrical).
                                Default: `"cartesian"`

        **Returns**

        :rtype: numpy.ndarray
        :return: The magnetic field at each observation location

        """
        return self.magnetic_flux_density(xyz, coordinates=coordinates) / self.mu


class MagneticPoleWholeSpace(BaseMagneticDipole, BaseEM):
    """
    Static magnetic pole in a wholespace.
    """

    def magnetic_flux_density(self, xyz, coordinates="cartesian"):
        """Magnetic flux (:math:`\\vec{b}`) of a static magnetic dipole

        **Required**

        :param numpy.ndarray xyz: Location of the receivers(s)

        **Optional**

        :param str coordinates: coordinate system that the xyz is provided
                                in and that the solution will be returned
                                in (cartesian or cylindrical).
                                Default: `"cartesian"`

        **Returns**

        :rtype: numpy.ndarray
        :return: The magnetic flux at each observation location
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
        """Magnetic field (:math:`\\vec{h}`) of a static magnetic dipole

        **Required**

        :param numpy.ndarray xyz: Location of the receivers(s)

        **Optional**

        :param str coordinates: coordinate system that the xyz is provided
                                in and that the solution will be returned
                                in (cartesian or cylindrical).
                                Default: `"cartesian"`

        **Returns**

        :rtype: numpy.ndarray
        :return: The magnetic field at each observation location

        """
        return self.magnetic_flux_density(xyz, coordinates=coordinates) / self.mu


class CircularLoopWholeSpace(BaseDipole, BaseEM):

    """
    Static magnetic field from a circular loop in a wholespace.
    """

    current = properties.Float(
        "Electric current through the loop (A)", default=1.
    )

    radius = properties.Float(
        "radius of the loop (m)", default=1., min=0.
    )

    def vector_potential(self, xyz, coordinates="cartesian"):
        """Vector potential due to the a steady-state current through a
        circular loop. We solve in cylindrical coordinates

        .. math::

            A_\\theta(\\rho, z) = \\frac{\mu_0 I}{\pi k}
            \sqrt{R / \\rho^2}[(1 - k^2/2) * K(k^2) - K(k^2)]

        where

        .. math::

            k^2 = \\frac{4 R \\rho}{(R + \\rho)^2 + z^2}

        and

        - :math:`\\rho = \sqrt{x^2 + y^2}` is the horizontal distance to the test point
        - :math:`r` is the distance to a test point
        - :math:`I` is the current through the loop
        - :math:`R` is the radius of the loop
        - :math:`E(k^2)` and :math:`K(k^2)` are the complete elliptic integrals


        **Required**

        :param numpy.ndarray xyz: Location where we calculate the vector
                                potential

        **Optional**

        :param str coordinates: coordinate system that the xyz is provided
                                in and that the solution will be returned
                                in (cartesian or cylindrical).
                                Default: `"cartesian"`

        **Returns**

        :rtype: numpy.ndarray
        :return: The magnetic vector potential at each observation location

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
        """Calculates the magnetic flux density (B) due to a circular current loop

        Parameters
        ----------
        xyz : np.ndarray
            locations to evaluate the function at shape (3, ) or (*, 3)
        coordinates : {cartesian, cylindrical}
            which coordinate system the input and output points are defined in.

        Returns
        -------
        B_field : np.ndarray
            Magnetic Flux Density vector at the given points, shape (*, 3)
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
        """Calculates the magnetic field (H) due to a circular current loop

        Parameters
        ----------
        xyz : np.ndarray
            locations to evaluate the function at shape (3, ) or (*, 3)
        coordinates : {cartesian, cylindrical}
            which coordinate system the input and output points are defined in.

        Returns
        -------
        H_field : np.ndarray
            Magnetic Field vector at the given points, shape (*, 3)
        """
        return self.magnetic_flux_density(xyz, coordinates=coordinates) / self.mu


class ElectrostaticSphere():
    """
    Calculates static responses of a sphere in a halfspace given an x-directed
    static electric field.

    Parameters
    ----------
    radius : float
        radius of sphere (m)
    sigma_sphere : float
        conductivity of target sphere (S/m)
    sigma_background : float
        background conductivity (S/m)
    amplitude : float, optional
        amplitude of electric field (V/m)
    location : (3, ) np.ndarray, optional
        Center of the sphere, defaults to origin (0, 0, 0).
    """

    def __init__(self, radius, sigma_sphere, sigma_background, amplitude=1.0, location=None):

        self.radius = radius
        self.sigma_sphere = sigma_sphere
        self.sigma_background = sigma_background
        self.amplitude = amplitude
        self.location = location

    @property
    def sigma_sphere(self):
        return self._sig_sph

    @sigma_sphere.setter
    def sigma_sphere(self, item):
        item = float(item)
        if item <= 0.0:
            raise ValueError('Conductiviy must be positive')
        self._sig_sph = item

    @property
    def sigma_background(self):
        return self._sig_back

    @sigma_background.setter
    def sigma_background(self, item):
        item = float(item)
        if item <= 0.0:
            raise ValueError('Conductiviy must be positive')
        self._sig_back = item

    @property
    def radius(self):
        return self._r

    @radius.setter
    def radius(self, item):
        item = float(item)
        if item < 0.0:
            raise ValueError('radius must be non-negative')
        self._r = item

    @property
    def amplitude(self):
        return self._amp

    @amplitude.setter
    def amplitude(self, item):
        self._amp = float(item)

    @property
    def location(self):
        return self._loc

    @location.setter
    def location(self, item):
        if item is None:
            item = np.array([0, 0, 0])

        item = np.squeeze(np.asanyarray(item))
        if len(item.shape) != 1:
            raise ValueError("location must be a 1D array")
        if len(item) != 3:
            raise ValueError("location must be length 3 array")
        self._loc = item

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
        """Electric potential for a sphere in a uniform wholespace

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
