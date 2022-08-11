"""
======================================================
Gravity (:mod:`geoana.gravity`)
======================================================
.. currentmodule:: geoana.gravity

The ``geoana.gravity`` module contains simulation classes for solving
basic gravitational problems.

Simulation Classes
==================
.. autosummary::
  :toctree: generated/

  PointMass
  Sphere
  Prism
"""

import numpy as np
from scipy.constants import G
from geoana.utils import check_xyz_dim
from geoana.shapes import BasePrism
from geoana.kernels import prism_f, prism_fz, prism_fzx, prism_fzy, prism_fzz


class PointMass:
    """Class for gravitational solutions for a point mass.

    The ``PointMass`` class is used to analytically compute the gravitational
    potentials, fields, and gradients for a point mass.

    Parameters
    ----------
    mass : float
        Mass of the point particle (kg). Default is m = 1 kg
    location : array_like, optional
        Location of the point mass in 3D space (m). Default is (0, 0, 0)
    """

    def __init__(self, mass=1.0, location=None, **kwargs):

        self.mass = mass
        if location is None:
            location = np.r_[0, 0, 0]
        self.location = location
        super().__init__(**kwargs)

    @property
    def mass(self):
        """Mass of the point particle in kilograms.

        Returns
        -------
        float
            Mass of the point particle in kilograms.
        """
        return self._mass

    @mass.setter
    def mass(self, value):

        try:
            value = float(value)
        except:
            raise TypeError(f"mass must be a number, got {type(value)}")

        self._mass = value

    @property
    def location(self):
        """Location of the point mass.

        Returns
        -------
        (3) numpy.ndarray of float
            Location of the point mass in meters.  Default = np.r_[0,0,0]
        """
        return self._location

    @location.setter
    def location(self, vec):

        try:
            vec = np.asarray(vec, dtype=float)
        except:
            raise TypeError(f"location must be array_like of float, got {type(vec)}")

        vec = np.squeeze(vec)
        if vec.shape != (3,):
            raise ValueError(
                f"location must be array_like with shape (3,), got {vec.shape}"
            )

        self._location = vec

    def gravitational_potential(self, xyz):
        """
        Gravitational potential due to a point mass.  See Blakely, 1996
        equation 3.4.

        .. math::

            U(P) = \\gamma \\frac{m}{r}

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Observation locations in units m.

        Returns
        -------
        (..., ) numpy.ndarray
            Gravitational potential at observation locations xyz in units :math:`\\frac{m^2}{s^2}`.

        Examples
        --------
        Here, we define a point mass with mass=1kg and plot the gravitational
        potential as a function of distance.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from geoana.gravity import PointMass

        Define the point mass.

        >>> location = np.r_[0., 0., 0.]
        >>> mass = 1.0
        >>> simulation = PointMass(
        >>>     mass=mass, location=location
        >>> )

        Now we create a set of gridded locations, take the distances and compute the gravitational potential.

        >>> X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> r = np.linalg.norm(xyz, axis=-1)
        >>> u = simulation.gravitational_potential(xyz)

        Finally, we plot the gravitational potential as a function of distance.

        >>> plt.plot(r, u)
        >>> plt.xlabel('Distance from point mass')
        >>> plt.ylabel('Gravitational potential')
        >>> plt.title('Gravitational Potential as a function of distance from point mass')
        >>> plt.show()
        """
        xyz = check_xyz_dim(xyz)
        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)
        u_g = (G * self.mass) / r
        return u_g

    def gravitational_field(self, xyz):
        """
        Gravitational field due to a point mass.  See Blakely, 1996
        equation 3.3.

        .. math::

            \\mathbf{g} = \\nabla U(P)

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Observation locations in units m.

        Returns
        -------
        (..., 3) numpy.ndarray
            Gravitational field at observation locations xyz in units :math:`\\frac{m}{s^2}`.

        Examples
        --------
        Here, we define a point mass with mass=1kg and plot the gravitational
        field lines in the xy-plane.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from geoana.gravity import PointMass

        Define the point mass.

        >>> location = np.r_[0., 0., 0.]
        >>> mass = 1.0
        >>> simulation = PointMass(
        >>>     mass=mass, location=location
        >>> )

        Now we create a set of gridded locations and compute the gravitational field.

        >>> X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> g = simulation.gravitational_field(xyz)

        Finally, we plot the gravitational field lines.

        >>> plt.quiver(X, Y, g[:,:,0], g[:,:,1])
        >>> plt.xlabel('x')
        >>> plt.ylabel('y')
        >>> plt.title('Gravitational Field Lines for a Point Mass')
        >>> plt.show()
        """
        xyz = check_xyz_dim(xyz)
        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)
        g_vec = -G * self.mass * r_vec / r[..., None] ** 3
        return g_vec

    def gravitational_gradient(self, xyz):
        """
        Gravitational gradient for a point mass.

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Observation locations in units m.

        Returns
        -------
        (..., 3, 3) numpy.ndarray
            Gravitational gradient at observation locations xyz in units :math:`\\frac{1}{s^2}`.

        Examples
        --------
        Here, we define a point mass with mass=1kg and plot the gravitational
        gradient.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from geoana.gravity import PointMass

        Define the point mass.

        >>> location = np.r_[0., 0., 0.]
        >>> mass = 1.0
        >>> simulation = PointMass(
        >>>     mass=mass, location=location
        >>> )

        Now we create a set of gridded locations and compute the gravitational gradient.

        >>> X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> g_tens = simulation.gravitational_gradient(xyz)

        Finally, we plot the gravitational gradient for each element of the 3 x 3 matrix.

        >>> fig = plt.figure(figsize=(10, 10))
        >>> gs = fig.add_gridspec(3, 3, hspace=0, wspace=0)
        >>> (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) = gs.subplots(sharex='col', sharey='row')
        >>> fig.suptitle('Gravitational Gradients for a Point Mass')
        >>> ax1.contourf(X, Y, g_tens[:,:,0,0])
        >>> ax2.contourf(X, Y, g_tens[:,:,0,1])
        >>> ax3.contourf(X, Y, g_tens[:,:,0,2])
        >>> ax4.contourf(X, Y, g_tens[:,:,1,0])
        >>> ax5.contourf(X, Y, g_tens[:,:,1,1])
        >>> ax6.contourf(X, Y, g_tens[:,:,1,2])
        >>> ax7.contourf(X, Y, g_tens[:,:,2,0])
        >>> ax8.contourf(X, Y, g_tens[:,:,2,1])
        >>> ax9.contourf(X, Y, g_tens[:,:,2,2])
        >>> plt.show()
        """
        xyz = check_xyz_dim(xyz)
        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)
        g_tens = -G * self.mass * (np.eye(3) / r[..., None, None] ** 3 -
                                   3 * r_vec[..., None] * r_vec[..., None, :] / r[..., None, None] ** 5)
        return g_tens


class Sphere(PointMass):
    """Class for gravitational solutions for a sphere.

    The ``Sphere`` class is used to analytically compute the gravitational
    potentials, fields, and gradients for a sphere.

    Parameters
    ----------
    rho : float
        Density of sphere (:math:`\\frac{kg}{m^3}`).
    radius : float
        Radius of sphere (m).
    mass : float
        Mass of the sphere (kg). Default is :math:`m = \\frac{4}{3} \\pi R^3 \\rho` kg.
    location : array_like, optional
        Center of the sphere (m). Default is (0, 0, 0).
    """

    def __init__(self, radius, rho, location=None, **kwargs):
        self.radius = radius
        super().__init__(location=location, **kwargs)
        self.rho = rho

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
    def rho(self):
        """Density of the sphere in kilograms over meters cubed.

        Returns
        -------
        float
            Density of the sphere in kilograms over meters cubed.
        """
        return self._rho

    @rho.setter
    def rho(self, item):
        item = float(item)
        self._rho = item

    @property
    def mass(self):
        """Mass of sphere in kilograms.

        Returns
        -------
        float
            Mass of the sphere in kilograms.
        """
        return 4 / 3 * np.pi * self.radius ** 3 * self.rho

    @mass.setter
    def mass(self, value):

        try:
            value = float(value)
        except:
            raise TypeError(f"mass must be a number, got {type(value)}")

        rho = value * 3 / (4 * np.pi * self.radius ** 3)
        self._rho = rho

    def gravitational_potential(self, xyz):
        """
        Gravitational potential due to a sphere.

        .. math::

            r > R

            \\phi (\\mathbf{r}) = \\gamma \\frac{m}{r}

            r < R

            \\phi (\\mathbf{r}) = \\gamma \\frac{2}{3} \\pi \\rho (3R^2 - r^2)

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Locations to evaluate at in units m.

        Returns
        -------
        (..., ) numpy.ndarray
            Gravitational potential at sphere location xyz in units :math:`\\frac{m^2}{s^2}`.

        Examples
        --------
        Here, we define a sphere with mass m and plot the gravitational
        potential as a function of distance.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from geoana.gravity import Sphere
        >>> from geoana.utils import ndgrid

        Define the sphere.

        >>> location = np.r_[0., 0., 0.]
        >>> rho = 1.0
        >>> radius = 1.0
        >>> simulation = Sphere(
        >>>     location=location, rho=rho, radius=radius
        >>> )

        Now we create a set of gridded locations, take the distances and compute the gravitational potential.

        >>> X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> r = np.linalg.norm(xyz, axis=-1)
        >>> u = simulation.gravitational_potential(xyz)

        Finally, we plot the gravitational potential as a function of distance.

        >>> plt.plot(r, u)
        >>> plt.xlabel('Distance from sphere')
        >>> plt.ylabel('Gravitational potential')
        >>> plt.title('Gravitational Potential as a function of distance from sphere')
        >>> plt.show()
        """
        xyz = check_xyz_dim(xyz)
        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)
        u_g = np.zeros_like(r)
        ind0 = r > self.radius
        u_g[ind0] = super().gravitational_potential(xyz[ind0])
        u_g[~ind0] = G * 2 / 3 * np.pi * self.rho * (3 * self.radius ** 2 - r[~ind0] ** 2)
        return u_g

    def gravitational_field(self, xyz):
        """
        Gravitational field due to a sphere.

        .. math::

            r > R

            \\mathbf{g} (\\mathbf{r}) = \\nabla \\phi (\\mathbf{r})

            r < R

            \\mathbf{g} (\\mathbf{r}) = - \\gamma \\frac{4}{3} \\pi \\rho \\mathbf{r}

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Locations to evaluate at in units m.

        Returns
        -------
        (..., 3) numpy.ndarray
            Gravitational field at sphere location xyz in units :math:`\\frac{m}{s^2}`.

        Examples
        --------
        Here, we define a sphere with mass m and plot the gravitational
        field lines in the xy-plane.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from geoana.gravity import Sphere

        Define the sphere.

        >>> location = np.r_[0., 0., 0.]
        >>> rho = 1.0
        >>> radius = 1.0
        >>> simulation = Sphere(
        >>>     location=location, rho=rho, radius=radius
        >>> )

        Now we create a set of gridded locations and compute the gravitational field.

        >>> X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> g = simulation.gravitational_field(xyz)

        Finally, we plot the gravitational field lines.

        >>> plt.quiver(X, Y, g[:,:,0], g[:,:,1])
        >>> plt.xlabel('x')
        >>> plt.ylabel('y')
        >>> plt.title('Gravitational Field Lines for a Sphere')
        >>> plt.show()
        """
        xyz = check_xyz_dim(xyz)
        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)
        g_vec = np.zeros((*r.shape, 3))
        ind0 = r > self.radius
        g_vec[ind0] = super().gravitational_field(xyz[ind0])
        g_vec[~ind0] = -G * 4 / 3 * np.pi * self.rho * r_vec[~ind0]
        return g_vec

    def gravitational_gradient(self, xyz):
        """
        Gravitational gradient for a sphere.

        .. math::

            r < R

            \\mathbf{T} (\\mathbf{r}) = -\\gamma \\frac{4}{3} \\pi \\rho \\mathbf{I}

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Locations to evaluate at in units m.

        Returns
        -------
        (..., 3, 3) numpy.ndarray
            Gravitational gradient at sphere location xyz in units :math:`\\frac{1}{s^2}`.

        Examples
        --------
        Here, we define a sphere with mass m and plot the gravitational
        gradient.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from geoana.gravity import Sphere

        Define the sphere.

        >>> location = np.r_[0., 0., 0.]
        >>> rho = 1.0
        >>> radius = 1.0
        >>> simulation = Sphere(
        >>>     location=location, rho=rho, radius=radius
        >>> )

        Now we create a set of gridded locations and compute the gravitational gradient.

        >>> X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        >>> Z = np.zeros_like(X) + 0.25
        >>> xyz = np.stack((X, Y, Z), axis=-1)
        >>> g_tens = simulation.gravitational_gradient(xyz)

        Finally, we plot the gravitational gradient for each element of the 3 x 3 matrix.

        >>> fig = plt.figure(figsize=(10, 10))
        >>> gs = fig.add_gridspec(3, 3, hspace=0, wspace=0)
        >>> (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) = gs.subplots(sharex='col', sharey='row')
        >>> fig.suptitle('Gravitational Gradients for a Sphere')
        >>> ax1.contourf(X, Y, g_tens[:,:,0,0])
        >>> ax2.contourf(X, Y, g_tens[:,:,0,1])
        >>> ax3.contourf(X, Y, g_tens[:,:,0,2])
        >>> ax4.contourf(X, Y, g_tens[:,:,1,0])
        >>> ax5.contourf(X, Y, g_tens[:,:,1,1])
        >>> ax6.contourf(X, Y, g_tens[:,:,1,2])
        >>> ax7.contourf(X, Y, g_tens[:,:,2,0])
        >>> ax8.contourf(X, Y, g_tens[:,:,2,1])
        >>> ax9.contourf(X, Y, g_tens[:,:,2,2])
        >>> plt.show()
        """
        xyz = check_xyz_dim(xyz)
        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)
        g_tens = np.zeros((*r.shape, 3, 3))
        ind0 = r > self.radius
        ind1 = r == self.radius
        g_tens[ind0] = super().gravitational_gradient(xyz[ind0])
        g_tens[~ind0] = -G * 4 / 3 * np.pi * self.rho * np.eye(3)
        g_tens[ind1] = np.NaN
        return g_tens


class Prism(BasePrism):
    """Class for gravitational solutions for a prism.

    The ``Prism`` class is used to analytically compute the gravitational
    potentials, fields, and gradients for a prism with constant denisty.

    Parameters
    ----------
    min_location : (3,) array_like
        (x, y, z) triplet of the minimum locations in each dimension
    max_location : (3,) array_like
        (x, y, z) triplet of the maximum locations in each dimension
    rho : float, optional
        Density of prism (:math:`\\frac{kg}{m^3}`).
    """

    def __init__(self, min_location, max_location, rho=1.0):
        self.rho = rho
        super().__init__(min_location=min_location, max_location=max_location)

    @property
    def rho(self):
        """ The density of the prism.

        Returns
        -------
        density : float
            In :math:`\\frac{kg}{m^3}`.
        """
        return self._rho

    @rho.setter
    def rho(self, value):
        try:
            value = float(value)
        except:
            raise TypeError(f"mass must be a number, got {type(value)}")

        self._rho = value

    @property
    def mass(self):
        """ The mass of the prism

        Returns
        -------
        mass : float
            In :math:`kg`.
        """
        return self.volume * self.rho

    def gravitational_potential(self, xyz):
        """
        Gravitational potential due to a prism.


        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Observation locations in units m.

        Returns
        -------
        (..., ) numpy.ndarray
            Gravitational potential of prism at location xyz in units :math:`\\frac{m^2}{s^2}`.
        """
        xyz = check_xyz_dim(xyz)
        # need to evaluate f node at each source locations
        return - G * self.rho * self._eval_def_int(prism_f, xyz[..., 0], xyz[..., 1], xyz[..., 2])

    def gravitational_field(self, xyz):
        """
        Gravitational field due to a prism.


        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Observation locations in units m.

        Returns
        -------
        (..., 3) numpy.ndarray
            Gravitational field of prism at location xyz in units :math:`\\frac{m}{s^2}`.
        """
        xyz = check_xyz_dim(xyz)
        # need to evaluate f node at each source locations
        gx = self._eval_def_int(prism_fz, xyz[..., 0], xyz[..., 1], xyz[..., 2], cycle=1)
        gy = self._eval_def_int(prism_fz, xyz[..., 0], xyz[..., 1], xyz[..., 2], cycle=2)
        gz = self._eval_def_int(prism_fz, xyz[..., 0], xyz[..., 1], xyz[..., 2])
        return - G * self.rho * np.stack((gx, gy, gz), axis=-1)

    def gravitational_gradient(self, xyz):
        """
        Gravitational gradient due to a prism.

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Observation locations in units m.

        Returns
        -------
        (..., 3) numpy.ndarray
            Gravitational gradient of prism at location xyz in units :math:`\\frac{m}{s^2}`.
        """
        xyz = check_xyz_dim(xyz)

        # need to evaluate f node at each source locations
        gxx = self._eval_def_int(prism_fzz, xyz[..., 0], xyz[..., 1], xyz[..., 2], cycle=1)
        gxy = self._eval_def_int(prism_fzx, xyz[..., 0], xyz[..., 1], xyz[..., 2], cycle=1)
        gxz = self._eval_def_int(prism_fzx, xyz[..., 0], xyz[..., 1], xyz[..., 2])

        gyy = self._eval_def_int(prism_fzz, xyz[..., 0], xyz[..., 1], xyz[..., 2], cycle=2)
        gyz = self._eval_def_int(prism_fzy, xyz[..., 0], xyz[..., 1], xyz[..., 2])

        # gzz = - gxx - gyy - 4 * np.pi * G * rho[in_cell]
        # easiest to just calculate it using another integral
        gzz = self._eval_def_int(prism_fzz, xyz[..., 0], xyz[..., 1], xyz[..., 2])

        first = np.stack([gxx, gxy, gxz], axis=-1)
        second = np.stack([gxy, gyy, gyz], axis=-1)
        third = np.stack([gxz, gyz, gzz], axis=-1)

        return - G * self.rho * np.stack([first, second, third], axis=-1)
