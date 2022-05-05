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
"""

import numpy as np

from scipy.constants import G


class PointMass:
    """Gravity of a point mass.

        Parameters
        ----------
        mass : float, optional
            Mass of the point particle in kg. Default is m = 1 kg
        location : array_like, optional
            Location of the point mass in 3D space. Default is (0,0,0)
        """

    def __init__(self, mass=1.0, location=None, **kwargs):

        self.mass = mass
        if location is None:
            location = np.r_[0, 0, 0]
        self.location = location
        super().__init__(**kwargs)

    @property
    def mass(self):
        """Mass of the point particle in kg

        Returns
        -------
        float
            Mass of the point particle in kg
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
        """Location of the point mass

        Returns
        -------
        (3) numpy.ndarray of float
            xyz point mass location
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
        equation 3.4

        .. math::

            U(P) = \\gamma \\frac{m}{r}

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            point mass location

        Returns
        -------
        (..., ) numpy.ndarray
            gravitational potential at point mass location xyz

        Examples
        --------
        Here, we define a point mass with mass=1kg and plot the gravitational
        potential as a function of distance.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from geoana.gravity import PointMass
        >>> from geoana.utils import ndgrid

        Define the point mass.

        >>> location = np.r_[0., 0., 0.]
        >>> mass = 1.0
        >>> simulation = PointMass(
        >>>     mass=mass, location=location
        >>> )

        Now we create a set of gridded locations, take the distances and compute the gravitational potential.

        >>> xyz = ndgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10), np.array([0]))
        >>> r = np.linalg.norm(xyz, axis=-1)
        >>> u = simulation.gravitational_potential(xyz)

        Finally, we plot the gravitational potential as a function of distance.

        >>> plt.plot(r, u)
        >>> plt.show()
        """

        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)
        u_g = (G * self.mass) / r
        return u_g

    def gravitational_field(self, xyz):
        """
        Gravitational field due to a point mass.  See Blakely, 1996
        equation 3.3

        .. math::

            \\mathbf{g} = \\nabla U(P)

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            point mass location

        Returns
        -------
        (..., 3) numpy.ndarray
            gravitational field at point mass location xyz

        Examples
        --------
        Here, we define a point mass with mass=1kg and plot the gravitational
        field lines in the xy-plane at z=0.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from geoana.gravity import PointMass
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata

        Define the point mass.

        >>> location = np.r_[0., 0., 0.]
        >>> mass = 1.0
        >>> simulation = PointMass(
        >>>     mass=mass, location=location
        >>> )

        Now we create a set of gridded locations and compute the gravitational field.

        >>> xyz = ndgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10), np.array([0]))
        >>> g = simulation.gravitational_field(xyz)

        Finally, we plot the gravitational field lines.

        >>> fig = plt.figure(figsize=(4, 4))
        >>> ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        >>> plot2Ddata(xyz[:, 0::2], g[:, 0::2], ax=ax, vec=True, scale='log')
        >>> ax.set_xlabel('x')
        >>> ax.set_ylabel('y')
        >>> ax.set_title('Gravitational field at z=0')
        """

        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)
        g_vec = (G * self.mass * r_vec) / r[..., None]
        return g_vec

    def gravitational_gradient(self, xyz):
        """
        Gravitational gradient for a point mass.

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            point mass location

        Returns
        -------
        (..., 3, 3) numpy.ndarray
            gravitational gradient at point mass location xyz

        Examples
        --------
        Here, we define a point mass with mass=1kg and plot the gravitational
        gradient.

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from geoana.gravity import PointMass
        >>> from geoana.utils import ndgrid

        Define the point mass.

        >>> location = np.r_[0., 0., 0.]
        >>> mass = 1.0
        >>> simulation = PointMass(
        >>>     mass=mass, location=location
        >>> )

        Now we create a set of gridded locations and compute the gravitational gradient.

        >>> xyz = ndgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        >>> g_tens = simulation.gravitational_gradient(xyz)

        Finally, we plot the gravitational gradient of each element of the 3 x 3 matrix.

        >>> fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,6))
        >>> ax1.set_aspect('equal')
        >>> ax1.set_title('1')
        >>> cf1 = ax1.contourf(xyz, xyz, g_tens[:,0,:], cmap='jet')
        >>> fig.colorbar(cf1, ax=ax1)
        >>> ax2.set_aspect('equal')
        >>> ax2.set_title('2')
        >>> cf2 = ax2.contourf(xyz, xyz, g_tens[:,1,:], cmap='jet')
        >>> fig.colorbar(cf2, ax=ax2)
        >>> ax3.set_aspect('equal')
        >>> ax3.set_title('3')
        >>> cf3 = ax3.contourf(xyz, xyz, g_tens[:,2,:], cmap='jet')
        >>> fig.colorbar(cf3, ax=ax3)
        >>> plt.show()
        """

        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)
        g_tens = (G * self.mass * np.eye(3)) / r[..., None, None] ** 3 +\
                 (3 * r_vec[..., None] * r_vec[..., None, :]) / r[..., None, None] ** 5
        return g_tens
