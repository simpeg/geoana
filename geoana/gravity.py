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
    """Class for gravitational solutions for a point mass.

    The ``PointMass`` class is used to analytically compute the gravitational
    potentials, fields, and gradients for a point mass.

        Parameters
        ----------
        mass : float
            Mass of the point particle (kg). Default is m = 1 kg
        location : array_like, optional
            Location of the point mass in 3D space. Default is (0, 0, 0)
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
            Location of the point mass.  Default = np.r_[0,0,0]
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
            Point mass location.

        Returns
        -------
        (..., ) numpy.ndarray
            Gravitational potential at point mass location xyz.

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
        equation 3.3.

        .. math::

            \\mathbf{g} = \\nabla U(P)

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Point mass location.

        Returns
        -------
        (..., 3) numpy.ndarray
            Gravitational field at point mass location xyz.

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

        >>> plt.quiver(xyz[:,0], xyz[:,1], g[:,0], g[:,1])
        >>> plt.xlabel('y')
        >>> plt.ylabel('x')
        >>> plt.title('Gravitational Field Lines at z=0')
        >>> plt.show()
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
            Point mass location.

        Returns
        -------
        (..., 3, 3) numpy.ndarray
            Gravitational gradient at point mass location xyz.

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

        >>> xyz = ndgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20), np.array([0]))
        >>> g_tens = simulation.gravitational_gradient(xyz[None,...] * xyz[...,None,:])

        Finally, we plot the gravitational gradient for each element of the 3 x 3 matrix.

        >>> fig = plt.figure(figsize=(15, 15))
        >>> gs = fig.add_gridspec(3, 3, hspace=0, wspace=0)
        >>> (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) = gs.subplots(sharex='col', sharey='row')
        >>> fig.suptitle('Gravitational Gradients')
        >>> ax1.contourf(g_tens[:,:,0,0], cmap='jet')
        >>> ax2.contourf(g_tens[:,:,0,1], cmap='jet')
        >>> ax3.contourf(g_tens[:,:,0,2], cmap='jet')
        >>> ax4.contourf(g_tens[:,:,1,0], cmap='jet')
        >>> ax5.contourf(g_tens[:,:,1,1], cmap='jet')
        >>> ax6.contourf(g_tens[:,:,1,2], cmap='jet')
        >>> ax7.contourf(g_tens[:,:,2,0], cmap='jet')
        >>> ax8.contourf(g_tens[:,:,2,1], cmap='jet')
        >>> ax9.contourf(g_tens[:,:,2,2], cmap='jet')
        >>> plt.show()
        """

        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)
        g_tens = G * self.mass * (np.eye(3) / r[..., None, None] ** 3 +
                                  (3 * r_vec[..., None] * r_vec[..., None, :]) / r[..., None, None] ** 5)
        return g_tens
