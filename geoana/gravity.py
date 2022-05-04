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
        except TypeError:
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
        except (TypeError, ValueError):
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
        xyz : (3) numpy.ndarray
            point mass location

        Returns
        -------
        numpy.ndarray
            gravitational potential at point mass location xyz
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
        """
        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)
        g_vec = (G * self.mass * r_vec) / r
        return g_vec

    def gravitational_gradient(self, xyz):
        """
        Gravitational gradient for a point mass.

        Parameters
        ----------
        xyz : (3) numpy.ndarray
            point mass location

        Returns
        -------
        numpy.ndarray
            gravitational gradient at point mass location xyz
        """
        r_vec = xyz - self.location
        r = np.linalg.norm(r_vec, axis=-1)
        g_tens = (G * self.mass * np.eye(3)) / r ** 3 + (3 * (r_vec[..., None] * r_vec[..., None, :])) / r ** 5
        return g_tens
