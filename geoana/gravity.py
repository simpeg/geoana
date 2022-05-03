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

    def __init__(self, mass=1.0, location=np.r_[0., 0., 0.], **kwargs):

        self.mass = mass
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

        if value <= 0.0:
            raise ValueError("mass must be greater than 0")

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

    def vector_distance(self, xyz):
        r"""Vector distance to a set of gridded xyz locations.

        Where :math:`\mathbf{p}` is the location of the source and :math:`\mathbf{q}`
        is a point in 3D space, this method returns the vector distance:

        .. math::
            \mathbf{v} = \mathbf{q} - \mathbf{p}

        for all locations :math:`\mathbf{q}` supplied in the inputed argument `xyz`.
        For dipoles, :math:`\mathbf{p}` is the dipole location. For circular loops,
        :math:`\mathbf{p}` defines the center location for the loop.

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n, 3) numpy.ndarray
            Vector distances in the x, y and z directions
        """
        return xyz - self.location

    def distance(self, xyz):
        r"""Scalar distance from dipole to a set of gridded xyz locations

        Where :math:`\mathbf{p}` is the location of the source and :math:`\mathbf{q}`
        is a point in 3D space, this method returns the scalar distance:

        .. math::
            d = \sqrt{(q_x - p_x)^2 + (q_y - p_y)^2 + (q_z - p_z)^2}

        for all locations :math:`\mathbf{q}` supplied in the input argument `xyz`.
        For dipoles, :math:`\mathbf{p}` is the dipole location. For circular loops,
        :math:`\mathbf{p}` defines the center location for the loop.

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            Gridded xyz locations

        Returns
        -------
        (n) numpy.ndarray
            Scalar distances from dipole to xyz locations
        """
        return np.linalg.norm(xyz - self.location, axis=-1)

    def gravitational_potential(self, xyz):
        """
        Gravitational potential for a point mass.  See Blakely, 1996
        equation 3.4

        .. math::

            U(P) = \\gamma \\frac{m}{r}

        """
        r = self.distance(xyz)
        u_g = (G * self.mass) / r
        return u_g

    def gravitational_field(self, xyz):
        """
        Gravitational field for a point mass.  See Blakely, 1996
        equation 3.3

        .. math::

            \\mathbf{g} = \\nabla U(P)

        """
        r_vec = self.vector_distance(xyz)
        r = self.distance(xyz)
        g_vec = (G * self.mass * r_vec) / r
        return g_vec

    def gravitational_gradient(self, xyz):
        """
        Gravitational gradient for a point mass.

        """
        r_vec = self.vector_distance(xyz)
        r = self.distance(xyz)
        gg_tens = (G * self.mass * np.eye) / r ** 3 + (3 * np.outer(r_vec, r_vec)) / r ** 5
        return gg_tens

