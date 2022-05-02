import numpy as np
from scipy.constants import mu_0, epsilon_0

###############################################################################
#                                                                             #
#                              Base Classes                                   #
#                                                                             #
###############################################################################


class BaseEM:
    """Base electromagnetics class.

    The base EM class stores the physical properties that are relevant to all problems
    which solve Maxwell's equations. The base EM class assumes that all physical properties
    (conductivity, magnetic permeability and dielectric permittivity) are homogeneously
    distributed within a wholespace and are non-dispersive. These properties are
    overwritten in child classes as necessary.

    Parameters
    ----------
    sigma : float, optional
        Electrical conductivity in S/m. Default is :math:`\\sigma` = 1 S/m
    mu : float, optional
        Magnetic permeability in H/m. Default is :math:`\\mu_0 = 4\\pi \\times 10^{-7}` H/m
    epsilon : float, optional
        Dielectric permittivity F/m. Default is :math:`\\epsilon_0 = 8.85 \\times 10^{-12}` F/m
    """

    def __init__(self, sigma=1.0, mu=mu_0, epsilon=epsilon_0, **kwargs):

        self.sigma = sigma
        self.mu = mu
        self.epsilon = epsilon
        super().__init__(**kwargs)


    @property
    def sigma(self):
        """Electrical conductivity in S/m

        Returns
        -------
        float
            Electrical conductivity in S/m
        """
        return self._sigma

    @sigma.setter
    def sigma(self, value):

        try:
            value = float(value)
        except:
            raise TypeError(f"sigma must be a number, got {type(value)}")

        if value <= 0.0:
            raise ValueError("sigma must be greater than 0")

        self._sigma = value

    @property
    def mu(self):
        """Magnetic permeability in H/m

        Returns
        -------
        float
            Magnetic permeability in H/m
        """
        return self._mu

    @mu.setter
    def mu(self, value):

        try:
            value = float(value)
        except:
            raise TypeError(f"mu must be a number, got {type(value)}")

        if value <= 0.0:
            raise ValueError("mu must be greater than 0")

        self._mu = value

    @property
    def epsilon(self):
        """Dielectric permittivity in F/m

        Returns
        -------
        float
            dielectric permittivity in F/m
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):

        try:
            value = float(value)
        except:
            raise TypeError(f"epsilon must be a number, got {type(value)}")

        if value <= 0.0:
            raise ValueError("epsilon must be greater than 0")

        self._epsilon = value

class BaseDipole:
    """Base class for dipoles; namely the location and orientation.
    This class is inherited by both electric current and magnetic dipoles.

    Parameters
    ----------
    location: (3) array_like, optional
        Location of the dipole in 3D space. Default is (0,0,0)
    orientation : (3) array_like or {'X','Y','Z'}
        Orientation of the dipole. Can be defined using as an ``array_like`` of length 3,
        or by using one of {'X','Y','Z'} to define a unit dipole along the x, y or z direction.
        Default is 'X'.
    """

    def __init__(self, location=np.r_[0.,0.,0.], orientation='X', **kwargs):

        self.location = location
        self.orientation = orientation
        super().__init__(**kwargs)


    @property
    def location(self):
        """Location of the dipole

        Returns
        -------
        (3) numpy.ndarray of float
            xyz dipole location
        """
        return self._location

    @location.setter
    def location(self, vec):

        try:
            vec = np.asarray(vec, dtype=float)
        except:
            raise TypeError(f"location must be array_like of float, got {type(vec)}")
        vec = np.squeeze(vec)
        if vec.shape != (3, ):
            raise ValueError(
                f"location must be array_like with shape (3,), got {vec.shape}"
            )

        self._location = vec

    @property
    def orientation(self):
        """Orientation of the dipole as a normalized vector

        Returns
        -------
        (3) numpy.ndarray of float or str in {'X','Y','Z'}
            dipole orientation, normalized to unit magnitude
        """
        return self._orientation

    @orientation.setter
    def orientation(self, var):

        if isinstance(var, str):
            if var.upper() == 'X':
                var = np.r_[1., 0., 0.]
            elif var.upper() == 'Y':
                var = np.r_[0., 1., 0.]
            elif var.upper() == 'Z':
                var = np.r_[0., 0., 1.]
        else:
            try:
                var = np.asarray(var, dtype=float)
            except:
                raise TypeError(
                    f"orientation must be str or array_like, got {type(var)}"
                )
            var = np.squeeze(var)
            if var.shape != (3, ):
                raise ValueError(
                    f"orientation must be array_like with shape (3,), got {len(var)}"
                )

            # Normalize the orientation
            var /= np.linalg.norm(var)

        self._orientation = var

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

    def dot_orientation(self, vecs):
        r"""Dot product between the orientation of the source and a gridded set of vectors.

        Where :math:`\mathbf{p}` is the vector defining the dipole moment direction and
        :math:`\mathbf{v}` is a 3D vector, this method returns the dot product:

        .. math::
            \mathbf{p} \cdot \mathbf{v}

        for all vectors :math:`\mathbf{v}` supplied in the input argument ``vecs``.

        Parameters
        ----------
        vecs : (n, 3) numpy.ndarray
            A set of 3D vectors

        Returns
        -------
        (n) numpy.ndarray
            Dot product between the dipole moment direction (orientation) and each vector supplied.
        """
        return np.dot(vecs, self.orientation)

    def cross_orientation(self, xyz):
        r"""Cross products between a gridded set of vectors and the orientation of the source.

        Where :math:`\mathbf{p}` is the vector defining the orientation of the source and
        :math:`\mathbf{v}` is a 3D vector, this method returns the cross product:

        .. math::
            \mathbf{v} \times \mathbf{p}


        for all vectors :math:`\mathbf{v}` supplied in the input argument ``vecs``.

        Parameters
        ----------
        vecs : (n, 3) numpy.ndarray
            A set of 3D vectors

        Returns
        -------
        (n) numpy.ndarray
            Cross product between each vector supplied and the source orientation.

        """
        return np.cross(xyz, self.orientation)


class BaseElectricDipole(BaseDipole):
    r"""Base class for electric current dipoles.

    The ``BaseElectricDipole`` class defines the basic properties for electric
    current dipoles. Where :math:`I` is the current and :math:`d\mathbf{s}` defines
    the direction and length of the dipole, the current dipole is defined as:

    .. math::
        \mathbf{j_s}(\mathbf{r}) = Id \mathbf{s} \, \delta(|\mathbf{r} - \mathbf{r_s}|)

    where :math:`\mathbf{r_s}` is the location of the source.
    For more, visit `EM GeoSci <https://em.geosci.xyz/content/maxwell1_fundamentals/dipole_sources_in_homogeneous_media/electric_dipole_definition/index.html>`__ .

    Parameters
    ----------
    length : float, int
        Length of the electric current dipole (m). Default is 1 m.
    current : float, int
        Current of the electric current dipole (A). Default is 1 A.
    """

    def __init__(self, length=1.0, current=1.0, **kwargs):

        self.length = length
        self.current = current

        super().__init__(**kwargs)

    @property
    def length(self):
        """Length of the electric current dipole (m)

        Returns
        -------
        float
            Length of the electric current dipole (m)
        """
        return self._length

    @length.setter
    def length(self, value):

        try:
            value = float(value)
        except:
            raise TypeError(f"length must be a number, got {type(value)}")

        if value <= 0.0:
            raise ValueError("length must be greater than 0")

        self._length = value

    @property
    def current(self):
        """Current in the electric current dipole (A)

        Returns
        -------
        float
            Current in the electric current dipole (A)
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


class BaseMagneticDipole(BaseDipole):
    r"""Base class for magnetic dipoles.

    The ``BaseMagneticDipole`` class defines the basic properties for magnetic dipoles.
    Where :math:`d\mathbf{m}` defines the dipole moment, the magnetic dipole is defined as:

    .. math::
        \mathbf{s_m}(\mathbf{r}) = \mathbf{m} \, \delta(|\mathbf{r} - \mathbf{r_s}|)

    where :math:`\mathbf{r_s}` is the location of the source.
    For more, visit `EM GeoSci <https://em.geosci.xyz/content/maxwell1_fundamentals/dipole_sources_in_homogeneous_media/index.html>`__ .


    Parameters
    ----------
    moment : float, optional
        Amplitude of the dipole moment for the magnetic dipole (:math:`A/m^2`).
        Default is 1 :math:`A/m^2`.
    """

    def __init__(self, moment=1.0, **kwargs):

        self.moment = moment
        super().__init__(**kwargs)

    @property
    def moment(self):
        """Amplitude of the dipole moment of the magnetic dipole (:math:`A/m^2`)

        Returns
        -------
        float
            Amplitude of the dipole moment of the magnetic dipole (:math:`A/m^2`)
        """
        return self._moment

    @moment.setter
    def moment(self, value):

        try:
            value = float(value)
        except:
            raise TypeError(f"moment must be a number, got {type(value)}")

        if value <= 0.0:
            raise ValueError("moment must be greater than 0")

        self._moment = value


class BaseLineCurrent:
    """Base class for connected segments of current-carrying wire.

    Parameters
    ----------
    nodes : (n, 3) array_like
        Nodes defining the segments of current-carrying wire segments. For an
        inductive source, you must close the loop.
    """

    def __init__(self, nodes, current=1.0, **kwargs):

        self.nodes = nodes
        self.current = current
        super().__init__(**kwargs)

    @property
    def nodes(self):
        """Nodes defining the segments of current-carrying wire segments;
        close the loop if inductive.

        Returns
        -------
        (n_nodes, 3) numpy.ndarray of float
            node locations
        """
        return self._nodes

    @nodes.setter
    def nodes(self, xyz):

        try:
            xyz = np.asarray(xyz, dtype=np.float64)
        except:
            raise TypeError(f"nodes must be array_like, got {type(xyz)}")
        xyz = np.atleast_2d(xyz)

        if (xyz.shape[1] != 3) | (xyz.ndim != 2):
            raise ValueError(
                f"nodes must be array_like with shape (n, 3), got {xyz.shape}"
            )

        self._nodes = xyz

    @property
    def n_segments(self):
        """Number of wire segments

        Returns
        -------
        int
            number of wire segments

        """
        return self.nodes.shape[0] - 1


    @property
    def current(self):
        """Current in wire (A)

        Returns
        -------
        float
            Current in the wire (A)
        """
        return self._current

    @current.setter
    def current(self, value):

        try:
            value = float(value)
        except:
            raise TypeError(f"current must be a number, got {type(value)}")

        self._current = value
