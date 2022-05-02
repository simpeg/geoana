"""
=================================================
Spatial Utilities (:mod:`geoana.spatial`)
=================================================
.. currentmodule:: geoana.spatial

The ``geoana.spatial`` module provides rudimentary functions for vectors and
functions for converting between Cartesian, cylindrical and spherical coordinates.


.. autosummary::
  :toctree: generated/

  cylindrical_to_cartesian
  cartesian_to_cylindrical
  vector_magnitude
  vector_distance
  distance
  vector_dot
  repeat_scalar
  rotation_matrix_from_normals
  rotate_points_from_normals

"""
import numpy as np
from .utils import mkvc


def cylindrical_to_cartesian(grid, vec=None):
    """
    Transform gridded locations or a set of vectors from cylindrical coordinates
    :math:`(r, \\phi, z)` to Cartesian coordinates :math:`(x, y, z)`.
    The azimuthal angle :math:`\\phi` is given in radians.

    Parameters
    ----------
    grid : (n, 3) array_like
        Location points defined in cylindrical coordinates :math:`(r, \\phi, z)`.
    vec : (n, 3) array_like, optional
        Vectors defined in cylindrical coordinates :math:`(v_r, v_\\phi, v_z)` at the
        gridded locations. Will also except a flattend array in column major order
        with the same number of elements.

    Returns
    -------
    (n, 3) numpy.ndarray
        If `vec` is ``None``, this returns the transformed `grid` array, otherwise
        this is the transformed `vec` array.

    Examples
    --------
    Here, we convert a series of vectors in 3D space from cylindrical coordinates
    to Cartesian coordinates.

    >>> from geoana.spatial import cylindrical_to_cartesian
    >>> import numpy as np

    Construct original set of vectors in cylindrical coordinates

    >>> r = np.ones(9)
    >>> phi = np.linspace(0, 2*np.pi, 9)
    >>> z = np.linspace(-4., 4., 9)
    >>> u = np.c_[r, phi, z]
    >>> u
    array([[ 1.        ,  0.        , -4.        ],
           [ 1.        ,  0.78539816, -3.        ],
           [ 1.        ,  1.57079633, -2.        ],
           [ 1.        ,  2.35619449, -1.        ],
           [ 1.        ,  3.14159265,  0.        ],
           [ 1.        ,  3.92699082,  1.        ],
           [ 1.        ,  4.71238898,  2.        ],
           [ 1.        ,  5.49778714,  3.        ],
           [ 1.        ,  6.28318531,  4.        ]])

    Create equivalent set of vectors in Cartesian coordinates

    >>> v = cylindrical_to_cartesian(u)
    >>> v
    array([[ 1.00000000e+00,  0.00000000e+00, -4.00000000e+00],
           [ 7.07106781e-01,  7.07106781e-01, -3.00000000e+00],
           [ 6.12323400e-17,  1.00000000e+00, -2.00000000e+00],
           [-7.07106781e-01,  7.07106781e-01, -1.00000000e+00],
           [-1.00000000e+00,  1.22464680e-16,  0.00000000e+00],
           [-7.07106781e-01, -7.07106781e-01,  1.00000000e+00],
           [-1.83697020e-16, -1.00000000e+00,  2.00000000e+00],
           [ 7.07106781e-01, -7.07106781e-01,  3.00000000e+00],
           [ 1.00000000e+00, -2.44929360e-16,  4.00000000e+00]])
    """
    grid = np.atleast_2d(grid)

    if vec is None:
        return np.hstack([
            mkvc(grid[:, 0]*np.cos(grid[:, 1]), 2),
            mkvc(grid[:, 0]*np.sin(grid[:, 1]), 2),
            mkvc(grid[:, 2], 2)
        ])

    if len(vec.shape) == 1 or vec.shape[1] == 1:
        vec = vec.reshape(grid.shape, order='F')

    x = vec[:, 0] * np.cos(grid[:, 1]) - vec[:, 1] * np.sin(grid[:, 1])
    y = vec[:, 0] * np.sin(grid[:, 1]) + vec[:, 1] * np.cos(grid[:, 1])

    newvec = [x, y]
    if grid.shape[1] == 3:
        z = vec[:, 2]
        newvec += [z]

    return np.vstack(newvec).T


def cartesian_to_cylindrical(grid, vec=None):
    """
    Transform gridded locations or a set of vectors from Cartesian coordinates
    :math:`(x, y, z)` to cylindrical coordinates :math:`(r, \\phi, z)`. Where
    the azimuthal angle :math:`\\phi \\in [-\\pi , \\pi ]` will be given output
    in radians.

    Parameters
    ----------
    grid : (n, 3) array_like
        Gridded locations defined in Cartesian coordinates :math:`(x, y z)`.
    vec : (n, 3) array_like, optional
        Vectors defined in Cartesian coordinates :math:`(v_x, v_y, v_z)` at the
        gridded locations. Also accepts a flattened array with the same total
        elements in column major order.

    Returns
    -------
    (n, 3) numpy.ndarray
        If `vec` is ``None``, this returns the transformed `grid` array, otherwise
        this is the transformed `vec` array.

    Examples
    --------
    Here, we convert a series of vectors in 3D space from Cartesian coordinates
    to cylindrical coordinates.

    >>> from geoana.spatial import cartesian_to_cylindrical
    >>> import numpy as np

    Create set of vectors in Cartesian coordinates

    >>> r = np.ones(9)
    >>> phi = np.linspace(0, 2*np.pi, 9)
    >>> z = np.linspace(-4., 4., 9)
    >>> x = r*np.cos(phi)
    >>> y = r*np.sin(phi)
    >>> u = np.c_[x, y, z]
    >>> u
    array([[ 1.00000000e+00,  0.00000000e+00, -4.00000000e+00],
           [ 7.07106781e-01,  7.07106781e-01, -3.00000000e+00],
           [ 6.12323400e-17,  1.00000000e+00, -2.00000000e+00],
           [-7.07106781e-01,  7.07106781e-01, -1.00000000e+00],
           [-1.00000000e+00,  1.22464680e-16,  0.00000000e+00],
           [-7.07106781e-01, -7.07106781e-01,  1.00000000e+00],
           [-1.83697020e-16, -1.00000000e+00,  2.00000000e+00],
           [ 7.07106781e-01, -7.07106781e-01,  3.00000000e+00],
           [ 1.00000000e+00, -2.44929360e-16,  4.00000000e+00]])

    Compute equivalent set of vectors in cylindrical coordinates

    >>> v = cartesian_to_cylindrical(u)
    >>> v
    array([[ 1.00000000e+00,  0.00000000e+00, -4.00000000e+00],
           [ 1.00000000e+00,  7.85398163e-01, -3.00000000e+00],
           [ 1.00000000e+00,  1.57079633e+00, -2.00000000e+00],
           [ 1.00000000e+00,  2.35619449e+00, -1.00000000e+00],
           [ 1.00000000e+00,  3.14159265e+00,  0.00000000e+00],
           [ 1.00000000e+00, -2.35619449e+00,  1.00000000e+00],
           [ 1.00000000e+00, -1.57079633e+00,  2.00000000e+00],
           [ 1.00000000e+00, -7.85398163e-01,  3.00000000e+00],
           [ 1.00000000e+00, -2.44929360e-16,  4.00000000e+00]])
    """

    grid = np.atleast_2d(grid)

    if vec is None:
        return np.hstack([
            mkvc(np.sqrt(grid[:, 0]**2 + grid[:, 1]**2), 2),
            mkvc(np.arctan2(grid[:, 1], grid[:, 0]), 2),
            mkvc(grid[:, 2], 2)
        ])

    if len(vec.shape) == 1 or vec.shape[1] == 1:
        vec = vec.reshape(grid.shape, order='F')

    theta = np.arctan2(grid[:, 1], grid[:, 0])

    return np.hstack([
        mkvc(np.cos(theta)*vec[:, 0] + np.sin(theta)*vec[:, 1], 2),
        mkvc(-np.sin(theta)*vec[:, 0] + np.cos(theta)*vec[:, 1], 2),
        mkvc(vec[:, 2], 2)
    ])


def spherical_to_cartesian(grid, vec=None):
    """
    Transform gridded locations of a set of vectors from spherical coordinates
    :math:`(r, \\phi, \\theta)` to Cartesian coordinates :math:`(x, y, z)`.
    :math:`\\phi` and :math:`\\theta` are the azimuthal and polar angles, respectively.
    :math:`\\phi` and :math:`\\theta` are given in radians.

    Parameters
    ----------
    grid : (n, 3) array_like
        Gridded locations defined in spherical coordinates :math:`(r, \\phi, \\theta)`.
    vec : (n, 3) array_like, optional
        Vectors defined in spherical coordinates :math:`(v_r, v_\\phi, v_\\theta)` at the
        gridded locations. Will also except a flattend array in column major order with the
        same number of elements.

    Returns
    -------
    (n, 3) numpy.ndarray
        If `vec` is ``None``, this returns the transformed `grid` array, otherwise
        this is the transformed `vec` array.

    Examples
    --------
    Here, we convert a series of vectors in 3D space from spherical coordinates
    to Cartesian coordinates.

    >>> from geoana.spatial import spherical_to_cartesian
    >>> import numpy as np

    Construct original set of vectors in spherical coordinates

    >>> r = np.ones(9)
    >>> phi = np.linspace(0, 2*np.pi, 9)
    >>> theta = np.linspace(0, np.pi, 9)
    >>> u = np.c_[r, phi, theta]
    >>> u
    array([[1.        , 0.        , 0.        ],
           [1.        , 0.78539816, 0.39269908],
           [1.        , 1.57079633, 0.78539816],
           [1.        , 2.35619449, 1.17809725],
           [1.        , 3.14159265, 1.57079633],
           [1.        , 3.92699082, 1.96349541],
           [1.        , 4.71238898, 2.35619449],
           [1.        , 5.49778714, 2.74889357],
           [1.        , 6.28318531, 3.14159265]])

    Create equivalent set of vectors in Cartesian coordinates

    >>> v = spherical_to_cartesian(u)
    >>> v
    array([[ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
           [ 2.70598050e-01,  2.70598050e-01,  9.23879533e-01],
           [ 4.32978028e-17,  7.07106781e-01,  7.07106781e-01],
           [-6.53281482e-01,  6.53281482e-01,  3.82683432e-01],
           [-1.00000000e+00,  1.22464680e-16,  6.12323400e-17],
           [-6.53281482e-01, -6.53281482e-01, -3.82683432e-01],
           [-1.29893408e-16, -7.07106781e-01, -7.07106781e-01],
           [ 2.70598050e-01, -2.70598050e-01, -9.23879533e-01],
           [ 1.22464680e-16, -2.99951957e-32, -1.00000000e+00]])
    """
    grid = np.atleast_2d(grid)

    if vec is None:
        return np.hstack([
            mkvc(grid[:, 0] * np.sin(grid[:, 2]) * np.cos(grid[:, 1]), 2),
            mkvc(grid[:, 0] * np.sin(grid[:, 2]) * np.sin(grid[:, 1]), 2),
            mkvc(grid[:, 0] * np.cos(grid[:, 2]), 2)
        ])

    if len(vec.shape) == 1 or vec.shape[1] == 1:
        vec = vec.reshape(grid.shape, order='F')

    x = (
        vec[:, 0] * np.sin(grid[:, 2]) * np.cos(grid[:, 1]) +
        vec[:, 2] * np.cos(grid[:, 2]) * np.cos(grid[:, 1]) -
        vec[:, 1] * np.sin(grid[:, 1])
    )
    y = (
        vec[:, 0] * np.sin(grid[:, 2]) * np.sin(grid[:, 1]) +
        vec[:, 2] * np.cos(grid[:, 2]) * np.sin(grid[:, 1]) -
        vec[:, 1] * np.cos(grid[:, 1])
    )
    z = (
        vec[:, 0] * np.cos(grid[:, 2]) -
        vec[:, 2] * np.sin(grid[:, 2])
    )

    newvec = [x, y, z]

    return np.vstack(newvec).T


def cartesian_to_spherical(grid, vec=None):
    """
    Transform gridded locations or a set of vectors from Cartesian coordinates
    :math:`(x, y, z)` to  spherical coordinates :math:`(r, \\phi, \\theta)`.
    :math:`\\phi` and :math:`\\theta` are the azimuthal and polar angle, respectively.
    :math:`\\phi` and :math:`\\theta` are given in radians.

    Parameters
    ----------
    grid : (n, 3) array_like
        Gridded locations defined in Cartesian coordinates :math:`(x, y, z)`.
    vec : (n, 3) array_like, optional
        Vectors defined in Cartesian coordinates :math:`(v_x, v_y, v_z)` at the
        gridded locations. Will also except a flattend array in column major order with the
        same number of elements.

    Returns
    -------
    (n, 3) numpy.ndarray
        If `vec` is ``None``, this returns the transformed `grid` array, otherwise
        this is the transformed `vec` array.

    Examples
    --------
    Here, we convert a series of vectors in 3D space from Cartesian coordinates
    to spherical coordinates.

    >>> from geoana.spatial import cartesian_to_spherical
    >>> import numpy as np

    Construct original set of vectors in cartesian coordinates

    >>> r = np.ones(9)
    >>> phi = np.linspace(0, 2*np.pi, 9)
    >>> theta = np.linspace(0., np.pi, 9)
    >>> x = r*np.sin(theta)*np.cos(phi)
    >>> y = r*np.sin(theta)*np.sin(phi)
    >>> z = r*np.cos(theta)
    >>> u = np.c_[x, y, z]
    >>> u
    array([[ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
           [ 2.70598050e-01,  2.70598050e-01,  9.23879533e-01],
           [ 4.32978028e-17,  7.07106781e-01,  7.07106781e-01],
           [-6.53281482e-01,  6.53281482e-01,  3.82683432e-01],
           [-1.00000000e+00,  1.22464680e-16,  6.12323400e-17],
           [-6.53281482e-01, -6.53281482e-01, -3.82683432e-01],
           [-1.29893408e-16, -7.07106781e-01, -7.07106781e-01],
           [ 2.70598050e-01, -2.70598050e-01, -9.23879533e-01],
           [ 1.22464680e-16, -2.99951957e-32, -1.00000000e+00]])

    Compute equivalent set of vectors in spherical coordinates

    >>> v = cartesian_to_spherical(u)
    >>> v
    array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
           [ 1.00000000e+00,  7.85398163e-01,  3.92699082e-01],
           [ 1.00000000e+00,  1.57079633e+00,  7.85398163e-01],
           [ 1.00000000e+00,  2.35619449e+00,  1.17809725e+00],
           [ 1.00000000e+00,  3.14159265e+00,  1.57079633e+00],
           [ 1.00000000e+00, -2.35619449e+00,  1.96349541e+00],
           [ 1.00000000e+00, -1.57079633e+00,  2.35619449e+00],
           [ 1.00000000e+00, -7.85398163e-01,  2.74889357e+00],
           [ 1.00000000e+00, -2.44929360e-16,  3.14159265e+00]])
    """

    grid = np.atleast_2d(grid)

    if vec is None:
        return np.hstack([
            mkvc(np.sqrt(grid[:, 0]**2 + grid[:, 1]**2 + grid[:, 2]**2), 2),
            mkvc(np.arctan2(grid[:, 1], grid[:, 0]), 2),
            mkvc(
                np.arctan2(np.sqrt(grid[:, 0]**2 + grid[:, 1]**2), grid[:, 2]),
                2
            ),
        ])

    if len(vec.shape) == 1 or vec.shape[1] == 1:
        vec = vec.reshape(grid.shape, order='F')

    theta = np.arctan2(grid[:, 1], grid[:, 0])
    phi = np.arctan2(np.sqrt(grid[:, 0]**2 + grid[:, 1]**2), grid[:, 2])

    r = (
        vec[:, 0] * np.sin(phi) * np.cos(theta) +
        vec[:, 1] * np.sin(phi) * np.sin(theta) +
        vec[:, 2] * np.cos(phi)
    )

    theta = - vec[:, 0] * np.sin(theta) + vec[:, 1] * np.cos(theta)

    phi = (
        vec[:, 0] * np.cos(phi) * np.cos(theta) +
        vec[:, 1] * np.cos(phi) * np.sin(theta) -
        vec[:, 2] * np.sin(phi)
    )

    newvec = [r, theta, phi]

    return np.vstack(newvec).T


def vector_magnitude(v):
    """Compute the amplitudes for a set of input vectors in Cartesian coordinates.

    Parameters
    ----------
    (n, dim) numpy.ndarray
        A set of input vectors (2D or 3D)

    Returns
    -------
    (n) numpy.ndarray
        Magnitudes of the vectors
    """

    v = np.atleast_2d(v)

    return np.sqrt((v**2).sum(axis=1))


def vector_distance(xyz, origin=np.r_[0., 0., 0.]):
    r"""Vector distances from a reference location to a set of xyz locations.

    Where :math:`\mathbf{p}` is an reference location define by the input argument
    `origin`, this method returns the vector distance:

    .. math::
        \mathbf{v} = \mathbf{q} - \mathbf{p}

    for all locations :math:`\mathbf{q}` supplied in the input argument `xyz`.
    By default, the reference location is (0,0,0).

    Parameters
    ----------
    xyz : (n, 3) numpy.ndarray
        Gridded xyz locations
    origin : (3) array_like, optional
        Reference location. Default = np.r_[0,0,0]

    Returns
    -------
    (n, 3) numpy.ndarray
        Vector distances along the x, y and z directions

    """
    assert(xyz.shape[1] == 3), (
        "the xyz grid should be npoints by 3, the shape provided is {}".format(
            xyz.shape
        )
    )

    if len(origin) != 3:
        raise Exception(
            "the origin must be length 3, the length provided is {}".format(
                len(origin)
            )
        )


    dx = xyz[:, 0] - origin[0]
    dy = xyz[:, 1] - origin[1]
    dz = xyz[:, 2] - origin[2]

    return np.c_[dx, dy, dz]


def distance(xyz, origin=np.r_[0., 0., 0.]):
    r"""Scalar distances between a reference location to a set of xyz locations.

    Where :math:`\mathbf{p}` is an reference location define by the input argument
    `origin`, this method returns the scalar distance:

    .. math::
        d = \big | \mathbf{q} - \mathbf{p} \big |

    for all locations :math:`\mathbf{q}` supplied in the input argument `xyz`.
    By default, the reference location is (0,0,0).

    Parameters
    ----------
    xyz : (n, 3) numpy.ndarray
        Gridded xyz locations
    origin : (3) array_like (optional)
        Reference location. Default = np.r_[0,0,0]

    Returns
    -------
    (n) numpy.ndarray
        Scalar distances
    """
    dxyz = vector_distance(xyz, origin)
    return vector_magnitude(dxyz)


def vector_dot(xyz, vector):
    r"""Dot product between a vector and a set of vectors.

    Where :math:`\mathbf{u}` is a vector defined by the input argument `vector`,
    this method returns the dot products between :math:`\mathbf{u}` and every
    vector supplied in the input argument `xyz`. I.e. for all
    :math:`\mathbf{v} \\in \mathbf{V}`, we compute:

    .. math::
        \mathbf{u} \cdot \mathbf{v}

    Parameters
    ----------
    xyz : (n, 3) numpy.ndarray
        A set of 3D vectors
    vector : (3) numpy.array_like
        A single 3D vector

    Returns
    -------
    (n) numpy.ndarray
        Dot products between a vector and a set of vectors.
    """
    if len(vector) != 3:
        raise Exception(
            "vector should be length 3, the provided length is {}".format(
                len(vector)
            )
        )
    return vector[0]*xyz[:, 0] + vector[1]*xyz[:, 1] + vector[2]*xyz[:, 2]


def repeat_scalar(scalar, dim=3):
    """Stack spatially distributed scalar values to simplify multiplication with a vector.

    Where the input argument `scalar` defines a set of *n* spatially distributed scalar
    values, ``repeat_scalar`` repeats and stacks the input array to produce an
    (`n`, `dim`) array which is better for multiplying the scalar values with vector arrays.

    Parameters
    ----------
    scalar : (n) numpy.ndarray
        The set of scalar values
    dim : int, optional
        The dimension. Default=3

    Returns
    -------
    (n, dim) numpy.ndarray
        The repeated and stacked set of scalar values
    """
    assert len(scalar) in scalar.shape, (
        "input must be a scalar. The shape you provided is {}".format(
            scalar.shape
        )
    )

    return np.kron(np.ones((1, dim)), np.atleast_2d(scalar).T)

def rotation_matrix_from_normals(v0, v1, tol=1e-20):
    """
    Generate a 3x3 rotation matrix defining the rotation from vector v0 to v1.

    This function uses Rodrigues' rotation formula to generate the rotation
    matrix :math:`\\mathbf{A}` going from vector :math:`\\mathbf{v_0}` to
    vector :math:`\\mathbf{v_1}`. Thus:

    .. math::
        \\mathbf{Av_0} = \\mathbf{v_1}

    For detailed desciption of the algorithm, see
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    Parameters
    ----------
    v0 : (3) numpy.ndarray
        Starting orientation direction
    v1 : (3) numpy.ndarray
        Finishing orientation direction
    tol : float, optional
        Numerical tolerance. If the length of the rotation axis is below this value,
        it is assumed to be no rotation, and an identity matrix is returned.

    Returns
    -------
    (3, 3) numpy.ndarray
        The rotation matrix from v0 to v1.
    """

    v0 = mkvc(v0)
    v1 = mkvc(v1)

    # ensure both n0, n1 are vectors of length 1
    assert len(v0) == 3, "Length of n0 should be 3"
    assert len(v1) == 3, "Length of n1 should be 3"

    # ensure both are true normals
    n0 = v0*1./np.linalg.norm(v0)
    n1 = v1*1./np.linalg.norm(v1)

    n0dotn1 = n0.dot(n1)

    # define the rotation axis, which is the cross product of the two vectors
    rotAx = np.cross(n0, n1)

    if np.linalg.norm(rotAx) < tol:
        return np.eye(3, dtype=float)

    rotAx *= 1./np.linalg.norm(rotAx)

    cosT = n0dotn1/(np.linalg.norm(n0)*np.linalg.norm(n1))
    sinT = np.sqrt(1.-n0dotn1**2)

    ux = np.array(
        [
            [0., -rotAx[2], rotAx[1]],
            [rotAx[2], 0., -rotAx[0]],
            [-rotAx[1], rotAx[0], 0.]
        ], dtype=float
    )

    return np.eye(3, dtype=float) + sinT*ux + (1.-cosT)*(ux.dot(ux))


def rotate_points_from_normals(xyz, n0, n1, x0=np.r_[0., 0., 0.]):
    """Rotate a set of xyz locations about a specified point.

    Rotate a grid of Cartesian points about a location x0 according to the
    rotation defined from vector v0 to v1.

    Let :math:`\\mathbf{x}` represent an input xyz location, let :math:`\\mathbf{x_0}` be
    the origin of rotation, and let :math:`\\mathbf{R}` denote the rotation matrix from
    vector v0 to v1. Where :math:`\\mathbf{x'}` is the new xyz location, this function
    outputs the following operation for all input locations:

    .. math::
        \\mathbf{x'} = \\mathbf{R (x - x_0)} + \\mathbf{x_0}

    Parameters
    ----------
    xyz : (n, 3) numpy.ndarray
        locations to rotate
    v0 : (3) numpy.ndarray
        Starting orientation direction
    v1 : (3) numpy.ndarray
        Finishing orientation direction
    x0 : (3) numpy.ndarray, optional
        The origin of rotation.

    Returns
    -------
    (n, 3) numpy.ndarray
        The rotated xyz locations.
    """

    R = rotation_matrix_from_normals(n0, n1)

    if xyz.shape[1] != 3:
        raise AssertionError("Grid xyz should be 3 wide")

    return (xyz - x0)@R.T + x0



# Aliases
def cylindrical_2_cartesian(grid, vec=None):
    """An alias for cylindrical_to_cartesian"""
    return cylindrical_to_cartesian(grid, vec)

def cartesian_2_cylindrical(grid, vec=None):
    """An alias for cartesian_to_cylindrical"""
    return cartesian_to_cylindrical(grid, vec)

def spherical_2_cartesian(grid, vec=None):
    """An alias for spherical_to_cartesian"""
    return spherical_to_cartesian(grid, vec)

def cartesian_2_spherical(grid, vec=None):
    """An Alias for cartesian_to_spherical"""
    return cartesian_to_spherical(grid, vec)
