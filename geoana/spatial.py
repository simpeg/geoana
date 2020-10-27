from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from .utils import mkvc


def cylindrical_2_cartesian(grid, vec=None):
    """
    Take a grid or vector (if provided)
    defined in cylindrical coordinates :math:`(r, \\theta, z)` and
    transform it to cartesian coordinates, :math:`(x, y, z)`.

    **Required**

    :param numpy.ndarray grid: grid in cylindrical coordinates
                               :math:`(r, \\theta, z)`

    **Optional**

    :param numpy.ndarray vec: (optional) vector defined in cylindrical
                              coordinates

    **Returns**

    :return: grid or vector (if provided) in cartesian coordinates
             :math:`(x, y, z)`
    :rtype: numpy.ndarray

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


def cartesian_2_cylindrical(grid, vec=None):
    """
    Takes a grid or vector (if provided)
    defined in cartesian coordinates :math:`(x, y, z)` and
    transform it to cylindrical coordinates, :math:`(r, \\theta, z)`.

    **Required**

    :param numpy.ndarray grid: grid in cartesian coordinates
                               :math:`(x, y, z)`

    **Optional**

    :param numpy.ndarray vec: (optional) vector defined in cartesian
                              coordinates

    **Returns**

    :return: grid or vector (if provided) in cylindrical coordinates
             :math:`(r, \\theta, z)`
    :rtype: numpy.ndarray
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


def spherical_2_cartesian(grid, vec=None):
    """
    Take a grid or vector (if provided)
    defined in spherical coordinates :math:`(r, \\theta, \\phi)` and
    transform it to cartesian coordinates, :math:`(x, y, z)`.

    **Required**

    :param numpy.ndarray grid: grid in spherical coordinates
                             :math:`(r, \\theta, \\phi)`

    **Optional**

    :param numpy.ndarray vec: (optional) vector defined in spherical
                              coordinates

    **Returns**

    :return: grid or vector (if provided) in cartesian coordinates
             :math:`(x, y, z)`
    :rtype: numpy.ndarray

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


def cartesian_2_spherical(grid, vec=None):
    """
    Takes a grid or vector (if provided)
    defined in cartesian coordinates :math:`(x, y, z)` and
    transform it to spherical coordinates, :math:`(r, \\theta, \\phi)`.

    **Required**

    :param numpy.ndarray grid: grid in cartesian coordinates
                               :math:`(x, y, z)`

    **Optional**

    :param numpy.ndarray vec: (optional) vector defined in cartesian
                              coordinates

    **Returns**

    :return: grid or vector (if provided) in spherical coordinates
             :math:`(r, \\theta, \\phi)`
    :rtype: numpy.ndarray
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
    """
    Amplitude of a vector, v.

    **Required**

    :param numpy.ndarray v: vector array

    **Returns**

    :returns: magnitude of a vector (n, 1)
    :rtype: numpy.ndarray
    """

    v = np.atleast_2d(v)

    return np.sqrt((v**2).sum(axis=1))


def vector_distance(xyz, origin=np.r_[0., 0., 0.]):
    """
    Vector distance of a grid, xyz from an origin origin.

    **Required**

    :param numpy.ndarray xyz: grid (npoints x 3)

    **Optional**

    :param numpy.ndarray origin: origin (default: [0., 0., 0.])

    **Returns**

    :returns: vector distance from a grid of points from the origin
              (npoints x 3)
    :rtype: numpy.ndarray
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
    """
    Radial distance from an grid of points to the origin

    **Required**

    :param numpy.ndarray xyz: grid (npoints x 3)

    **Optional**

    :param numpy.ndarray origin: origin (default: [0., 0., 0.])

    **Returns**

    :returns: distance between each point and the origin (npoints x 1)
    :rtype: numpy.ndarray
    """
    dxyz = vector_distance(xyz, origin)
    return vector_magnitude(dxyz)


def vector_dot(xyz, vector):
    """
    Take a dot product between an array of vectors, xyz and a vector [x, y, z]

    **Required**

    :param numpy.ndarray xyz: grid (npoints x 3)
    :param numpy.ndarray vector: vector (1 x 3)

    **Returns**

    :returns: dot product between the grid and the (1 x 3) vector, returns an
              (npoints x 1) array
    :rtype: numpy.ndarray
    """
    if len(vector) != 3:
        raise Exception(
            "vector should be length 3, the provided length is {}".format(
                len(vector)
            )
        )
    return vector[0]*xyz[:, 0] + vector[1]*xyz[:, 1] + vector[2]*xyz[:, 2]


def repeat_scalar(scalar, dim=3):
    """
    Repeat a spatially distributed scalar value dim times to simplify
    multiplication with a vector.

    **Required**

    :param numpy.ndarray scalar: (n x 1) array of scalars

    **Optional**

    :param int dim: dimension of the second axis for the output (default = 3)

    **Returns**

    :returns: (n x dim) array of the repeated vector
    :rtype: numpy.ndarray
    """
    assert len(scalar) in scalar.shape, (
        "input must be a scalar. The shape you provided is {}".format(
            scalar.shape
        )
    )

    return np.kron(np.ones((1, dim)), np.atleast_2d(scalar).T)

def rotation_matrix_from_normals(v0, v1, tol=1e-20):
    """
    Performs the minimum number of rotations to define a rotation from the
    direction indicated by the vector n0 to the direction indicated by n1.
    The axis of rotation is n0 x n1
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    :param numpy.ndarray v0: vector of length 3
    :param numpy.ndarray v1: vector of length 3
    :param float tol: tolerance. If the norm of the cross product between the
                        two vectors is below this, no rotation is performed
                        default = 1e-20
    :rtype: numpy.ndarray
    :return: 3 x 3 rotation matrix which rotates the frame so that n0 is
             aligned with n1
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
    """
    rotates a grid so that the vector n0 is aligned with the vector n1

    **Required**

    :param numpy.ndarray xyz:
    :param numpy.ndarray n0: vector of length 3, should have norm 1
    :param numpy.ndarray n1: vector of length 3, should have norm 1

    **Optional**

    :param numpy.ndarray x0: vector of length 3, point about which we perform the
                           rotation

    **Returns**

    :rtype: numpy.ndarray
    :return: (3x3) rotation matrix which rotates the frame so that n0 is
             aligned with n1

    """

    R = rotation_matrix_from_normals(n0, n1)

    if xyz.shape[1] != 3:
        raise AssertionError("Grid xyz should be 3 wide")

    return (xyz - x0)@R.T + x0
