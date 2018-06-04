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

def vector_magnitude(v):
    """
    Amplitude of a vector, v.

    **Required**

    :param numpy.array v: vector array :code:`np.r_[x, y, z]`, with shape
                          (n, 3)

    **Returns**

    :returns: magnitude of a vector (n, 1)
    :rtype: numpy.ndarray
    """

    assert (v.shape[1] == 3), (
        "the vector, v, should be npoints by 3. The shape provided is {}".format(
            v.shape
        )
    )

    return np.sqrt((v**2).sum(axis=1))


def vector_distance(xyz, origin=np.r_[0., 0., 0.]):
    """
    Vector distance of a grid, xyz from an origin origin.

    **Required**

    :param numpy.array xyz: grid (npoints x 3)

    **Optional**

    :param numpy.array origin: origin (default: [0., 0., 0.])

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

    dx = xyz[:, 0] - origin[0]
    dy = xyz[:, 1] - origin[1]
    dz = xyz[:, 2] - origin[2]

    return np.c_[dx, dy, dz]


def distance(xyz, origin=np.r_[0., 0., 0.]):
    """
    Radial distance from an grid of points to the origin

    **Required**

    :param numpy.array xyz: grid (npoints x 3)

    **Optional**

    :param numpy.array origin: origin (default: [0., 0., 0.])

    **Returns**

    :returns: distance between each point and the origin (npoints x 1)
    :rtype: nunmpy.ndarray
    """
    dxyz = vector_distance(xyz, origin)
    return vector_magnitude(dxyz)


def vector_dot(xyz, vector):
    """
    Take a dot product between an array of vectors, xyz and a vector [x, y, z]

    **Required**

    :param numpy.array xyz: grid (npoints x 3)
    :param numpy.array vector: vector (1 x 3)

    **Returns**

    :returns: dot product between the grid and the (1 x 3) vector, returns an
              (npoints x 1) array
    :rtype: numpy.ndarray
    """
    assert len(vector) == 3, "vector should be length 3"
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
