from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import properties


def vector_distance(xyz, origin=np.r_[0., 0., 0.]):
    """
    Vector distance of a grid, xyz from an origin origin.
    :param numpy.array xyz: grid (npoints x 3)
    :param numpy.array origin: origin (default: [0., 0., 0.])
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
    Radial distance from an origin origin
    :param numpy.array xyz: grid (npoints x 3)
    :param numpy.array origin: origin (default: [0., 0., 0.])
    """
    dxyz = vector_distance(xyz, origin)
    return np.sqrt((dxyz**2).sum(axis=1))


def vector_dot(xyz, vector):
    """
    Take a dot product between an array of vectors, xyz and a vector [x, y, z]
    :param numpy.array xyz: grid (npoints x 3)
    :param numpy.array vector: vector (1 x 3)
    """
    assert len(vector) == 3, "vector should be length 3"
    return np.hstack([
        vector[0]*xyz[:, 0], vector[1]*xyz[:, 1], vector[2]*xyz[:, 2]
    ])
