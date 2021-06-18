import numpy as np
import properties
from scipy.constants import mu_0, epsilon_0

from .. import spatial


###############################################################################
#                                                                             #
#                              Base Classes                                   #
#                                                                             #
###############################################################################


class BaseEM(properties.HasProperties):
    """
    Base class for electromanetics. Contains physical properties that are
    relevant to all problems that use Maxwell's equations
    """

    mu = properties.Float(
        "Magnetic permeability (H/m)",
        default=mu_0,
        min=0.0
    )

    sigma = properties.Float(
        "Electrical conductivity (S/m)",
        default=1.0,
        min=0.0
    )

    epsilon = properties.Float(
        "Permitivity value (F/m)",
        default=epsilon_0,
        min=0.0
    )


class BaseDipole(BaseEM):
    """
    Base class for dipoles.
    """

    orientation = properties.Vector3(
        "orientation of dipole",
        default="X",
        length=1.0
    )

    location = properties.Vector3(
        "location of the electric dipole source",
        default="ZERO"
    )

    def vector_distance(self, xyz):
        """
        Vector distance from the dipole location
        :param numpy.ndarray xyz: grid
        """
        return spatial.vector_distance(xyz, np.array(self.location))

    def distance(self, xyz):
        """
        Distance from the dipole location
        """
        return spatial.distance(xyz, np.array(self.location))

    def dot_orientation(self, xyz):
        """
        Take the dot product between a grid and the orientation of the dipole
        """
        return spatial.vector_dot(xyz, np.array(self.orientation))

    def cross_orientation(self, xyz):
        """
        Take the cross product between a grid and the orientation of the dipole
        """
        orientation = np.kron(
            np.atleast_2d(
                np.array(self.orientation)
            ), np.ones((xyz.shape[0], 1))
        )
        return np.cross(xyz, orientation)


class BaseElectricDipole(BaseDipole):
    """
    Base class for electric current dipoles
    """

    length = properties.Float(
        "length of the dipole (m)",
        default=1.0,
        min=0.0
    )

    current = properties.Float(
        "magnitude of the injected current (A)",
        default=1.0,
        min=0.0
    )


class BaseMagneticDipole(BaseDipole):
    """
    Base class for magnetic dipoles
    """

    moment = properties.Float(
        "moment of the dipole (Am^2)",
        default=1.0,
        min=0.0
    )
