from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import properties
from scipy.constants import mu_0, pi, epsilon_0


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
        default='X',
        length=1.0
    )

    location = properties.Vector3(
        "location of the electric dipole source",
        default='ZERO'
    )

    def offset_from_location(self, xyz):

        # TODO: validate stuff
        # xyz = Utils.asArray_N_x_Dim(xyz, 3)

        return np.c_[
            xyz[:, 0] - self.location[0],
            xyz[:, 1] - self.location[1],
            xyz[:, 2] - self.location[2]
        ]

    def distance_from_location(self, xyz):
        return np.sqrt((self.offset_from_location(xyz)**2).sum(axis=1))


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
        "size of the injected current (A)",
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
