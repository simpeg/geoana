from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ..base import BaseAnalytic
from .. import traits as tr

import numpy as np
from scipy.constants import mu_0, pi, epsilon_0


class BaseEM(BaseAnalytic):

    mu = tr.Float(
        help="Magnetic permeability.",
        default_value=mu_0,
        min=0.0
    )

    sigma = tr.Float(
        help="Electrical conductivity (S/m)",
        default_value=1.0,
        min=0.0
    )

    epsilon = tr.Float(
        help="Permitivity value",
        default_value=epsilon_0,
        min=0.0
    )


class BaseDipole(BaseEM):

    orientation = tr.Vector(
        help="orientation of dipole",
        default_value='X',
        normalize=True
    )

    location = tr.Vector(
        help="location of the electric dipole source",
        default_value='ZERO'
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


class BaseFDEM(BaseEM):

    frequency = tr.Float(
        help="Source frequency (Hz)",
        default_value=1e2,
        min=0.0
    )

    @property
    def omega(self):
        return 2.0*pi*self.frequency

    @property
    def sigma_hat(self):
        return self.sigma + 1j*self.omega*self.epsilon

    @property
    def wave_number(self):
        np.sqrt(
            self.omega**2. * self.mu * self.epsilon -
            1j * self.omega * self.mu * self.sigma
        )


class BaseElectricDipole(BaseDipole):

    length = tr.Float(
        help="length of the dipole (m)",
        default_value=1.0,
        min=0.0
    )

    current = tr.Float(
        help="size of the injected current (A)",
        default_value=1.0,
        min=0.0
    )


class BaseMagneticDipole(BaseDipole):

    moment = tr.Float(
        help="moment of the dipole (Am^2)",
        default_value=1.0,
        min=0.0
    )
