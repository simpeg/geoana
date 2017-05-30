from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import properties
from scipy.constants import mu_0, pi, epsilon_0


class BaseEM(properties.HasProperties):

    mu = properties.Float(
        'Magnetic permeability.',
        default=mu_0,
        min=0.0
    )

    sigma = properties.Float(
        'Electrical conductivity (S/m)',
        default=1.0,
        min=0.0
    )

    epsilon = properties.Float(
        'Permitivity value',
        default=epsilon_0,
        min=0.0
    )


class BaseDipole(BaseEM):

    orientation = properties.Vector3(
        'orientation of dipole',
        default='X',
        length=1.0
    )

    location = properties.Vector3(
        'location of the electric dipole source',
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


class BaseFDEM(BaseEM):

    frequency = properties.Float(
        'Source frequency (Hz)',
        default=1e2,
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

    length = properties.Float(
        'length of the dipole (m)',
        default=1.0,
        min=0.0
    )

    current = properties.Float(
        'size of the injected current (A)',
        default=1.0,
        min=0.0
    )


class BaseMagneticDipole(BaseDipole):

    moment = properties.Float(
        'moment of the dipole (Am^2)',
        default=1.0,
        min=0.0
    )
