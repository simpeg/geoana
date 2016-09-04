from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .base import BaseEM, BaseMagneticDipole
from .. import traits as tr

import numpy as np
from ipywidgets import Latex
import warnings


class MagneticDipole_WholeSpace(BaseMagneticDipole, BaseEM):

    @tr.observe('sigma')
    def _sigma_changed(self, change):
        warnings.warn("Sigma is not involved in the calculation", UserWarning)

    def vector_potential(self, xyz, **kwargs):
        """Vector potential of a static magnetic dipole

            :param numpy.ndarray xyz: Location of the receivers(s)
            :rtype: numpy.ndarray
            :return: The magnetic vector potential at each observation location
        """

        n_obs = xyz.shape[0]

        offset = self.offset_from_location(xyz)
        dist = self.distance_from_location(xyz)
        m_vec = self.moment * np.atleast_2d(self.orientation).repeat(n_obs, axis=0)

        # Repeat the scalars
        dist = np.atleast_2d(dist).T.repeat(3, axis=1)

        m_cross_r = np.cross(m_vec, offset)
        A = (self.mu / (4 * np.pi)) * m_cross_r / (dist**3)

        return A

    def magnetic_flux(self, xyz, **kwargs):
        """Magnetic flux of a static magnetic dipole

            :param numpy.ndarray xyz: Location of the receivers(s)
            :rtype: numpy.ndarray
            :return: The magnetic vector potential at each observation location
        """

        n_obs = xyz.shape[0]

        offset = self.offset_from_location(xyz)
        dist = self.distance_from_location(xyz)
        m_vec = self.moment * np.atleast_2d(self.orientation).repeat(n_obs, axis=0)

        m_dot_r = (m_vec * offset).sum(axis=1)

        # Repeat the scalars
        m_dot_r = np.atleast_2d(m_dot_r).T.repeat(3, axis=1)
        dist = np.atleast_2d(dist).T.repeat(3, axis=1)

        b = (self.mu / (4 * np.pi)) * (
            (3.0 * offset * m_dot_r / (dist ** 5)) -
            m_vec / (dist ** 3)
        )
        return b

    @staticmethod
    def magnetic_flux_equation():
        return Latex("$\\frac{\mu}{4\pi} \\frac{\mathbf{m} \\times \mathbf{\hat{r}}}{r^2}$")


    def magnetic_field(self, xyz, **kwargs):
        return self.magnetic_flux(xyz, **kwargs) / self.mu
