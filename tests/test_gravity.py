import pytest
import numpy as np
import discretize
from scipy.constants import G

from geoana import gravity


class TestPointMass:

    def test_defaults(self):
        pm = gravity.PointMass()
        assert(pm.mass == 1)
        assert(pm.location == 1)

    def test_gravitational_potential(self):
        mass = np.random.random_integers(1)
        location = np.random.random_integers(1)
        pm = gravity.PointMass(
            mass=mass,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])

        u = pm.gravitational_potential(xyz)

        r_vec = xyz - location
        r = np.linalg.norm(r_vec)
        u_g = (G * mass) / location

        self.assertEqual(u, u_g)
