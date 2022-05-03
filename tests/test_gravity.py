import pytest
import discretize
import numpy as np
from scipy.constants import G

from geoana import gravity

def U_from_PointMass(
    XYZ, loc, m
):

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)
    # Check

    dx = XYZ[:, 0]-loc[0]
    dy = XYZ[:, 1]-loc[1]
    dz = XYZ[:, 2]-loc[2]

    r = np.sqrt(dx**2. + dy**2. + dz**2.)

    u_g = (G * m) / r
    return u_g


class TestPointMass:

    def test_defaults(self):
        pm = gravity.PointMass()
        assert pm.mass == 1
        assert np.all(pm.location == np.r_[0., 0., 0.])

    def test_gravitational_potential(self):
        mass = 1
        pm = gravity.PointMass(
            mass=mass,
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])

        utest = U_from_PointMass(
            xyz, pm.location, pm.mass
        )

        u = pm.gravitational_potential(xyz)

        assert np.all(utest == u)


