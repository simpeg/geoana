import pytest
import discretize
import numpy as np
from scipy.constants import G

from geoana import gravity


def U_from_PointMass(
    XYZ, loc, m
):

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

    dx = XYZ[:, 0]-loc[0]
    dy = XYZ[:, 1]-loc[1]
    dz = XYZ[:, 2]-loc[2]

    r = np.sqrt(dx**2. + dy**2. + dz**2.)

    u_g = (G * m) / r
    return u_g


def g_from_PointMass(
        XYZ, loc, m
):

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

    dx = XYZ[:, 0] - loc[0]
    dy = XYZ[:, 1] - loc[1]
    dz = XYZ[:, 2] - loc[2]

    r_vec = np.array([dx, dy, dz])
    r = np.sqrt(dx ** 2. + dy ** 2. + dz ** 2.)

    g_vec = (G * m * r_vec) / r
    return g_vec


def gtens_from_PointMass(
        XYZ, loc, m
):

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

    dx = XYZ[:, 0] - loc[0]
    dy = XYZ[:, 1] - loc[1]
    dz = XYZ[:, 2] - loc[2]

    r_vec = np.array([dx, dy, dz])
    r = np.sqrt(dx ** 2. + dy ** 2. + dz ** 2.)

    g_tens = (G * m * np.eye(3)) / r[..., None, None] ** 3 +\
             (3 * r_vec[..., None] * r_vec[..., None, :]) / r[..., None, None] ** 5
    return g_tens


class TestPointMass:

    def test_defaults(self):
        pm = gravity.PointMass()
        assert pm.mass == 1
        assert np.all(pm.location == np.r_[0., 0., 0.])

    def test_errors(self):
        pm = gravity.PointMass()
        with pytest.raises(TypeError):
            raise TypeError(f"mass must be a number, got {type(pm.mass)}")
        with pytest.raises(TypeError):
            raise TypeError(f"location must be array_like of float, got {type(pm.location)}")
        with pytest.raises(ValueError):
            raise ValueError(f"location must be array_like with shape (3,), got {pm.location.shape}")

    def test_gravitational_potential(self):
        mass = 1.0
        pm = gravity.PointMass(
            mass=mass
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])

        utest = U_from_PointMass(
            xyz, pm.location, pm.mass
        )

        u = pm.gravitational_potential(xyz)

        np.testing.assert_equal(utest, u)

    def test_gravitational_field(self):
        mass = 1.0
        pm = gravity.PointMass(
            mass=mass
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])

        gtest = g_from_PointMass(
            xyz, pm.location, pm.mass
        )

        g = pm.gravitational_field(xyz)

        np.testing.assert_equal(gtest, g)

    def test_gravitational_gradient(self):
        mass = 1.0
        pm = gravity.PointMass(
            mass=mass
        )
        x = np.linspace(-20., 20., 5)
        y = np.linspace(-30., 30., 5)
        z = np.linspace(-40., 40., 5)
        xyz = discretize.utils.ndgrid([x, y, z])

        gtenstest = gtens_from_PointMass(
            xyz, pm.location, pm.mass
        )

        gtens = pm.gravitational_gradient(xyz)

        np.testing.assert_equal(gtenstest, gtens)


