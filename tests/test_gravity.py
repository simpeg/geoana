import pytest
import discretize
import numpy as np
from scipy.constants import G

from geoana import gravity


def U_from_PointMass(
    XYZ, loc, m
):

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

    r_vec = XYZ - loc
    r = np.linalg.norm(r_vec, axis=-1)

    u_g = (G * m) / r
    return u_g


def g_from_PointMass(
        XYZ, loc, m
):

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

    r_vec = XYZ - loc
    r = np.linalg.norm(r_vec, axis=-1)

    g_vec = -G * m * r_vec / r[..., None] ** 3
    return g_vec


def gtens_from_PointMass(
        XYZ, loc, m
):

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

    r_vec = XYZ - loc
    r = np.linalg.norm(r_vec, axis=-1)

    g_tens = -G * m * (np.eye(3) / r[..., None, None] ** 3 -
                       3 * r_vec[..., None] * r_vec[..., None, :] / r[..., None, None] ** 5)
    return g_tens


class TestPointMass:

    def test_defaults(self):
        pm = gravity.PointMass()
        assert pm.mass == 1
        assert np.all(pm.location == np.r_[0., 0., 0.])

    def test_errors(self):
        pm = gravity.PointMass(mass=1.0, location=None)
        with pytest.raises(TypeError):
            pm.mass = "string"
        with pytest.raises(ValueError):
            pm.location = [0, 1, 2, 3]
        with pytest.raises(ValueError):
            pm.location = [[0, 0, 1, 4], [0, 1, 0, 3]]
        with pytest.raises(TypeError):
            pm.location = ["string"]

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
        print(
            "\n\nTesting Gravitational Potential U\n"
        )

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
        print(
            "\n\nTesting Gravitational Field g\n"
        )

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

        g_tenstest = gtens_from_PointMass(
            xyz, pm.location, pm.mass
        )

        g_tens = pm.gravitational_gradient(xyz)
        print(
            "\n\nTesting Gravitational Gradient g_tens\n"
        )

        np.testing.assert_equal(g_tenstest, g_tens)


