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


def U_from_Sphere(
    XYZ, loc, m, rho, radius
):

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

    r_vec = XYZ - loc
    r = np.linalg.norm(r_vec, axis=-1)

    u_g = np.zeros_like(r)
    ind0 = r > radius
    u_g[ind0] = (G * m) / r[ind0]
    u_g[~ind0] = G * 2 / 3 * np.pi * rho * (3 * radius ** 2 - r[~ind0] ** 2)
    return u_g


def g_from_Sphere(
        XYZ, loc, m, rho, radius
):

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

    r_vec = XYZ - loc
    r = np.linalg.norm(r_vec, axis=-1)

    g_vec = np.zeros((*r.shape, 3))
    ind0 = r > radius
    g_vec[ind0] = -G * m * r_vec[ind0] / r[ind0, None] ** 3
    g_vec[~ind0] = -G * 4 / 3 * np.pi * rho * r_vec[~ind0]
    return g_vec


def gtens_from_Sphere(
        XYZ, loc, m, rho, radius
):

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

    r_vec = XYZ - loc
    r = np.linalg.norm(r_vec, axis=-1)

    g_tens = np.zeros((*r.shape, 3, 3))
    ind0 = r > radius
    g_tens[ind0] = -G * m * (np.eye(3) / r[ind0, None, None] ** 3 -
                             3 * r_vec[ind0, None] * r_vec[ind0, None, :] / r[ind0, None, None] ** 5)
    g_tens[~ind0] = -G * 4 / 3 * np.pi * rho * np.eye(3)
    return g_tens


class TestSphere:

    def test_defaults(self):
        radius = 1.0
        rho = 1.0
        s = gravity.Sphere(radius, rho)
        assert s.rho == 1
        assert s.radius == 1
        assert s.mass == 4 / 3 * np.pi * s.radius ** 3 * s.rho
        assert np.all(s.location == np.r_[0., 0., 0.])

    def test_errors(self):
        s = gravity.Sphere(rho=1.0, radius=1.0, location=None)
        with pytest.raises(TypeError):
            s.mass = "string"
        with pytest.raises(ValueError):
            s.rho = -1
        with pytest.raises(ValueError):
            s.radius = -1
        with pytest.raises(ValueError):
            s.location = [0, 1, 2, 3, 4]
        with pytest.raises(ValueError):
            s.location = [[0, 0, 1, 4], [0, 1, 0, 3]]
        with pytest.raises(TypeError):
            s.location = ["string"]

    def test_gravitational_potential(self):
        radius = 1.0
        rho = 1.0
        location = [0., 0., 0.]
        s = gravity.Sphere(
            radius=radius,
            rho=rho,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])

        utest = U_from_Sphere(
            xyz, s.location, s.mass, s.rho, s.radius
        )

        u = s.gravitational_potential(xyz)
        np.testing.assert_equal(utest, u)

    def test_gravitational_field(self):
        radius = 1.0
        rho = 1.0
        location = [0., 0., 0.]
        s = gravity.Sphere(
            radius=radius,
            rho=rho,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])

        gtest = g_from_Sphere(
            xyz, s.location, s.mass, s.rho, s.radius
        )

        g = s.gravitational_field(xyz)
        np.testing.assert_equal(gtest, g)

    def test_gravitational_gradient(self):
        radius = 1.0
        rho = 1.0
        location = [0., 0., 0.]
        s = gravity.Sphere(
            radius=radius,
            rho=rho,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])

        g_tens_test = gtens_from_Sphere(
            xyz, s.location, s.mass, s.rho, s.radius
        )

        g_tens = s.gravitational_gradient(xyz)
        np.testing.assert_equal(g_tens_test, g_tens)
