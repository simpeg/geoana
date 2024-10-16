import pytest
import numpy as np

from geoana import gravity

METHODS = [
    "gravitational_potential",
    "gravitational_field",
    "gravitational_gradient",
]

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

    @pytest.mark.parametrize("method", METHODS)
    def test_correct(self, method, sympy_grav_point, grav_point_params):
        mass = grav_point_params["mass"]
        grav_obj = gravity.PointMass(
            mass=mass
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = np.meshgrid(x, y, z)

        func = getattr(grav_obj, method)

        out = func(xyz)

        verify = sympy_grav_point[method](*xyz)

        np.testing.assert_allclose(out, verify)


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
            s.radius = -1
        with pytest.raises(ValueError):
            s.location = [0, 1, 2, 3, 4]
        with pytest.raises(ValueError):
            s.location = [[0, 0, 1, 4], [0, 1, 0, 3]]
        with pytest.raises(TypeError):
            s.location = ["string"]

    @pytest.mark.parametrize("method", METHODS)
    def test_correct(self, method, sympy_grav_sphere, grav_point_params):
        mass = grav_point_params["mass"]
        radius = grav_point_params["radius"]
        vol = 4/3 * np.pi * radius**3
        rho = mass / vol

        grav_obj = gravity.Sphere(
            radius=radius,
            rho=rho,
        )
        x = np.linspace(-2., 2., 50)
        y = np.linspace(-3., 3., 50)
        z = np.linspace(-4., 4., 50)
        xyz = np.meshgrid(x, y, z)

        func = getattr(grav_obj, method)

        out = func(xyz)

        verify = sympy_grav_sphere[method](*xyz)

        np.testing.assert_allclose(out, verify)