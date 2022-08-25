import numpy as np
from numpy.testing import assert_allclose
from discretize.tests import check_derivative
import pytest

import geoana.kernels.potential_field_prism as pf
import geoana.gravity as grav
from geoana.em.static import MagneticPrism, MagneticDipoleWholeSpace


class TestCompiledVsNumpy():
    xyz = np.mgrid[-50:50:51j, -50:50:51j, -50:50:51j]

    def test_assert_using_compiled(self):
        assert pf._prism_f is not pf.prism_f

    def test_f(self):
        x, y, z = self.xyz
        v0 = pf._prism_f(x, y, z)
        v1 = pf.prism_f(x, y, z)
        assert_allclose(v0, v1)

    def test_fz(self):
        x, y, z = self.xyz
        v0 = pf._prism_fz(x, y, z)
        v1 = pf.prism_fz(x, y, z)
        assert_allclose(v0, v1)

    def test_fzz(self):
        x, y, z = self.xyz
        v0 = pf._prism_fzz(x, y, z)
        v1 = pf.prism_fzz(x, y, z)
        assert_allclose(v0, v1)

    def test_fzx(self):
        x, y, z = self.xyz
        v0 = pf._prism_fzx(x, y, z)
        v1 = pf.prism_fzx(x, y, z)
        assert_allclose(v0, v1)

    def test_fzy(self):
        x, y, z = self.xyz
        v0 = pf._prism_fzy(x, y, z)
        v1 = pf.prism_fzy(x, y, z)
        assert_allclose(v0, v1)

    def test_fzzz(self):
        x, y, z = self.xyz
        v0 = pf._prism_fzzz(x, y, z)
        v1 = pf.prism_fzzz(x, y, z)
        assert_allclose(v0, v1)

    def test_fxxy(self):
        x, y, z = self.xyz
        v0 = pf._prism_fxxy(x, y, z)
        v1 = pf.prism_fxxy(x, y, z)
        assert_allclose(v0, v1)

    def test_fxxz(self):
        x, y, z = self.xyz
        v0 = pf._prism_fxxz(x, y, z)
        v1 = pf.prism_fxxz(x, y, z)
        assert_allclose(v0, v1)

    def test_fxyz(self):
        x, y, z = self.xyz
        v0 = pf._prism_fxyz(x, y, z)
        v1 = pf.prism_fxyz(x, y, z)
        assert_allclose(v0, v1)


class TestGravityPrismDerivatives():
    xyz = np.mgrid[-100:100:26j, -100:100:26j, -100:100:26j]
    x = xyz[0].ravel()
    y = xyz[1].ravel()
    z = xyz[2].ravel()
    prism = grav.Prism([-10, -10, -10], [10, 10, 10], rho=2.5)

    def test_pot_and_field_x(self):
        def grav_pot(x):
            pot = self.prism.gravitational_potential((x, self.y, self.z))

            Gv = self.prism.gravitational_field((x, self.y, self.z))
            def J(v):
                return Gv[..., 0] * v
            return pot, J

        assert check_derivative(grav_pot, self.x, num=3, plotIt=False)

    def test_pot_and_field_y(self):
        def grav_pot(y):
            pot = self.prism.gravitational_potential((self.x, y, self.z))

            Gv = self.prism.gravitational_field((self.x, y, self.z))
            def J(v):
                return Gv[..., 1] * v
            return pot, J

        assert check_derivative(grav_pot, self.y, num=3, plotIt=False)

    def test_pot_and_field_z(self):
        def grav_pot(z):
            pot = self.prism.gravitational_potential((self.x, self.y, z))

            Gv = self.prism.gravitational_field((self.x, self.y, z))
            def J(v):
                return Gv[..., 2] * v
            return pot, J

        assert check_derivative(grav_pot, self.z, num=3, plotIt=False)

    def test_field_and_grad_x(self):
        def grav_field(x):
            Gv = self.prism.gravitational_field((x, self.y, self.z)).ravel()
            GG_x = self.prism.gravitational_gradient((x, self.y, self.z))[..., 0, :]
            def J(v):
                return (GG_x * v[:, None]).ravel()
            return Gv, J

        assert check_derivative(grav_field, self.x, num=3, plotIt=False)

    def test_field_and_grad_y(self):
        def grav_field(y):
            Gv = self.prism.gravitational_field((self.x, y, self.z)).ravel()
            GG_y = self.prism.gravitational_gradient((self.x, y, self.z))[..., 1, :]
            def J(v):
                return (GG_y * v[:, None]).ravel()
            return Gv, J

        assert check_derivative(grav_field, self.y, num=3, plotIt=False)

    def test_field_and_grad_z(self):
        def grav_field(z):
            Gv = self.prism.gravitational_field((self.x, self.y, z)).ravel()
            GG_z = self.prism.gravitational_gradient((self.x, self.y, z))[..., 2, :]
            def J(v):
                return (GG_z * v[:, None]).ravel()
            return Gv, J

        assert check_derivative(grav_field, self.z, num=3, plotIt=False)


class TestMagPrismDerivatives():
    xyz = np.mgrid[-100:100:26j, -100:100:26j, -100:100:26j]
    x = xyz[0].ravel()
    y = xyz[1].ravel()
    z = xyz[2].ravel()
    prism = MagneticPrism([-10, -10, -10], [10, 10, 10], magnetization=[-1, 2, 3])

    def test_pot_and_field_x(self):
        def grav_pot(x):
            pot = self.prism.scalar_potential((x, self.y, self.z))

            Gv = self.prism.magnetic_field((x, self.y, self.z))
            def J(v):
                return Gv[..., 0] * v
            return pot, J

        assert check_derivative(grav_pot, self.x, num=3, plotIt=False)

    def test_pot_and_field_y(self):
        def grav_pot(y):
            pot = self.prism.scalar_potential((self.x, y, self.z))

            Gv = self.prism.magnetic_field((self.x, y, self.z))
            def J(v):
                return Gv[..., 1] * v
            return pot, J

        assert check_derivative(grav_pot, self.y, num=3, plotIt=False)

    def test_pot_and_field_z(self):
        def grav_pot(z):
            pot = self.prism.scalar_potential((self.x, self.y, z))

            Gv = self.prism.magnetic_field((self.x, self.y, z))
            def J(v):
                return Gv[..., 2] * v
            return pot, J

        assert check_derivative(grav_pot, self.z, num=3, plotIt=False)

    def test_field_and_grad_x(self):
        def grav_field(x):
            Gv = self.prism.magnetic_field((x, self.y, self.z)).ravel()
            GG_x = self.prism.magnetic_field_gradient((x, self.y, self.z))[..., 0, :]
            def J(v):
                return (GG_x * v[:, None]).ravel()
            return Gv, J

        assert check_derivative(grav_field, self.x, num=3, plotIt=False)

    def test_field_and_grad_y(self):
        def grav_field(y):
            Gv = self.prism.magnetic_field((self.x, y, self.z)).ravel()
            GG_y = self.prism.magnetic_field_gradient((self.x, y, self.z))[..., 1, :]
            def J(v):
                return (GG_y * v[:, None]).ravel()
            return Gv, J

        assert check_derivative(grav_field, self.y, num=3, plotIt=False)

    def test_field_and_grad_z(self):
        def grav_field(z):
            Gv = self.prism.magnetic_field((self.x, self.y, z)).ravel()
            GG_z = self.prism.magnetic_field_gradient((self.x, self.y, z))[..., 2, :]
            def J(v):
                return (GG_z * v[:, None]).ravel()
            return Gv, J

        assert check_derivative(grav_field, self.z, num=3, plotIt=False)


class TestGravityAccuracy():
    x, y, z = np.mgrid[-100:100:101j, -100:100:101j, -100:100:101j]
    xyz = np.stack((x, y, z), axis=-1)
    xyz = xyz[np.linalg.norm(xyz, axis=-1) >= 10]
    prism = grav.Prism([-1, -1, -1], [1, 1, 1], 2.0)
    point = grav.PointMass(mass=prism.mass, location=prism.location)

    def test_potential(self):
        pot_prism = self.prism.gravitational_potential(self.xyz)
        pot_point = self.point.gravitational_potential(self.xyz)
        np.testing.assert_allclose(pot_prism, pot_point, rtol=1E-4)

    def test_field(self):
        gv_prism = self.prism.gravitational_field(self.xyz)
        gv_point = self.point.gravitational_field(self.xyz)
        np.testing.assert_allclose(gv_prism, gv_point, rtol=1E-3)

    def test_gradient(self):
        gg_prism = self.prism.gravitational_field(self.xyz)
        gg_point = self.point.gravitational_field(self.xyz)
        np.testing.assert_allclose(
            gg_prism, gg_point, atol=(gg_prism.max() - gg_prism.min())*1E-3
        )


class TestMagneticAccuracy():
    x, y, z = np.mgrid[-100:100:101j, -100:100:101j, -100:100:101j]
    xyz = np.stack((x, y, z), axis=-1)
    xyz = xyz[np.linalg.norm(xyz, axis=-1) >= 10]
    prism = MagneticPrism([-1, -1, -1], [1, 1, 1], magnetization=[-1, 2, -0.5])
    m_mag = np.linalg.norm(prism.moment)
    m_unit = prism.moment/m_mag
    dipole = MagneticDipoleWholeSpace(
        location=prism.location, moment=m_mag, orientation=m_unit
    )

    def test_mag_field(self):
        H_prism = self.prism.magnetic_field(self.xyz)
        H_dipole = self.dipole.magnetic_field(self.xyz)
        np.testing.assert_allclose(
            H_prism, H_dipole, atol=(H_prism.max()-H_prism.min())*1E-3
        )

    def test_mag_flux(self):
        B_prism = self.prism.magnetic_flux_density(self.xyz)
        B_dipole = self.dipole.magnetic_flux_density(self.xyz)
        np.testing.assert_allclose(
            B_prism, B_dipole, atol=(B_prism.max()-B_dipole.min())*1E-3
        )


def test_mag_init_and_errors():
    prism = MagneticPrism([-1, -1, -1], [1, 1, 1])
    np.testing.assert_equal(prism.magnetization, [0.0, 0.0, 1.0])

    np.testing.assert_equal(prism.moment, [0.0, 0.0, 8.0])

    with pytest.raises(TypeError):
        prism.magnetization = 'abc'

    with pytest.raises(ValueError):
        prism.magnetization = 1.0


def test_grav_init_and_errors():
    prism = grav.Prism([-1, -1, -1], [1, 1, 1])

    assert prism.mass == 8.0

    with pytest.raises(TypeError):
        prism.rho = 'abc'
