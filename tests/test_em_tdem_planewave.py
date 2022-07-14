import pytest
import numpy as np
import discretize
from geoana.em import tdem
from scipy.constants import mu_0, epsilon_0


class TestTransientPlaneWave:

    def test_defaults(self):
        sigma = 1.0
        time = 1.0
        tpw = tdem.TransientPlaneWave(sigma=sigma, time=time)
        assert tpw.amplitude == 1.0
        assert np.all(tpw.orientation == np.r_[1., 0., 0.])
        assert tpw.sigma == 1.0
        assert tpw.time == 1.0
        assert tpw.mu == mu_0
        assert tpw.epsilon == epsilon_0

    def test_errors(self):
        tpw = tdem.TransientPlaneWave(sigma=1.0, time=1.0)
        with pytest.raises(TypeError):
            tpw.time = "string"
        with pytest.raises(ValueError):
            tpw.time = -1
        with pytest.raises(TypeError):
            tpw.time = np.array([[1, 2], [3, 4]])
        with pytest.raises(TypeError):
            tpw.orientation = 1
        with pytest.raises(ValueError):
            tpw.orientation = np.r_[1., 0.]
        with pytest.raises(ValueError):
            tpw.orientation = np.r_[0., 0., 1.]

    def test_electric_field(self):
        tpw = tdem.TransientPlaneWave(sigma=1.0, time=1.0)

        # test x orientation
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])
        z = xyz[:, 2]

        bunja = -mu_0 ** 0.5 * z * np.exp(-(mu_0 * z ** 2) / 4)
        bunmo = 2 * np.pi ** 0.5

        ex = bunja / bunmo
        ey = np.zeros_like(z)
        ez = np.zeros_like(z)

        np.testing.assert_equal(ex, tpw.electric_field(xyz)[0])
        np.testing.assert_equal(ey, tpw.electric_field(xyz)[1])
        np.testing.assert_equal(ez, tpw.electric_field(xyz)[2])

        # test y orientation
        tpw.orientation = 'Y'

        ex = np.zeros_like(z)
        ey = bunja / bunmo
        ez = np.zeros_like(z)

        np.testing.assert_equal(ex, tpw.electric_field(xyz)[0])
        np.testing.assert_equal(ey, tpw.electric_field(xyz)[1])
        np.testing.assert_equal(ez, tpw.electric_field(xyz)[2])

    def test_current_density(self):
        tpw = tdem.TransientPlaneWave(sigma=1.0, time=1.0)

        # test x orientation
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])
        z = xyz[:, 2]

        bunja = -mu_0 ** 0.5 * z * np.exp(-(mu_0 * z ** 2) / 4)
        bunmo = 2 * np.pi ** 0.5

        jx = bunja / bunmo
        jy = np.zeros_like(z)
        jz = np.zeros_like(z)

        np.testing.assert_equal(jx, tpw.current_density(xyz)[0])
        np.testing.assert_equal(jy, tpw.current_density(xyz)[1])
        np.testing.assert_equal(jz, tpw.current_density(xyz)[2])

        # test y orientation
        tpw.orientation = 'Y'

        jx = np.zeros_like(z)
        jy = bunja / bunmo
        jz = np.zeros_like(z)

        np.testing.assert_equal(jx, tpw.current_density(xyz)[0])
        np.testing.assert_equal(jy, tpw.current_density(xyz)[1])
        np.testing.assert_equal(jz, tpw.current_density(xyz)[2])

    def test_magnetic_field(self):
        tpw = tdem.TransientPlaneWave(sigma=1.0, time=1.0)

        # test x orientation

        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])
        z = xyz[:, 2]

        hx = np.zeros_like(z)
        hy = (np.sqrt(1 / (np.pi * mu_0)) * np.exp(-(mu_0 * z ** 2) / 4))
        hz = np.zeros_like(z)

        np.testing.assert_equal(hx, tpw.magnetic_field(xyz)[0])
        np.testing.assert_equal(hy, tpw.magnetic_field(xyz)[1])
        np.testing.assert_equal(hz, tpw.magnetic_field(xyz)[2])

        # test y orientation
        tpw.orientation = 'Y'

        hx = (np.sqrt(1 / (np.pi * mu_0)) * np.exp(-(mu_0 * z ** 2) / 4))
        hy = np.zeros_like(z)
        hz = np.zeros_like(z)

        np.testing.assert_equal(hx, tpw.magnetic_field(xyz)[0])
        np.testing.assert_equal(hy, tpw.magnetic_field(xyz)[1])
        np.testing.assert_equal(hz, tpw.magnetic_field(xyz)[2])

    def test_magnetic_flux_density(self):
        tpw = tdem.TransientPlaneWave(sigma=1.0, time=1.0)

        # test x orientation

        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])
        z = xyz[:, 2]

        bx = np.zeros_like(z)
        by = mu_0 * (np.sqrt(1 / (np.pi * mu_0)) * np.exp(-(mu_0 * z ** 2) / 4))
        bz = np.zeros_like(z)

        np.testing.assert_equal(bx, tpw.magnetic_flux_density(xyz)[0])
        np.testing.assert_equal(by, tpw.magnetic_flux_density(xyz)[1])
        np.testing.assert_equal(bz, tpw.magnetic_flux_density(xyz)[2])

        # test y orientation
        tpw.orientation = 'Y'

        bx = mu_0 * (np.sqrt(1 / (np.pi * mu_0)) * np.exp(-(mu_0 * z ** 2) / 4))
        by = np.zeros_like(z)
        bz = np.zeros_like(z)

        np.testing.assert_equal(bx, tpw.magnetic_flux_density(xyz)[0])
        np.testing.assert_equal(by, tpw.magnetic_flux_density(xyz)[1])
        np.testing.assert_equal(bz, tpw.magnetic_flux_density(xyz)[2])
