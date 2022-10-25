import pytest
import numpy as np
import discretize
from geoana.em import fdem
from scipy.constants import mu_0, epsilon_0


def test_defaults():
    frequencies = np.logspace(1, 4, 3)
    sigma = 1.0
    w = 2 * np.pi * frequencies
    hpw = fdem.HarmonicPlaneWave(frequency=frequencies, sigma=sigma)
    assert np.all(hpw.frequency == np.logspace(1, 4, 3))
    assert hpw.amplitude == 1.0
    assert np.all(hpw.orientation == np.r_[1., 0., 0.])
    assert np.all(hpw.omega == w)
    assert np.all(hpw.wavenumber == np.sqrt(w**2 * mu_0 * epsilon_0 - 1j * w * mu_0))
    assert hpw.sigma == 1.0
    assert hpw.mu == mu_0
    assert hpw.epsilon == epsilon_0


def test_errors():
    frequencies = np.logspace(1, 4, 3)
    hpw = fdem.HarmonicPlaneWave(frequency=frequencies)
    with pytest.raises(TypeError):
        hpw.frequency = "string"
    with pytest.raises(ValueError):
        hpw.frequency = -1
    with pytest.raises(TypeError):
        hpw.frequency = np.array([[1, 2], [3, 4]])
    with pytest.raises(TypeError):
        hpw.orientation = 1
    with pytest.raises(ValueError):
        hpw.orientation = np.r_[1., 0.]
    with pytest.raises(ValueError):
        hpw.orientation = np.r_[0., 0., 1.]


def test_electric_field():
    frequencies = np.logspace(1, 4, 3)
    hpw = fdem.HarmonicPlaneWave(frequency=frequencies)

    # test x orientation
    w = 2 * np.pi * frequencies
    k = np.sqrt(w**2 * mu_0 * epsilon_0 - 1j * w * mu_0)

    x = np.linspace(-20., 20., 50)
    y = np.linspace(-30., 30., 50)
    z = np.linspace(-40., 40., 50)
    xyz = discretize.utils.ndgrid([x, y, z])
    z = xyz[:, 2]

    kz = np.outer(k, z)
    ikz = 1j * kz

    ex = np.exp(ikz)
    ey = np.zeros_like(ex)
    ez = np.zeros_like(ex)

    e_vec = hpw.electric_field(xyz)

    np.testing.assert_equal(ex, e_vec[..., 0])
    np.testing.assert_equal(ey, e_vec[..., 1])
    np.testing.assert_equal(ez, e_vec[..., 2])

    # test y orientation
    hpw.orientation = 'Y'

    ey = np.exp(ikz)
    ex = np.zeros_like(ey)
    ez = np.zeros_like(ey)

    e_vec = hpw.electric_field(xyz)

    np.testing.assert_equal(ex, e_vec[..., 0])
    np.testing.assert_equal(ey, e_vec[..., 1])
    np.testing.assert_equal(ez, e_vec[..., 2])


def test_current_density():
    frequencies = np.logspace(1, 4, 3)
    sigma = 2.0
    hpw = fdem.HarmonicPlaneWave(frequency=frequencies, sigma=sigma)

    # test x orientation
    w = 2 * np.pi * frequencies
    k = np.sqrt(w**2 * mu_0 * epsilon_0 - 1j * w * mu_0 * 2)

    x = np.linspace(-20., 20., 50)
    y = np.linspace(-30., 30., 50)
    z = np.linspace(-40., 40., 50)
    xyz = discretize.utils.ndgrid([x, y, z])
    z = xyz[:, 2]

    kz = np.outer(k, z)
    ikz = 1j * kz

    jx = 2 * np.exp(ikz)
    jy = np.zeros_like(jx)
    jz = np.zeros_like(jx)

    j_vec = hpw.current_density(xyz)

    np.testing.assert_equal(jx, j_vec[..., 0])
    np.testing.assert_equal(jy, j_vec[..., 1])
    np.testing.assert_equal(jz, j_vec[..., 2])

    # test y orientation
    hpw.orientation = 'Y'

    jy = 2 * np.exp(ikz)
    jx = np.zeros_like(jy)
    jz = np.zeros_like(jy)

    j_vec = hpw.current_density(xyz)

    np.testing.assert_equal(jx, j_vec[..., 0])
    np.testing.assert_equal(jy, j_vec[..., 1])
    np.testing.assert_equal(jz, j_vec[..., 2])


def test_magnetic_field():
    hpw = fdem.HarmonicPlaneWave(frequency=1, sigma=1)

    # test x orientation
    w = 2 * np.pi
    k = np.sqrt(w ** 2 * mu_0 * epsilon_0 - 1j * w * mu_0)

    x = np.linspace(-20., 20., 50)
    y = np.linspace(-30., 30., 50)
    z = np.linspace(-40., 40., 50)
    xyz = discretize.utils.ndgrid([x, y, z])
    z = xyz[:, 2]

    kz = z * k
    ikz = 1j * kz
    Z = w * mu_0 / k

    hx = np.zeros_like(z)
    hy = 1 / Z * np.exp(ikz)
    hz = np.zeros_like(z)

    h_vec = hpw.magnetic_field(xyz)[0]

    np.testing.assert_equal(hx, h_vec[:, 0])
    np.testing.assert_allclose(hy, h_vec[:, 1], rtol=1E-15)
    np.testing.assert_equal(hz, h_vec[:, 2])

    # test y orientation
    hpw.orientation = 'Y'

    hx = - 1 / Z * np.exp(ikz)
    hy = np.zeros_like(z)
    hz = np.zeros_like(z)

    h_vec = hpw.magnetic_field(xyz)[0]

    np.testing.assert_allclose(hx, h_vec[:, 0], rtol=1E-15)
    np.testing.assert_equal(hy, h_vec[:, 1])
    np.testing.assert_equal(hz, h_vec[:, 2])


def test_magnetic_flux_density():
    hpw = fdem.HarmonicPlaneWave(frequency=1, sigma=1)

    # test x orientation
    w = 2 * np.pi
    k = np.sqrt(w ** 2 * mu_0 * epsilon_0 - 1j * w * mu_0)

    x = np.linspace(-20., 20., 50)
    y = np.linspace(-30., 30., 50)
    z = np.linspace(-40., 40., 50)
    xyz = discretize.utils.ndgrid([x, y, z])
    z = xyz[:, 2]

    kz = z * k
    ikz = 1j * kz
    Z = w / k

    bx = np.zeros_like(z)
    by = 1 / Z * np.exp(ikz)
    bz = np.zeros_like(z)

    b_vec = hpw.magnetic_flux_density(xyz)[0]

    np.testing.assert_equal(bx, b_vec[:, 0])
    np.testing.assert_allclose(by, b_vec[:, 1], rtol=1E-15)
    np.testing.assert_equal(bz, b_vec[:, 2])

    # test y orientation
    hpw.orientation = 'Y'

    bx = -1 / Z * np.exp(ikz)
    by = np.zeros_like(z)
    bz = np.zeros_like(z)

    b_vec = hpw.magnetic_flux_density(xyz)[0]

    np.testing.assert_allclose(bx, b_vec[:, 0], rtol=1E-15)
    np.testing.assert_equal(by, b_vec[:, 1])
    np.testing.assert_equal(bz, b_vec[:, 2])
