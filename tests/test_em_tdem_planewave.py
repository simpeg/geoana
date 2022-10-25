import pytest
import numpy as np
import discretize
from geoana.em import tdem
from scipy.constants import mu_0, epsilon_0


def test_defaults():
    sigma = 1.0
    time = 1.0
    tpw = tdem.TransientPlaneWave(sigma=sigma, time=time)
    assert tpw.amplitude == 1.0
    assert np.all(tpw.orientation == np.r_[1., 0., 0.])
    assert tpw.sigma == 1.0
    assert tpw.time == 1.0
    assert tpw.mu == mu_0
    assert tpw.epsilon == epsilon_0


def test_errors():
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


def test_electric_field():
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
    ey = np.zeros_like(ex)
    ez = np.zeros_like(ex)

    e_vec = tpw.electric_field(xyz)[0]

    np.testing.assert_equal(ex, e_vec[:, 0])
    np.testing.assert_equal(ey, e_vec[:, 1])
    np.testing.assert_equal(ez, e_vec[:, 2])

    # test y orientation
    tpw.orientation = 'Y'

    ey = bunja / bunmo
    ex = np.zeros_like(ey)
    ez = np.zeros_like(ey)

    e_vec = tpw.electric_field(xyz)[0]

    np.testing.assert_equal(ex, e_vec[:, 0])
    np.testing.assert_equal(ey, e_vec[:, 1])
    np.testing.assert_equal(ez, e_vec[:, 2])


def test_current_density():
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
    jy = np.zeros_like(jx)
    jz = np.zeros_like(jx)

    j_vec = tpw.current_density(xyz)[0]

    np.testing.assert_equal(jx, j_vec[:, 0])
    np.testing.assert_equal(jy, j_vec[:, 1])
    np.testing.assert_equal(jz, j_vec[:, 2])

    # test y orientation
    tpw.orientation = 'Y'

    jy = bunja / bunmo
    jx = np.zeros_like(jy)
    jz = np.zeros_like(jy)

    j_vec = tpw.current_density(xyz)[0]

    np.testing.assert_equal(jx, j_vec[:, 0])
    np.testing.assert_equal(jy, j_vec[:, 1])
    np.testing.assert_equal(jz, j_vec[:, 2])


def test_magnetic_field():
    tpw = tdem.TransientPlaneWave(sigma=1.0, time=1.0)

    # test x orientation

    x = np.linspace(-20., 20., 50)
    y = np.linspace(-30., 30., 50)
    z = np.linspace(-40., 40., 50)
    xyz = discretize.utils.ndgrid([x, y, z])
    z = xyz[:, 2]

    hy = -(np.sqrt(1 / (np.pi * mu_0)) * np.exp(-(mu_0 * z ** 2) / 4))
    hx = np.zeros_like(hy)
    hz = np.zeros_like(hy)

    h_vec = tpw.magnetic_field(xyz)[0]

    np.testing.assert_equal(hx, h_vec[..., 0])
    np.testing.assert_allclose(hy, h_vec[..., 1], rtol=1E-15)
    np.testing.assert_equal(hz, h_vec[..., 2])

    # test y orientation
    tpw.orientation = 'Y'

    hx = (np.sqrt(1 / (np.pi * mu_0)) * np.exp(-(mu_0 * z ** 2) / 4))
    hy = np.zeros_like(hx)
    hz = np.zeros_like(hx)

    h_vec = tpw.magnetic_field(xyz)[0]

    np.testing.assert_allclose(hx, h_vec[..., 0], rtol=1E-15)
    np.testing.assert_equal(hy, h_vec[..., 1])
    np.testing.assert_equal(hz, h_vec[..., 2])


def test_magnetic_flux_density():
    tpw = tdem.TransientPlaneWave(sigma=1.0, time=1.0)

    # test x orientation

    x = np.linspace(-20., 20., 50)
    y = np.linspace(-30., 30., 50)
    z = np.linspace(-40., 40., 50)
    xyz = discretize.utils.ndgrid([x, y, z])
    z = xyz[:, 2]

    by = -mu_0 * (np.sqrt(1 / (np.pi * mu_0)) * np.exp(-(mu_0 * z ** 2) / 4))
    bx = np.zeros_like(by)
    bz = np.zeros_like(by)

    b_vec = tpw.magnetic_flux_density(xyz)[0]

    np.testing.assert_equal(bx, b_vec[..., 0])
    np.testing.assert_allclose(by, b_vec[..., 1], rtol=1E-15)
    np.testing.assert_equal(bz, b_vec[..., 2])

    # test y orientation
    tpw.orientation = 'Y'

    bx = mu_0 * (np.sqrt(1 / (np.pi * mu_0)) * np.exp(-(mu_0 * z ** 2) / 4))
    by = np.zeros_like(bx)
    bz = np.zeros_like(bx)

    b_vec = tpw.magnetic_flux_density(xyz)[0]

    np.testing.assert_allclose(bx, b_vec[..., 0], rtol=1E-15)
    np.testing.assert_equal(by, b_vec[..., 1])
    np.testing.assert_equal(bz, b_vec[..., 2])


def test_prop_direction():
    tpw = tdem.TransientPlaneWave(sigma=1.0, time=1.0)
    loc = [0, 0, -10]

    tpw.orientation = "x"

    e_vec = tpw.electric_field(loc).squeeze()
    b_vec = tpw.magnetic_flux_density(loc).squeeze()

    prop_dir = np.cross(e_vec, b_vec)
    prop_dir /= np.linalg.norm(prop_dir)
    np.testing.assert_allclose(prop_dir, [0, 0, -1])

    tpw.orientation = "y"

    e_vec = tpw.electric_field(loc).squeeze()
    b_vec = tpw.magnetic_flux_density(loc).squeeze()

    prop_dir = np.cross(e_vec, b_vec)
    prop_dir /= np.linalg.norm(prop_dir)
    np.testing.assert_allclose(prop_dir, [0, 0, -1])
