import pytest
import numpy as np
import numpy.testing as npt

from geoana.em.fdem import MagneticDipoleWholeSpace

METHODS = [
    'vector_potential',
    'magnetic_field',
    'electric_field',
    'current_density',
    'magnetic_flux_density',
]


@pytest.fixture()
def h_dipole(em_dipole_params):
    nf = 5

    moment = float(em_dipole_params['moment'])
    mu = float(em_dipole_params['mu'])
    conductivity = float(em_dipole_params['sigma'])
    eps = float(em_dipole_params['epsilon'])

    frequencies = np.logspace(2, 5, nf)
    dip = MagneticDipoleWholeSpace(
        frequency=frequencies,
        sigma=conductivity,
        mu=mu,
        moment=moment,
        orientation='x',
        epsilon=eps,
        quasistatic=False
    )
    return dip


@pytest.fixture()
def xyz():
    nx, ny, nz = (2, 3, 4)
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-5, 5, ny)
    z = np.linspace(-5, 5, nz)
    xyz = np.meshgrid(x, y, z)
    return xyz


@pytest.mark.parametrize('method', METHODS)
def test_broadcasting(h_dipole, xyz, method):
    dipole_func = getattr(h_dipole, method)
    nf = len(h_dipole.frequency)

    out = dipole_func(xyz)
    assert out.shape == (nf, *xyz[0].shape, 3)

    xyz = np.stack(xyz, axis=-1)
    out = dipole_func(xyz)
    assert out.shape == (nf, *xyz.shape[:-1], 3)

    xyz = xyz.reshape((-1, 3))
    out = dipole_func(xyz)
    assert out.shape == (nf, xyz.shape[0], 3)

    xyz = xyz.reshape((-1, 3))
    out = dipole_func(xyz)
    assert out.shape == (nf, xyz.shape[0], 3)

    out = dipole_func(xyz[0])
    assert out.shape == (nf, 3)


@pytest.mark.parametrize('method', METHODS)
def test_correct(method, h_dipole, xyz, sympy_fdem_hx_dipole):
    x, y, z = xyz
    out = getattr(h_dipole, method)(xyz)

    sympy_func = sympy_fdem_hx_dipole[method]

    verify = sympy_func(
        h_dipole.frequency[:, None, None, None], x, y, z
    )
    npt.assert_allclose(verify, out)
