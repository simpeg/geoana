import numpy as np
import pytest
import numpy.testing as npt

from geoana.em.static import MagneticDipoleWholeSpace

METHODS = [
    'vector_potential',
    'magnetic_field',
    'magnetic_flux_density',
]

@pytest.fixture()
def h_dipole(em_dipole_params):
    moment = float(em_dipole_params['moment'])
    mu = float(em_dipole_params['mu'])

    dip = MagneticDipoleWholeSpace(
        mu=mu,
        moment=moment,
        orientation='x',
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

    out = dipole_func(xyz)
    assert out.shape == (*xyz[0].shape, 3)

    xyz = np.stack(xyz, axis=-1)
    out = dipole_func(xyz)
    assert out.shape == (*xyz.shape[:-1], 3)

    xyz = xyz.reshape((-1, 3))
    out = dipole_func(xyz)
    assert out.shape == (xyz.shape[0], 3)

    xyz = xyz.reshape((-1, 3))
    out = dipole_func(xyz)
    assert out.shape == (xyz.shape[0], 3)

    out = dipole_func(xyz[0])
    assert out.shape == (3,)


@pytest.mark.parametrize('method', METHODS)
def test_correct(method, h_dipole, xyz, sympy_static_hx_dipole):
    x, y, z = xyz
    out = getattr(h_dipole, method)(xyz)

    sympy_func = sympy_static_hx_dipole[method]

    verify = sympy_func(x, y, z)
    npt.assert_allclose(verify, out, atol=1E-20)