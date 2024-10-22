import pytest
import numpy as np
import numpy.testing as npt

from geoana.em.fdem import ElectricDipoleWholeSpace

METHODS = [
    'vector_potential',
    'magnetic_field',
    'electric_field',
    'current_density',
    'magnetic_flux_density',
]

@pytest.fixture()
def e_dipole(em_dipole_params):
    nf = 5

    current = float(em_dipole_params['current'])
    length = float(em_dipole_params['length'])
    mu = float(em_dipole_params['mu'])
    conductivity = float(em_dipole_params['sigma'])
    eps = float(em_dipole_params['epsilon'])

    frequencies = np.logspace(2, 5, nf)
    dip = ElectricDipoleWholeSpace(
        frequency=frequencies,
        sigma=conductivity,
        mu=mu,
        current=current,
        orientation='x',
        length=length,
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
@pytest.mark.parametrize('nf', [0, 1, 5])
def test_broadcasting(e_dipole, xyz, method, nf):
    if nf == 0:
        e_dipole.frequency = e_dipole.frequency[0]
    else:
        e_dipole.frequency = e_dipole.frequency[:nf]
    dipole_func = getattr(e_dipole, method)
    freq_shape = e_dipole.frequency.shape

    out = dipole_func(xyz)
    assert out.shape == (*freq_shape, *xyz[0].shape, 3)

    xyz = np.stack(xyz, axis=-1)
    out = dipole_func(xyz)
    assert out.shape == (*freq_shape, *xyz.shape[:-1], 3)

    xyz = xyz.reshape((-1, 3))
    out = dipole_func(xyz)
    assert out.shape == (*freq_shape, xyz.shape[0], 3)

    xyz = xyz.reshape((-1, 3))
    out = dipole_func(xyz)
    assert out.shape == (*freq_shape, xyz.shape[0], 3)

    out = dipole_func(xyz[0])
    assert out.shape == (*freq_shape, 3)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('orient', ['x', 'y', 'z'])
def test_correct(method, orient, e_dipole, xyz, sympy_fdem_ex_dipole):
    x, y, z = xyz
    e_dipole.orientation = orient
    out = getattr(e_dipole, method)(xyz)

    sympy_func = sympy_fdem_ex_dipole[method]

    # cycle the input and output if not x
    if orient == 'x':
        verify = sympy_func(e_dipole.frequency[:, None, None, None], x, y, z)
    if orient == 'y':
        verify = sympy_func(e_dipole.frequency[:, None, None, None], y, z, x)
        verify = verify[..., [2, 0, 1]]
    elif orient == 'z':
        verify = sympy_func(e_dipole.frequency[:, None, None, None], z, x, y)
        verify = verify[..., [1, 2, 0]]

    npt.assert_allclose(verify, out)
