import numpy as np
import pytest
import numpy.testing as npt
from scipy.constants import point, gravitational_constant

from geoana.em.static import MagneticDipoleWholeSpace, MagneticPoleWholeSpace
from geoana.em.static.wholespace import LineCurrentWholeSpace, PointCurrentWholeSpace

METHODS = [
    'vector_potential',
    'magnetic_field',
    'magnetic_flux_density',
]

ELEC_METHODS = [
    "scalar_potential",
    "electric_field",
    "current_density",
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
def h_pole(em_dipole_params):
    moment = float(em_dipole_params['moment'])
    mu = float(em_dipole_params['mu'])

    dip = MagneticPoleWholeSpace(
        mu=mu,
        moment=moment,
    )
    return dip

@pytest.fixture()
def line_current(em_dipole_params):
    length = float(em_dipole_params['length'])
    current = float(em_dipole_params['current'])
    mu = float(em_dipole_params['mu'])
    sigma = float(em_dipole_params['sigma'])

    nodes = np.asarray([
        [0, 0, 0,],
        [length, 0, 0],
    ])

    line = LineCurrentWholeSpace(nodes, current, mu=mu, sigma=sigma)
    return line


@pytest.fixture()
def point_current(em_dipole_params):
    current = float(em_dipole_params['current'])
    sigma = float(em_dipole_params['sigma'])

    location = [0, 0, 0]

    point = PointCurrentWholeSpace(1/sigma, current, location)
    return point


@pytest.fixture()
def xyz():
    nx, ny, nz = (2, 3, 4)
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-5, 5, ny)
    z = np.linspace(-5, 5, nz)
    xyz = np.meshgrid(x, y, z)
    return xyz


@pytest.mark.parametrize('method', METHODS)
def test_h_dip_broadcasting(h_dipole, xyz, method):
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
@pytest.mark.parametrize('orient', ['x', 'y', 'z'])
def test_h_dip_correct(method, orient, h_dipole, xyz, sympy_static_hx_dipole):
    x, y, z = xyz
    h_dipole.orientation = orient
    out = getattr(h_dipole, method)(xyz)

    sympy_func = sympy_static_hx_dipole[method]

    # cycle the input and output if not x
    if orient == 'x':
        verify = sympy_func(x, y, z)
    if orient == 'y':
        verify = sympy_func(y, z, x)
        verify = verify[..., [2, 0, 1]]
    elif orient == 'z':
        verify = sympy_func(z, x, y)
        verify = verify[..., [1, 2, 0]]
    npt.assert_allclose(verify, out, atol=1E-20)

@pytest.mark.parametrize('method', METHODS[1:])
def test_h_pole_broadcasting(h_pole, xyz, method):
    func = getattr(h_pole, method)

    out = func(xyz)
    assert out.shape == (*xyz[0].shape, 3)

    xyz = np.stack(xyz, axis=-1)
    out = func(xyz)
    assert out.shape == (*xyz.shape[:-1], 3)

    xyz = xyz.reshape((-1, 3))
    out = func(xyz)
    assert out.shape == (xyz.shape[0], 3)

    xyz = xyz.reshape((-1, 3))
    out = func(xyz)
    assert out.shape == (xyz.shape[0], 3)

    out = func(xyz[0])
    assert out.shape == (3,)

@pytest.mark.parametrize('method', METHODS[1:])
def test_h_pole_correct(method, h_pole, xyz, sympy_static_h_pole):
    x, y, z = xyz
    out = getattr(h_pole, method)(xyz)

    sympy_func = sympy_static_h_pole[method]

    verify = sympy_func(x, y, z)
    npt.assert_allclose(out, verify)

@pytest.mark.parametrize('method', METHODS + ELEC_METHODS)
def test_line_broadcasting(line_current, xyz, method):
    func = getattr(line_current, method)

    if method == 'scalar_potential':
        out_shape = tuple()
    else:
        out_shape = (3, )

    out = func(xyz)
    assert out.shape == (*xyz[0].shape, *out_shape)

    xyz = np.stack(xyz, axis=-1)
    out = func(xyz)
    assert out.shape == (*xyz.shape[:-1], *out_shape)

    xyz = xyz.reshape((-1, 3))
    out = func(xyz)
    assert out.shape == (xyz.shape[0], *out_shape)

    xyz = xyz.reshape((-1, 3))
    out = func(xyz)
    assert out.shape == (xyz.shape[0], *out_shape)

    out = func(xyz[0])
    assert out.shape == out_shape

@pytest.mark.parametrize('method', METHODS + ELEC_METHODS)
@pytest.mark.parametrize('orient', ['x', 'y', 'z'])
def test_line_correct(method, orient, line_current, xyz, sympy_linex_segment):
    x, y, z = xyz
    # they all start out as oriented in the x-direction
    if orient == 'y':
        l = line_current.nodes[1, 0]
        line_current.nodes[1] = [0, l, 0]
    elif orient == 'z':
        l = line_current.nodes[1, 0]
        line_current.nodes[1] = [0, 0, l]

    out = getattr(line_current, method)(xyz)

    sympy_func = sympy_linex_segment[method]

    # cycle the input and output if not x
    if orient == 'x':
        verify = sympy_func(x, y, z)
    if orient == 'y':
        verify = sympy_func(y, z, x)
        if method != 'scalar_potential':
            verify = verify[..., [2, 0, 1]]
    elif orient == 'z':
        verify = sympy_func(z, x, y)
        if method != 'scalar_potential':
            verify = verify[..., [1, 2, 0]]

    atol = 1E-18
    if method != 'magnetic_field':
        atol *= 1E-6  # to account for mu

    npt.assert_allclose(out, verify, atol=atol)


@pytest.mark.parametrize('method', ['potential', 'electric_field', 'current_density'])
def test_point_current_broadcast(point_current, xyz, method):
    func = getattr(point_current, method)

    if method == 'potential':
        out_shape = tuple()
    else:
        out_shape = (3,)

    out = func(xyz)
    assert out.shape == (*xyz[0].shape, *out_shape)

    xyz = np.stack(xyz, axis=-1)
    out = func(xyz)
    assert out.shape == (*xyz.shape[:-1], *out_shape)

    xyz = xyz.reshape((-1, 3))
    out = func(xyz)
    assert out.shape == (xyz.shape[0], *out_shape)

    xyz = xyz.reshape((-1, 3))
    out = func(xyz)
    assert out.shape == (xyz.shape[0], *out_shape)

    out = func(xyz[0])
    assert out.shape == out_shape


@pytest.mark.parametrize('method', ['potential', 'electric_field', 'current_density'])
def test_point_current_correct(method, xyz, point_current, sympy_grav_point):

    # scale from gravity to electric...
    scale = -point_current.current/(4 * np.pi * point_current.sigma * gravitational_constant * 10**5)
    if method == 'potential':
        scale *= -1
    elif method == 'current_density':
        scale *= point_current.sigma

    x, y, z = xyz

    grav_method = {
        'potential' : 'gravitational_potential',
        'electric_field' : 'gravitational_field',
        'current_density' : 'gravitational_field',
    }[method]

    out = getattr(point_current, method)(xyz)

    verify = scale * sympy_grav_point[grav_method](x, y, z)

    npt.assert_allclose(out, verify)

