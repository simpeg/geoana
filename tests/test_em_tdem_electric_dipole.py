import numpy as np
from dolfinx.la import vector
from scipy.constants import mu_0
from scipy.special import erf, erfc
import pytest
import numpy.testing as npt
from sympy.vector import CoordSys3D, Del
import sympy
from geoana.utils import vector_lambdify

from geoana.em.tdem import ElectricDipoleWholeSpace, theta

@pytest.fixture()
def e_dipole():
    nt = 5
    conductivity = 1e-3
    current = 2.0
    length = 0.5
    mu = 2 * mu_0
    ts = np.logspace(-6, -3, nt)
    dip = ElectricDipoleWholeSpace(
        time=ts, sigma=conductivity, mu=mu, current=current, orientation='x', length=length
    )
    return dip


@pytest.fixture()
def sympy_derive():
    R = CoordSys3D('R')
    delop = Del()

    I = sympy.sympify(2)
    ds = sympy.sympify(1)/2
    mu = 2 * (4 * sympy.pi * sympy.sympify(10)**-7)
    sigma = sympy.sympify(10)**-3
    t = sympy.symbols('t')

    r = sympy.sqrt(R.x**2 + R.y**2 + R.z**2)
    theta = sympy.sqrt(mu * sigma / (4 * t))

    A = I * ds / (4 * sympy.pi * r) * sympy.erf(theta * r) * R.i
    H = delop.cross(A).doit().simplify()
    dH_dt = sympy.diff(H, t).doit().simplify()
    E = (delop.cross(H).doit() / sigma).simplify()

    lamb_funcs = {
        'vector_potential' : vector_lambdify(A, R, t),
        'magnetic_field' : vector_lambdify(H, R, t),
        'electric_field' : vector_lambdify(E, R, t),
        'magnetic_field_time_deriv' : vector_lambdify(dH_dt, R, t),
    }
    return lamb_funcs

@pytest.fixture()
def xyz():
    nx, ny, nz = (2, 3, 4)
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-5, 5, ny)
    z = np.linspace(-5, 5, nz)
    xyz = np.meshgrid(x, y, z)
    return xyz

@pytest.mark.parametrize('method', [
    'vector_potential',
    'magnetic_field',
    'magnetic_field_time_deriv',
    'electric_field',
])
def test_broadcasting(e_dipole, xyz, method):
    e_dipole_func = getattr(e_dipole, method)
    nt = len(e_dipole.time)

    out = e_dipole_func(xyz)
    assert out.shape == (nt, *xyz[0].shape, 3)

    xyz = np.stack(xyz, axis=-1)
    out = e_dipole_func(xyz)
    assert out.shape == (nt, *xyz.shape[:-1], 3)

    xyz = xyz.reshape((-1, 3))
    out = e_dipole_func(xyz)
    assert out.shape == (nt, xyz.shape[0], 3)

    xyz = xyz.reshape((-1, 3))
    out = e_dipole_func(xyz)
    assert out.shape == (nt, xyz.shape[0], 3)

    out = e_dipole_func(xyz[0])
    assert out.shape == (nt, 3)


def test_h_field_orientation(e_dipole):
    # if e is in x direction:
    # mag field should curl CCW about the +x direction.
    # so directly above.. it should point towards -y
    out = e_dipole.magnetic_field([0, 0, 1])[0]
    assert out[0] == 0.0
    assert out[1] < 0.0
    assert out[2] == 0.0

    # below it should be +y
    out = e_dipole.magnetic_field([0, 0, -1])[0]
    assert out[0] == 0.0
    assert out[1] > 0.0
    assert out[2] == 0.0

    # In the +y direction, it should be up
    out = e_dipole.magnetic_field([0, 1, 0])[0]
    assert out[0] == 0.0
    assert out[1] == 0.0
    assert out[2] > 0.0

    # In the -y direction, it should be down
    out = e_dipole.magnetic_field([0, -1, 0])[0]
    assert out[0] == 0.0
    assert out[1] == 0.0
    assert out[2] < 0.0

@pytest.mark.parametrize(
    'method',
    [
        'vector_potential',
        'magnetic_field',
        'magnetic_field_time_deriv',
        'electric_field',
    ]
)
def test_correct(method, e_dipole, xyz, sympy_derive):
    x, y, z = xyz
    out = getattr(e_dipole, method)(xyz)

    sympy_func = sympy_derive[method]

    verify = sympy_func(
        e_dipole.time[:, None, None, None], x, y, z
    )

    # find smallest value that's not zero...
    npt.assert_allclose(verify, out, rtol=1E-5)