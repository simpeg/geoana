import pytest
import numpy as np

from sympy.vector import CoordSys3D, Del
import sympy
from scipy.constants import gravitational_constant as gamma

c = 299792458 # m/s
mu_0 = (4 * sympy.pi * sympy.sympify(10)**-7)
epsilon_0 = 1/(mu_0 * c**2)

def vector_lambdify(vec_func, coord_sys, *args):

    vec_func = vec_func.to_matrix(coord_sys)
    lambs = [sympy.lambdify((*args, *coord_sys.base_scalars()), f) for f in vec_func]

    def out(*inner_args):
        out_shape = np.broadcast(*inner_args).shape
        expansion = np.ones(out_shape)
        outs = [lamb(*inner_args) * expansion for lamb in lambs]
        return np.stack(outs, axis=-1)
    return out

def tensor_lambdify(row_vec_funcs, coord_sys, *args):
    lambs = []
    for vec_func in row_vec_funcs:
        vec_func = vec_func.to_matrix(coord_sys)
        lambs.append([sympy.lambdify((*args, *coord_sys.base_scalars()), f) for f in vec_func])

    def out(*inner_args):
        out_shape = np.broadcast(*inner_args).shape
        expansion = np.ones(out_shape)
        outs = []
        for row in lambs:
            out_row = [lamb(*inner_args) * expansion for lamb in row]
            outs.append(np.stack(out_row, axis=-1))
        return np.stack(outs, axis=-1)
    return out

@pytest.fixture(scope='session')
def em_dipole_params():
    param_dict = {
        "current" : sympy.sympify(2),
        "length" : sympy.sympify(1)/2,
        "mu" : 2 * mu_0,
        "sigma" : sympy.sympify(10)**-3,
        "epsilon" : 4 * epsilon_0,
        "moment" : sympy.sympify(1)/2
    }
    return param_dict


@pytest.fixture(scope='session')
def grav_point_params():
    param_dict = {
        "mass" : sympy.sympify(10)**5,
        "radius" : sympy.sympify(10),
    }
    return param_dict

def prism_params():
    param_dict = {
        'x0' : -4,
        'x1' : 4,
        'y0' : -3,
        'y1' : 3,
        'z0' : -2,
        'z1' : 2,
    }
    return param_dict

@pytest.fixture(scope='session')
def sympy_tdem_ex_dipole(em_dipole_params):
    R = CoordSys3D('R')
    delop = Del()

    I = em_dipole_params['current']
    ds = em_dipole_params['length']
    mu = em_dipole_params['mu']
    sigma = em_dipole_params['sigma']
    t = sympy.symbols('t')

    r = sympy.sqrt(R.x**2 + R.y**2 + R.z**2)
    theta = sympy.sqrt(mu * sigma / (4 * t))

    A = I * ds / (4 * sympy.pi * r) * sympy.erf(theta * r) * R.i
    H = delop.cross(A).doit().simplify()
    B = mu * H
    dH_dt = sympy.diff(H, t).doit().simplify()
    dB_dt = mu * dH_dt
    E = (delop.cross(H).doit() / sigma).simplify()
    J = sigma * E

    lamb_funcs = {
        'vector_potential' : vector_lambdify(A, R, t),
        'magnetic_field' : vector_lambdify(H, R, t),
        'magnetic_flux_density' : vector_lambdify(B, R, t),
        'electric_field' : vector_lambdify(E, R, t),
        'current_density' : vector_lambdify(J, R, t),
        'magnetic_field_time_deriv' : vector_lambdify(dH_dt, R, t),
        'magnetic_flux_density_time_deriv' : vector_lambdify(dB_dt, R, t),
    }
    return lamb_funcs


@pytest.fixture(scope='session')
def sympy_fdem_ex_dipole(em_dipole_params):
    R = CoordSys3D('R')
    delop = Del()

    I = em_dipole_params['current']
    ds = em_dipole_params['length']
    mu = em_dipole_params['mu']
    sigma = em_dipole_params['sigma']
    eps = em_dipole_params['epsilon']
    f = sympy.symbols(r'f')
    w = 2 * sympy.pi * f

    r = sympy.sqrt(R.x**2 + R.y**2 + R.z**2)
    k = sympy.sqrt(w**2 * mu * eps - sympy.I * w * mu * sigma)

    A = I * ds / (4 * sympy.pi * r) * sympy.exp(-sympy.I * k * r) * R.i
    H = delop.cross(A).doit().simplify()
    B = mu * H
    E = (delop.cross(H).doit() / sigma).simplify()
    J = sigma * E

    lamb_funcs = {
        'vector_potential' : vector_lambdify(A, R, f),
        'magnetic_field' : vector_lambdify(H, R, f),
        'magnetic_flux_density' : vector_lambdify(B, R, f),
        'electric_field' : vector_lambdify(E, R, f),
        'current_density' : vector_lambdify(J, R, f),
    }
    return lamb_funcs


@pytest.fixture(scope='session')
def sympy_fdem_hx_dipole(em_dipole_params):

    R = CoordSys3D('R')
    delop = Del()

    moment = em_dipole_params['moment']
    mu = em_dipole_params['mu']
    sigma = em_dipole_params['sigma']
    eps = em_dipole_params['epsilon']
    f = sympy.symbols(r'f')
    w = 2 * sympy.pi * f

    r = sympy.sqrt(R.x**2 + R.y**2 + R.z**2)
    k = sympy.sqrt(w**2 * mu * eps - sympy.I * w * mu * sigma)

    F = sympy.I * w * mu * moment / (4 * sympy.pi * r) * sympy.exp(-sympy.I * k * r) * R.i
    E = -delop.cross(F).doit().simplify()
    J = sigma * E
    B = delop.cross(E).doit().simplify()/(-sympy.I * w)
    H = B/mu

    lamb_funcs = {
        'vector_potential' : vector_lambdify(F, R, f),
        'magnetic_field' : vector_lambdify(H, R, f),
        'magnetic_flux_density' : vector_lambdify(B, R, f),
        'electric_field' : vector_lambdify(E, R, f),
        'current_density' : vector_lambdify(J, R, f),
    }
    return lamb_funcs


@pytest.fixture(scope='session')
def sympy_static_hx_dipole(em_dipole_params):
    R = CoordSys3D('R')
    delop = Del()

    moment = em_dipole_params['moment'] * R.i
    mu = em_dipole_params['mu']

    r_vec = R.x * R.i + R.y * R.j + R.z * R.k
    r = sympy.sqrt(R.x**2 + R.y**2 + R.z**2)

    A = (mu / (4 * sympy.pi * r**3) * moment.cross(r_vec)).doit()
    B = delop.cross(A).doit().simplify()
    H = B / mu

    lamb_funcs = {
        'vector_potential' : vector_lambdify(A, R),
        'magnetic_field' : vector_lambdify(H, R),
        'magnetic_flux_density' : vector_lambdify(B, R),
    }
    return lamb_funcs


@pytest.fixture(scope='session')
def sympy_grav_point(grav_point_params):
    R = CoordSys3D('R')
    delop = Del()
    mass = grav_point_params['mass']

    r = sympy.sqrt(R.x ** 2 + R.y ** 2 + R.z ** 2)
    grav_potential = gamma * mass / r

    grav_vector = delop.gradient(grav_potential).doit()

    grav_grads = [delop.gradient(grav_vector.components[comp]).doit() for comp in R.base_vectors()]

    lamb_funcs = {
        'gravitational_potential': sympy.lambdify((R.x, R.y, R.z), grav_potential),
        'gravitational_field': vector_lambdify(grav_vector, R),
        'gravitational_gradient': tensor_lambdify(grav_grads, R),
    }

    return lamb_funcs

@pytest.fixture(scope='session')
def sympy_grav_sphere(grav_point_params):
    R = CoordSys3D('R')
    delop = Del()
    mass = grav_point_params['mass']
    radius = grav_point_params['radius']
    volume = 4 * sympy.pi * radius**3 / 3
    density = mass/volume

    r = sympy.sqrt(R.x ** 2 + R.y ** 2 + R.z ** 2)
    grav_potential = sympy.Piecewise(
        (gamma * 2 * sympy.pi * density * (radius**2 - r**2/3), r < radius),
        (gamma * mass / r, True),
    )

    grav_vector = delop.gradient(grav_potential).doit()

    grav_grads = [delop.gradient(grav_vector.components[comp]).doit() for comp in R.base_vectors()]

    lamb_funcs = {
        'gravitational_potential': sympy.lambdify((R.x, R.y, R.z), grav_potential),
        'gravitational_field': vector_lambdify(grav_vector, R),
        'gravitational_gradient': tensor_lambdify(grav_grads, R),
    }

    return lamb_funcs

@pytest.fixture(scope='session')
def sympy_potential_prism():
    x, y, z = sympy.symbols('x y z')
    r = sympy.sqrt(x ** 2 + y ** 2 + z ** 2)
    f = (
            - x * y * sympy.log(z + r)
            - y * z * sympy.log(x + r)
            - z * x * sympy.log(y + r)
            + x ** 2 * sympy.atan(y * z / (x * r)) / 2
            + y ** 2 * sympy.atan(x * z / (y * r)) / 2
            + z ** 2 * sympy.atan(x * y / (z * r)) / 2
    )

    fz = -sympy.diff(f, z)

    fzx = sympy.diff(f, z, x)
    fzy = sympy.diff(f, z, y)
    fzz = sympy.diff(f, z, z)

    fzzz = -sympy.diff(f, z, z, z)
    fxxy = -sympy.diff(f, x, x, y)
    fxxz = -sympy.diff(f, x, x, z)
    fxyz = -sympy.diff(f, x, y, z)

    lamb_funcs = {
        'prism_f': sympy.lambdify((x, y, z), f),
        'prism_fz': sympy.lambdify((x, y, z), fz),
        'prism_fzx': sympy.lambdify((x, y, z), fzx),
        'prism_fzy': sympy.lambdify((x, y, z), fzy),
        'prism_fzz': sympy.lambdify((x, y, z), fzz),
        'prism_fzzz': sympy.lambdify((x, y, z), fzzz),
        'prism_fxxy': sympy.lambdify((x, y, z), fxxy),
        'prism_fxxz': sympy.lambdify((x, y, z), fxxz),
        'prism_fxyz': sympy.lambdify((x, y, z), fxyz),
    }

    return lamb_funcs
