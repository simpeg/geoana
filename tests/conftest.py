import pytest
import numpy as np

from sympy.vector import CoordSys3D, Del
import sympy

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