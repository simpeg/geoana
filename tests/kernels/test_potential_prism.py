import numpy as np
import numpy.testing as npt
import pytest

import geoana.kernels.potential_field_prism as pf
import geoana.gravity as grav
from geoana.em.static import MagneticPrism
try:
    from numba import njit
except ImportError:
    njit = None


class TestCompiledVsNumpy():
    xyz = np.mgrid[-50:50:51j, -50:50:51j, -50:50:51j]

    def test_assert_using_compiled(self):
        assert pf._prism_f is not pf.prism_f

    def test_f(self):
        x, y, z = self.xyz
        v0 = pf._prism_f(x, y, z)
        v1 = pf.prism_f(x, y, z)
        npt.assert_allclose(v0, v1)

    def test_fz(self):
        x, y, z = self.xyz
        v0 = pf._prism_fz(x, y, z)
        v1 = pf.prism_fz(x, y, z)
        npt.assert_allclose(v0, v1)

    def test_fzz(self):
        x, y, z = self.xyz
        v0 = pf._prism_fzz(x, y, z)
        v1 = pf.prism_fzz(x, y, z)
        npt.assert_allclose(v0, v1)

    def test_fzx(self):
        x, y, z = self.xyz
        v0 = pf._prism_fzx(x, y, z)
        v1 = pf.prism_fzx(x, y, z)
        npt.assert_allclose(v0, v1)

    def test_fzy(self):
        x, y, z = self.xyz
        v0 = pf._prism_fzy(x, y, z)
        v1 = pf.prism_fzy(x, y, z)
        npt.assert_allclose(v0, v1)

    def test_fzzz(self):
        x, y, z = self.xyz
        v0 = pf._prism_fzzz(x, y, z)
        v1 = pf.prism_fzzz(x, y, z)
        npt.assert_allclose(v0, v1)

    def test_fxxy(self):
        x, y, z = self.xyz
        v0 = pf._prism_fxxy(x, y, z)
        v1 = pf.prism_fxxy(x, y, z)
        npt.assert_allclose(v0, v1)

    def test_fxxz(self):
        x, y, z = self.xyz
        v0 = pf._prism_fxxz(x, y, z)
        v1 = pf.prism_fxxz(x, y, z)
        npt.assert_allclose(v0, v1)

    def test_fxyz(self):
        x, y, z = self.xyz
        v0 = pf._prism_fxyz(x, y, z)
        v1 = pf.prism_fxyz(x, y, z)
        npt.assert_allclose(v0, v1)


PRISM_FUNCTIONS = [
    'prism_f',
    'prism_fz',
    'prism_fzx',
    'prism_fzy',
    'prism_fzz',
    'prism_fzzz',
    'prism_fxxy',
    'prism_fxxz',
    'prism_fxyz'
]

@pytest.mark.parametrize('method', PRISM_FUNCTIONS)
def test_prism_correct(method, sympy_potential_prism):
    x, y, z = np.mgrid[-10:10:6j, -10:10:4j, -10:10:8j]
    dx, dy, dz = 0.1, 0.2, 0.3
    test_func = getattr(pf, method)
    verify_func = sympy_potential_prism[method]
    verify = 0
    test = 0
    # Prism kernels are accurate for evaluating
    # definite integrals so... mimic one here
    for ix in [0, 1]:
        xp = x + ix * dx
        sign_ix = 2 * ix - 1
        for iy in [0, 1]:
            yp = y + iy * dy
            sign_iy = 2 * iy - 1
            for iz in [0, 1]:
                zp = z + iz * dz
                sign_iz = 2 * iz - 1
                sign = sign_ix * sign_iy * sign_iz
                test = test + sign * test_func(xp, yp, zp)
                verify = verify + sign * verify_func(xp, yp, zp)
    npt.assert_allclose(test, verify, rtol=1E-6)


def test_mag_init_and_errors():
    prism = MagneticPrism([-1, -1, -1], [1, 1, 1])
    np.testing.assert_equal(prism.magnetization, [0.0, 0.0, 1.0])

    np.testing.assert_equal(prism.moment, [0.0, 0.0, 8.0])

    with pytest.raises(TypeError):
        prism.magnetization = 'abc'

    with pytest.raises(ValueError):
        prism.magnetization = 1.0


def test_grav_init_and_errors():
    prism = grav.Prism([-1, -1, -1], [1, 1, 1])

    assert prism.mass == 8.0

    with pytest.raises(TypeError):
        prism.rho = 'abc'


@pytest.mark.skipif(njit is None, reason="numba is not installed.")
@pytest.mark.parametrize('function', [
    pf.prism_f,
    pf.prism_fz,
    pf.prism_fzx,
    pf.prism_fzy,
    pf.prism_fzz,
    pf.prism_fzzz,
    pf.prism_fxxy,
    pf.prism_fxxz,
    pf.prism_fxyz
])
def test_numba_jitting_nopython(function):

    # create a vectorized jit function of it:

    x = np.random.rand(10)
    y = np.random.rand(10)
    z = np.random.rand(10)
    @njit
    def jitted_func(x, y, z):
        n = len(x)
        out = np.empty_like(x)
        for i in range(n):
            out[i] = function(x[i], y[i], z[i])
        return out

    v1 = jitted_func(x, y, z)
    v2 = function(x, y, z)

    npt.assert_allclose(v1, v2)


@pytest.mark.parametrize("compiled", [True, False])
def test_log_kernels_near_negative_axis(compiled):
    # Near the negative y axis y + r cancels; log(y + r) = log(d) + asinh(y / d)
    # with d the distance to the axis. On the axis, the finite part is kept.
    fzx = pf.prism_fzx if compiled else pf._prism_fzx
    fz = pf.prism_fz if compiled else pf._prism_fz
    d = np.array([1e-150, 1e-12, 1e-8, 1e-4])
    y = np.full_like(d, -2.0)
    zero = np.zeros_like(d)
    log_y_r = np.log(d) + np.arcsinh(y / d)
    npt.assert_allclose(fzx(d, y, zero), -log_y_r, rtol=1e-14)
    # prism_fz = x log(y + r) + y log(x + r) at z = 0.
    npt.assert_allclose(fz(d, y, zero), d * log_y_r + y * np.log(d + np.sqrt(d * d + 4)), rtol=1e-14)
    npt.assert_allclose(fzx(zero, y, zero), np.log(4.0))
