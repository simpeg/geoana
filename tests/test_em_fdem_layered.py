import unittest
import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import mu_0, epsilon_0
from empymod.utils import check_hankel
from empymod.transform import get_dlf_points
from geoana.em.fdem import MagneticDipoleHalfSpace, MagneticDipoleLayeredHalfSpace
from geoana.kernels.tranverse_electric_reflections import (
    rTE_forward, rTE_gradient, _rTE_forward, _rTE_gradient
)
import discretize
from scipy.special import iv, kv
from geoana.em.fdem.base import sigma_hat

from discretize.Tests import checkDerivative as check_derivative


class TestHalfSpace:

    def test_magnetic_field(self):
        sigma = 1.0
        moment = 2.0
        frequencies = np.logspace(1, 4, 3)
        w = 2 * np.pi * frequencies
        sigh = sigma_hat(frequencies, sigma)
        k = np.sqrt(-1j * w * mu_0 * sigh)

        location = np.r_[0, 0, 0]
        orientation = np.r_[1, 0, 0]
        xyz = np.c_[5, 0, 0]
        dxy = xyz[..., :2] - location[:2]
        r = np.linalg.norm(dxy, axis=-1)
        x = dxy[..., 0]
        y = dxy[..., 1]

        for dim in range(r.ndim):
            k = k[:, None]

        em_x = em_y = em_z = 0
        src_x, src_y, src_z = orientation

        alpha = 1j * k * r/2
        ik1 = iv(1, alpha) * kv(1, alpha)
        ik2 = iv(2, alpha) * kv(2, alpha)

        dip_half = MagneticDipoleHalfSpace(
            frequencies,
            sigma=sigma,
            orientation=orientation,
            moment=moment,
        )
        phi = 2 / (k ** 2 * r ** 4) * (3 + k ** 2 * r ** 2 - (3 + 3j * k * r - k ** 2 * r ** 2) * np.exp(-1j * k * r))
        dphi_dr = 2 / (k ** 2 * r ** 5) * (-2 * k ** 2 * r ** 2 - 12 + (-1j * k ** 3 * r ** 3 - 5 * k ** 2 * r ** 2 + 12j * k * r + 12) * np.exp(-1j * k * r))

        em_x += src_x * (-1 / r ** 3) * (y ** 2 * phi + x ** 2 * r * dphi_dr)
        em_y += src_x * (1 / r ** 3) * x * y * (phi - r * dphi_dr)
        em_z -= src_x * (k ** 2 * x / r ** 2) * (ik1 - ik2)
        e = moment / (4 * np.pi) * np.stack((em_x, em_y, em_z), axis=-1).squeeze()
        e_test = dip_half.magnetic_field(xyz, field='total')

        np.testing.assert_equal(e_test, e)


class TestLayeredHalfspace(unittest.TestCase):

    def test_defaults(self):
        frequencies = np.logspace(1, 4, 3)
        sigma = 1
        mu = mu_0
        epsilon = epsilon_0
        mag_layer = MagneticDipoleLayeredHalfSpace(
            frequency=frequencies,
            thickness=None,
            sigma=sigma,
            mu=mu,
            epsilon=epsilon,
        )

        np.testing.assert_equal(np.array([]), mag_layer.thickness)
        sh = np.stack((sigma_hat(frequencies, sigma), sigma_hat(frequencies, sigma), sigma_hat(frequencies, sigma)), axis=0)
        np.testing.assert_equal(mag_layer.sigma_hat, sh)

    def test_errors(self):
        frequencies = np.logspace(1, 4, 3)
        sigma = 1
        mu = mu_0
        epsilon = epsilon_0
        thickness = 10 * np.ones(5)
        mag_layer = MagneticDipoleLayeredHalfSpace(
            frequency=frequencies,
            thickness=thickness,
            sigma=sigma,
            mu=mu,
            epsilon=epsilon
        )
        with pytest.raises(ValueError):
            mag_layer.location = np.r_[1, 1, -1]
            mag_layer._check_is_valid_location()

        with pytest.raises(TypeError):
            mag_layer.frequency = "string"
        with pytest.raises(ValueError):
            mag_layer.frequency = -1
        with pytest.raises(TypeError):
            mag_layer.frequency = np.array([[1, 2], [3, 4]])

        with pytest.raises(TypeError):
            mag_layer.thickness = "string"
        with pytest.raises(ValueError):
            mag_layer.thickness = -1
        with pytest.raises(TypeError):
            mag_layer.thickness = np.array([[1, 2], [3, 4]])

        with pytest.raises(TypeError):
            mag_layer.sigma = "string"
        with pytest.raises(TypeError):
            mag_layer.sigma = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            mag_layer.sigma = 1-1j
        with pytest.raises(ValueError):
            mag_layer.sigma = -1
        with pytest.raises(TypeError):
            mag_layer.sigma = np.array([1+1j, 1+2j, 1+3j])
        with pytest.raises(TypeError):
            mag_layer.sigma = np.array([[1+1j], [1+2j], [1+3j]])
        with pytest.raises(TypeError):
            mag_layer.sigma = np.array([[[1+1j], [1+2j]]])

        with pytest.raises(TypeError):
            mag_layer.mu = "string"
        with pytest.raises(TypeError):
            mag_layer.mu = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            mag_layer.mu = 1-1j
        with pytest.raises(ValueError):
            mag_layer.mu = -1
        with pytest.raises(TypeError):
            mag_layer.mu = np.array([1+1j, 1+2j, 1+3j])
        with pytest.raises(TypeError):
            mag_layer.mu = np.array([[1+1j], [1+2j], [1+3j]])
        with pytest.raises(TypeError):
            mag_layer.mu = np.array([[[1+1j], [1+2j]]])

        with pytest.raises(TypeError):
            mag_layer.epsilon = "string"
        with pytest.raises(TypeError):
            mag_layer.epsilon = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            mag_layer.epsilon = 1-1j
        with pytest.raises(ValueError):
            mag_layer.epsilon = -1
        with pytest.raises(TypeError):
            mag_layer.epsilon = np.array([1+1j, 1+2j, 1+3j])
        with pytest.raises(TypeError):
            mag_layer.epsilon = np.array([[1+1j], [1+2j], [1+3j]])
        with pytest.raises(TypeError):
            mag_layer.epsilon = np.array([[[1+1j], [1+2j]]])

        with pytest.raises(NotImplementedError):
            mag_layer.wavenumber()
        with pytest.raises(NotImplementedError):
            mag_layer.skin_depth()

    def test_magnetic_field_errors(self):
        frequencies = np.logspace(1, 4, 3)
        mu = mu_0
        sigma = np.r_[0.1, 1, 0.01]
        epsilon = epsilon_0
        thickness = np.r_[5, 2]
        orientation = np.r_[1, 0, 0]
        mag_layer = MagneticDipoleLayeredHalfSpace(
            frequency=frequencies,
            thickness=thickness,
            sigma=sigma,
            mu=mu,
            epsilon=epsilon,
            orientation=orientation
        )
        with pytest.raises(ValueError):
            x = np.linspace(-20., 20., 50)
            y = np.linspace(-30., 30., 50)
            z = np.linspace(-40., 40., 50)
            xyz = discretize.utils.ndgrid([x, y, z])
            mag_layer.magnetic_field(xyz)

    def test_magnetic_field(self):
        frequencies = np.logspace(1, 4, 3)
        sigma = 1
        thickness = 10 * np.ones(5)
        location = np.r_[0, 0, 0]
        orientation = np.r_[1, 0, 0]
        moment = 1
        mag_layer = MagneticDipoleLayeredHalfSpace(
            frequency=frequencies,
            thickness=thickness,
            location=location,
            sigma=sigma,
            orientation=orientation,
            moment=moment
        )

        xyz = np.c_[5, 0, 0]
        h = mag_layer.location[2]
        dxyz = xyz - mag_layer.location
        offsets = np.linalg.norm(dxyz[:, :-1], axis=-1)
        ht, htarg = check_hankel('dlf', {'dlf': 'key_101_2009', 'pts_per_dec': 0}, 1)
        fhtfilt = htarg['dlf']
        pts_per_dec = htarg['pts_per_dec']

        f = frequencies
        n_frequency = len(f)

        lambd, int_points = get_dlf_points(fhtfilt, offsets, pts_per_dec)

        thick = mag_layer.thickness
        n_layer = len(thick) + 1
        thick, sigma, epsilon, mu = mag_layer._get_valid_properties_array()
        sigh = sigma_hat(np.tile(mag_layer.frequency.reshape(
            (1, n_frequency)), (n_layer, 1)), sigma, epsilon, quasistatic=mag_layer.quasistatic)

        rTE = rTE_forward(f, lambd.reshape(-1), sigh, mu, thick)
        rTE = rTE.reshape((n_frequency, *lambd.shape))
        rTE *= np.exp(-lambd * (xyz[:, -1] + h)[:, None])

        src_x, src_y, src_z = mag_layer.orientation
        C0x = C0y = C0z = 0
        C1x = C1y = C1z = 0
        C0x += src_x * (dxyz[:, 0] ** 2 / offsets ** 2)[:, None] * lambd ** 2
        C1x += src_x * (1 / offsets - 2 * dxyz[:, 0] ** 2 / offsets ** 3)[:, None] * lambd
        C0y += src_x * (dxyz[:, 0] * dxyz[:, 1] / offsets ** 2)[:, None] * lambd ** 2
        C1y -= src_x * (2 * dxyz[:, 0] * dxyz[:, 1] / offsets ** 3)[:, None] * lambd
        C1z -= (src_x * dxyz[:, 0] / offsets)[:, None] * lambd ** 2
        em_x = ((C0x * rTE) @ fhtfilt.j0 + (C1x * rTE) @ fhtfilt.j1) / offsets
        em_y = ((C0y * rTE) @ fhtfilt.j0 + (C1y * rTE) @ fhtfilt.j1) / offsets
        em_z = ((C0z * rTE) @ fhtfilt.j0 + (C1z * rTE) @ fhtfilt.j1) / offsets

        e = mag_layer.moment / (4 * np.pi) * np.stack((em_x, em_y, em_z), axis=-1).squeeze()
        e_test = mag_layer.magnetic_field(xyz, field='secondary')
        np.testing.assert_equal(e_test, e)

    def test_magnetic_dipole(self):
        sigma = 1.0
        moment = 2.0
        frequencies = np.logspace(1, 4, 3)
        thickness = 10*np.ones(5)
        X, Y, Z = np.mgrid[-10:10:10, -10:10:10, 0:1]*100
        xyz = np.c_[X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]

        for orientation in ['Z', "Y", "Z"]:
            dip_layer = MagneticDipoleLayeredHalfSpace(
                frequencies,
                thickness,
                sigma=sigma,
                orientation=orientation,
                moment=moment,
            )
            dip_half = MagneticDipoleHalfSpace(
                frequencies,
                sigma=sigma,
                orientation=orientation,
                moment=moment,
            )

            with pytest.raises(ValueError):
                _dip_half = MagneticDipoleHalfSpace(
                    frequencies,
                    sigma=sigma,
                    orientation=orientation,
                    moment=moment,
                    location=np.r_[1, 1, 1]
                )
                _dip_half._check_is_valid_location()


            em_layer_total = dip_layer.magnetic_field(xyz, field='total')
            em_half_total = dip_half.magnetic_field(xyz, field='total')

            em_layer_sec = dip_layer.magnetic_field(xyz, field='secondary')
            em_half_sec = dip_half.magnetic_field(xyz, field='secondary')

            assert_allclose(em_layer_total, em_half_total, rtol=1e-05, atol=1e-08)
            assert_allclose(em_layer_sec, em_half_sec, rtol=1e-05, atol=1e-08)


class TestrTEGradient(unittest.TestCase):
    def test_rTE_jacobian(self):
        """Test to make sure numpy and compiled give same results"""
        n_layer = 11
        n_frequency = 5
        n_lambda = 8
        frequencies = np.logspace(1, 4, 5)
        thicknesses = np.ones(n_layer-1)
        lamb = np.logspace(0, 3, n_lambda)
        np.random.seed(0)
        sigma = 1E-1*(1 + 0.1 * np.random.rand(n_layer, n_frequency))
        mu = mu_0 * (1 + 0.1 * np.random.rand(n_layer, n_frequency))
        dmu = mu_0 * 0.1 * np.random.rand(n_layer, n_frequency)

        def rte_sigma(x):
            sigma = x.reshape(n_layer, n_frequency)
            rTE = rTE_forward(frequencies, lamb, sigma, mu, thicknesses)

            J_sigma, _, _ = rTE_gradient(frequencies, lamb, sigma, mu, thicknesses)

            def J(y):
                y = y.reshape(n_layer, n_frequency)
                # do summation over layers, broadcast over frequencies only
                # equivalent to:
                # out = np.empty_like(rTE)
                # for i in range(n_frequency):
                #     out[i] = J_sigma[:, i, :]).T@y[:, i]
                return np.einsum('i...k,i...', J_sigma, y)
            return rTE, J

        def rte_h(x):
            thicknesses = x
            rTE = rTE_forward(frequencies, lamb, sigma, mu, thicknesses)
            _, J_h, _ = rTE_gradient(frequencies, lamb, sigma, mu, thicknesses)

            def J(y):
                #(J_h.T@y).T
                out = np.einsum('i...,i', J_h, y)
                # do sum over n_layer, broadcast over all others
                return out

            return rTE, J

        def rte_mu(x):
            mu = x.reshape(n_layer, n_frequency)
            rTE = rTE_forward(frequencies, lamb, sigma, mu, thicknesses)

            _, _, J_mu = rTE_gradient(frequencies, lamb, sigma, mu, thicknesses)

            def J(y):
                y = y.reshape(n_layer, n_frequency)
                # do summation over layers, broadcast over frequencies only
                return np.einsum('i...k,i...', J_mu, y)
            return rTE, J

        self.assertTrue(check_derivative(rte_sigma, sigma.reshape(-1), num=4, plotIt=False))
        self.assertTrue(check_derivative(rte_h, thicknesses, num=4, plotIt=False))
        self.assertTrue(check_derivative(rte_mu, mu.reshape(-1), dx=dmu.reshape(-1), num=4, plotIt=False))


class TestCompiledVsNumpy(unittest.TestCase):
    def setUp(self):
        """Test to make sure numpy and compiled give same results"""
        n_layer = 11
        n_frequency = 5
        n_lambda = 8
        frequencies = np.logspace(-4, -2, n_frequency)
        thicknesses = np.ones(n_layer-1)
        lamb = np.logspace(-5, -3, n_lambda)
        np.random.seed(123)
        sigma = 1E-1 * (1 + 1.0/(n_layer*n_frequency) * np.arange(n_layer*n_frequency).reshape(n_layer, n_frequency))
        mu = mu_0 * (1 + 1.0/(n_layer*n_frequency) * np.arange(n_layer*n_frequency).reshape(n_layer, n_frequency))

        self.rTE1 = rTE_forward(frequencies, lamb, sigma, mu, thicknesses)
        self.rTE2 = _rTE_forward(frequencies, lamb, sigma, mu, thicknesses)

        self.rTE1_dsigma, self.rTE1_dh, self.rTE1_dmu = rTE_gradient(frequencies, lamb, sigma, mu, thicknesses)
        self.rTE2_dsigma, self.rTE2_dh, self.rTE2_dmu = _rTE_gradient(frequencies, lamb, sigma, mu, thicknesses)

    def test_rTE(self):
        assert_allclose(self.rTE1, self.rTE2, atol=1E-15)

    def test_rTE_dsigma(self):
        non_zeros2 = np.abs(self.rTE2_dsigma) != 0.0
        # only compare non-zeros in derivatives rTE2
        # (the compiled routine (rTE1) is slightly more accurate)
        assert_allclose(self.rTE1_dsigma[non_zeros2], self.rTE2_dsigma[non_zeros2])

    def test_rTE_dh(self):
        non_zeros2 = np.abs(self.rTE2_dh) != 0.0
        # only compare non-zeros in derivatives rTE2
        # (the compiled routine (rTE1) is slightly more accurate)
        assert_allclose(self.rTE1_dh[non_zeros2], self.rTE2_dh[non_zeros2])

    def test_rTE_dmu(self):
        non_zeros2 = np.abs(self.rTE2_dmu) != 0.0
        # only compare non-zeros in derivatives rTE2
        # (the compiled routine (rTE1) is slightly more accurate)
        assert_allclose(self.rTE1_dmu[non_zeros2], self.rTE2_dmu[non_zeros2])


if __name__ == '__main__':
    unittest.main()

