import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import mu_0
from geoana.em.fdem import MagneticDipoleHalfSpace, MagneticDipoleLayeredHalfSpace
from geoana.kernels.tranverse_electric_reflections import (
    rTE_forward, rTE_gradient, _rTE_forward, _rTE_gradient
)

from discretize.Tests import checkDerivative as check_derivative

class TestLayeredHalfspace(unittest.TestCase):

    def test_magnetic_dipole(self):
        sigma = 1.0
        moment = 2.0
        frequencies = np.logspace(1, 4, 3)
        X, Y, Z = np.mgrid[-10:10:10j, -10:10:10j, 0:1]*100
        xyz = np.c_[X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]

        for orientation in ['Z', "Y", "Z"]:
            dip_layer = MagneticDipoleLayeredHalfSpace(
                frequency=frequencies,
                thickness=np.ones(5)*10,
                sigma=[sigma + 0j],
                orientation=orientation,
                moment=moment,
            )
            dip_half = MagneticDipoleHalfSpace(
                frequency=frequencies,
                sigma=sigma,
                orientation=orientation,
                moment=moment,
            )

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

    def test_rTE(self):
        """Test to make sure numpy and compiled give same results"""
        n_layer = 11
        n_frequency = 5
        n_lambda = 8
        frequencies = np.logspace(-3, 1, 5)
        thicknesses = np.ones(n_layer-1)
        lamb = np.logspace(0, 3, n_lambda)
        np.random.seed(123)
        sigma = 1E-1 * (1 + 1.0/(n_layer*n_frequency) * np.arange(n_layer*n_frequency).reshape(n_layer, n_frequency))
        mu = mu_0 * (1 + 1.0/(n_layer*n_frequency) * np.arange(n_layer*n_frequency).reshape(n_layer, n_frequency))

        rTE1 = rTE_forward(frequencies, lamb, sigma, mu, thicknesses)
        rTE2 = _rTE_forward(frequencies, lamb, sigma, mu, thicknesses)

        assert_allclose(rTE1, rTE2, atol=1E-16)

        rTE1_dsigma, rTE1_dh, rTE1_dmu = rTE_gradient(frequencies, lamb, sigma, mu, thicknesses)
        rTE2_dsigma, rTE2_dh, rTE2_dmu = _rTE_gradient(frequencies, lamb, sigma, mu, thicknesses)

        non_zeros2 = np.abs(rTE2_dsigma) != 0.0
        # only compare non-zeros in derivatives rTE2
        # (the compiled routine (rTE1) is slightly more accurate)
        assert_allclose(rTE1_dsigma[non_zeros2], rTE2_dsigma[non_zeros2])
        non_zeros2 = np.abs(rTE2_dh) != 0.0
        assert_allclose(rTE1_dh[non_zeros2], rTE2_dh[non_zeros2])
        non_zeros2 = np.abs(rTE2_dmu) != 0.0
        assert_allclose(rTE1_dmu[non_zeros2], rTE2_dmu[non_zeros2])
