import unittest
import numpy as np
from numpy.testing import assert_allclose
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
        sigma = np.random.rand(n_layer, n_frequency)
        chi = np.random.rand(n_layer, n_frequency)

        def rte_sigma(x):
            sigma = x.reshape(n_layer, n_frequency)
            rTE = rTE_forward(frequencies, lamb, sigma, chi, thicknesses)

            J_sigma, _, _ = rTE_gradient(frequencies, lamb, sigma, chi, thicknesses)

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
            rTE = rTE_forward(frequencies, lamb, sigma, chi, thicknesses)
            _, J_h, _ = rTE_gradient(frequencies, lamb, sigma, chi, thicknesses)

            def J(y):
                #(J_h.T@y).T
                out = np.einsum('i...,i', J_h, y)
                # do sum over n_layer, broadcast over all others
                return out

            return rTE, J

        def rte_chi(x):
            chi = x.reshape(n_layer, n_frequency)
            rTE = rTE_forward(frequencies, lamb, sigma, chi, thicknesses)

            _, _, J_chi = rTE_gradient(frequencies, lamb, sigma, chi, thicknesses)

            def J(y):
                y = y.reshape(n_layer, n_frequency)
                # do summation over layers, broadcast over frequencies only
                return np.einsum('i...k,i...', J_chi, y)
            return rTE, J

        self.assertTrue(check_derivative(rte_sigma, sigma.reshape(-1), num=4, plotIt=False))
        self.assertTrue(check_derivative(rte_h, thicknesses, num=4, plotIt=False))
        self.assertTrue(check_derivative(rte_chi, chi.reshape(-1), num=4, plotIt=False))

class TestCompiledVsNumpy(unittest.TestCase):

    def test_rTE(self):
        """Test to make sure numpy and compiled give same results"""
        n_layer = 11
        n_frequency = 5
        n_lambda = 8
        frequencies = np.logspace(1, 4, 5)
        thicknesses = np.ones(n_layer-1)
        lamb = np.logspace(0, 3, n_lambda)
        sigma = np.random.rand(n_layer, n_frequency)
        chi = np.random.rand(n_layer, n_frequency)

        rTE1 = rTE_forward(frequencies, lamb, sigma, chi, thicknesses)
        rTE2 = _rTE_forward(frequencies, lamb, sigma, chi, thicknesses)

        assert_allclose(rTE1, rTE1)

        rTE1_dsigma, rTE1_dh, rTE1_dchi = rTE_gradient(frequencies, lamb, sigma, chi, thicknesses)
        rTE2_dsigma, rTE2_dh, rTE2_dchi = _rTE_gradient(frequencies, lamb, sigma, chi, thicknesses)

        assert_allclose(rTE1_dsigma, rTE2_dsigma)
        assert_allclose(rTE1_dh, rTE2_dh)
        assert_allclose(rTE1_dchi, rTE2_dchi)
