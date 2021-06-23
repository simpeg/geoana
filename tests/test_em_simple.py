import numpy as np
import unittest
from scipy.constants import mu_0
from geoana.em.fdem import (
    vertical_magnetic_field_horizontal_loop as f_hz_loop,
    vertical_magnetic_flux_horizontal_loop as f_bz_loop
)
from geoana.em.tdem import(
    vertical_magnetic_field_horizontal_loop as t_hz_loop,
    vertical_magnetic_flux_horizontal_loop as t_bz_loop,
    vertical_magnetic_field_time_deriv_horizontal_loop as t_hzdt_loop,
    vertical_magnetic_flux_time_deriv_horizontal_loop as t_bzdt_loop,
    magnetic_field_vertical_magnetic_dipole as h_dipv,
    magnetic_field_time_deriv_magnetic_dipole as hdt_dipv,
    magnetic_flux_vertical_magnetic_dipole as b_dipv,
    magnetic_flux_time_deriv_magnetic_dipole as bdt_dipv,
)
from geoana.em.static import (
    CircularLoopWholeSpace
)


class testExamples(unittest.TestCase):

    def test_frequency_loop(self):
        frequencies = np.logspace(-3, 6, 25)
        radius = 50
        hz_total = f_hz_loop(frequencies, sigma=1E-2, radius=radius, secondary=False)
        # test 0 frequency limit
        np.testing.assert_allclose(hz_total[0].real, 1/(2 * radius))

        hz_secondary = f_hz_loop(frequencies, sigma=1E-2, radius=radius)

        h_p = hz_total - hz_secondary
        loop = CircularLoopWholeSpace(radius=radius)
        h_p2 = loop.magnetic_field(np.array([[0, 0, 0]]))[0, 0]
        np.testing.assert_allclose(h_p, h_p2)

        bz_secondary = f_bz_loop(frequencies, sigma=1E-2, radius=radius)
        np.testing.assert_equal(mu_0 * hz_secondary, bz_secondary)

    def test_time_loop(self):
        times = np.logspace(-14, -1, 200)
        radius = 50
        sigma = 1E-2
        hz = t_hz_loop(times, sigma=sigma, radius=radius)

        # test 0 time limit
        np.testing.assert_allclose(hz[0], 1/(2 * radius))

        bz = t_bz_loop(times, sigma=sigma, radius=radius)
        np.testing.assert_equal(mu_0 * hz, bz)

        dhz_dt = t_hzdt_loop(times, sigma=sigma, radius=radius)
        # test 0 time limit
        np.testing.assert_allclose(dhz_dt[0], -3/(mu_0 * sigma * radius**3))

        dbz_dt = t_bzdt_loop(times, sigma=sigma, radius=radius)
        np.testing.assert_allclose(mu_0 * dhz_dt, dbz_dt)

    def test_time_vertical_dipole(self):
        times = np.logspace(-14, 0, 200)
        offset = 100
        xy = np.array([[offset, 0, 0]])
        sigma = 1E-2
        h = h_dipv(times, xy, sigma=sigma)[:, 0, :]

        # test 0 time limit
        np.testing.assert_allclose(h[0, 2], -1/(4*np.pi * offset ** 3))
        # the first three are nans (due to stability)
        np.testing.assert_allclose(h[3, 0], 0.0, atol=1E-9)

        b = b_dipv(times, xy, sigma=sigma)[:, 0, :]
        np.testing.assert_equal(mu_0 * h, b)

        dh_dt = hdt_dipv(times, xy, sigma=sigma)[:, 0, :]
        # test 0 time limit
        np.testing.assert_allclose(dh_dt[0, 2], 9/(2 * np.pi * mu_0 * sigma * offset**5))
        np.testing.assert_allclose(dh_dt[3, 0], 0.0)

        db_dt = bdt_dipv(times, xy, sigma=sigma)[:, 0, :]
        np.testing.assert_allclose(mu_0 * dh_dt, db_dt)
