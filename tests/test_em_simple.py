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
        times = np.logspace(-14, 0, 200)
        radius = 50
        sigma = 1E-2
        hz = t_hz_loop(times, sigma=sigma, radius=radius)

        # test 0 time limit
        np.testing.assert_allclose(hz[0], 1/(2 * radius))

        # test late time limit
        np.testing.assert_allclose(
            hz[-1],
            radius**2/(30*np.sqrt(np.pi))*(sigma*mu_0/times[-1])**(1.5),
            rtol=1E-4
        )

        bz = t_bz_loop(times, sigma=sigma, radius=radius)
        np.testing.assert_equal(mu_0 * hz, bz)

        dhz_dt = t_hzdt_loop(times, sigma=sigma, radius=radius)
        # test 0 time limit
        np.testing.assert_allclose(dhz_dt[0], -3/(mu_0 * sigma * radius**3))
        # test late time limit
        np.testing.assert_allclose(
            dhz_dt[-1],
            -radius**2/(20*np.sqrt(np.pi))*(sigma*mu_0/times[-1])**(1.5)/times[-1],
            rtol=1E-4
        )

        dbz_dt = t_bzdt_loop(times, sigma=sigma, radius=radius)
        np.testing.assert_allclose(mu_0 * dhz_dt, dbz_dt)

    def test_time_vertical_dipole(self):
        times = np.logspace(-14, 1, 200)
        offset = 100
        xy = np.array([[offset, 0, 0]])
        sigma = 1E-2
        h = h_dipv(times, xy, sigma=sigma)[:, 0, :]

        # test 0 time limit
        np.testing.assert_allclose(h[0, 2], -1/(4*np.pi * offset ** 3))
        # test late time limit
        np.testing.assert_allclose(
            h[-1, 0],
            offset/(128*np.pi)*(mu_0*sigma/times[-1])**2,
            rtol=1E-3,
        )
        np.testing.assert_allclose(
            h[-1, 2],
            1/30 * (sigma * mu_0 /np.pi)**1.5*times[-1]**-1.5,
            rtol=1E-5,
        )


        b = b_dipv(times, xy, sigma=sigma)[:, 0, :]
        np.testing.assert_equal(mu_0 * h, b)

        dh_dt = hdt_dipv(times, xy, sigma=sigma)[:, 0, :]
        # test 0 time limit
        np.testing.assert_allclose(
            dh_dt[0, 2], 9/(2 * np.pi * mu_0 * sigma * offset**5)
        )
        # test late times
        np.testing.assert_allclose(
            dh_dt[-1, 0],
            -offset/(64*np.pi)*(mu_0*sigma/times[-1])**2/times[-1],
            rtol=1E-5,
        )
        np.testing.assert_allclose(
            dh_dt[-1, 2],
            -1/20 * (sigma * mu_0 / np.pi)**1.5*times[-1]**-2.5,
            rtol=1E-4
        )

        db_dt = bdt_dipv(times, xy, sigma=sigma)[:, 0, :]
        np.testing.assert_allclose(mu_0 * dh_dt, db_dt)
