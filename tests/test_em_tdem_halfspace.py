import numpy as np

from geoana.em.tdem import VerticalMagneticDipoleHalfSpace
import discretize
from geoana.em.tdem.base import theta
from scipy.special import erf, ive


def H_from_Vertical(
    XYZ, loc, time, sigma, mu, moment
):

    XYZ = discretize.utils.as_array_n_by_dim(XYZ, 3)

    r_vec = XYZ - loc
    r = np.linalg.norm(r_vec[:, :2], axis=-1)
    x = r_vec[:, 0]
    y = r_vec[:, 1]
    thr = theta(time, sigma, mu=mu)[:, None] * r

    h_z = 1.0 / r ** 3 * (
            (9 / (2 * thr ** 2) - 1) * erf(thr)
            - (9 / thr + 4 * thr) / np.sqrt(np.pi) * np.exp(-thr ** 2)
    )

    h_r = 2 * thr ** 2 / r ** 3 * (
            ive(1, thr ** 2 / 2) - ive(2, thr ** 2 / 2)
    )

    angle = np.arctan2(y, x)
    h_x = np.cos(angle) * h_r
    h_y = np.sin(angle) * h_r

    h = moment / (4 * np.pi) * np.stack((h_x, h_y, h_z), axis=-1)
    return h[0]


def dH_from_Vertical(
    XYZ, loc, time, sigma, mu, moment
):

    XYZ = discretize.utils.as_array_n_by_dim(XYZ, 3)

    r_vec = XYZ - loc
    r = np.linalg.norm(r_vec[:, :2], axis=-1)
    x = r_vec[:, 0]
    y = r_vec[:, 1]
    tr = theta(time, sigma, mu)[:, None] * r

    dhz_dt = 1 / (r ** 3 * time[:, None]) * (
            9 / (2 * tr ** 2) * erf(tr)
            - (4 * tr ** 3 + 6 * tr + 9 / tr) / np.sqrt(np.pi) * np.exp(-tr ** 2)
    )

    dhr_dt = - 2 * tr ** 2 / (r ** 3 * time[:, None]) * (
            (1 + tr ** 2) * ive(0, tr ** 2 / 2) -
            (2 + tr ** 2 + 4 / tr ** 2) * ive(1, tr ** 2 / 2)
    )
    angle = np.arctan2(y, x)
    dhx_dt = np.cos(angle) * dhr_dt
    dhy_dt = np.sin(angle) * dhr_dt
    dh_dt = moment / (4 * np.pi) * np.stack((dhx_dt, dhy_dt, dhz_dt), axis=-1)
    return dh_dt[0]


def B_from_Vertical(
    XYZ, loc, time, sigma, mu, moment
):
    b = mu * H_from_Vertical(XYZ, loc, time, sigma, mu, moment)
    return b


def dB_from_Vertical(
    XYZ, loc, time, sigma, mu, moment
):
    db = mu * dH_from_Vertical(XYZ, loc, time, sigma, mu, moment)
    return db


class TestVerticalMagneticDipoleHalfSpace:

    def test_magnetic_field(self):
        time = 1
        vmdhs = VerticalMagneticDipoleHalfSpace(time=time)
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])

        htest = H_from_Vertical(
            xyz, vmdhs.location, vmdhs.time, vmdhs.sigma, vmdhs.mu, vmdhs.moment
        )
        print(
            "\n\nTesting Vertical Magnetic Dipole Halfspace Magnetic Field H\n"
        )

        h = vmdhs.magnetic_field(xyz)
        np.testing.assert_equal(htest, h)

    def test_magnetic_flux(self):
        time = 1
        vmdhs = VerticalMagneticDipoleHalfSpace(time=time)
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])

        btest = B_from_Vertical(
            xyz, vmdhs.location, vmdhs.time, vmdhs.sigma, vmdhs.mu, vmdhs.moment
        )
        print(
            "\n\nTesting Vertical Magnetic Dipole Halfspace Magnetic Flux Density B\n"
        )

        b = vmdhs.magnetic_flux_density(xyz)
        np.testing.assert_equal(btest, b)

    def test_magnetic_field_dt(self):
        time = 1
        vmdhs = VerticalMagneticDipoleHalfSpace(time=time)
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])

        dh_test = dH_from_Vertical(
            xyz, vmdhs.location, vmdhs.time, vmdhs.sigma, vmdhs.mu, vmdhs.moment
        )
        print(
            "\n\nTesting Vertical Magnetic Dipole Halfspace Magnetic Field Derivative dH\n"
        )

        dh = vmdhs.magnetic_field_time_derivative(xyz)
        np.testing.assert_equal(dh_test, dh)

    def test_magnetic_flux_dt(self):
        time = 1
        vmdhs = VerticalMagneticDipoleHalfSpace(time=time)
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])

        db_test = dB_from_Vertical(
            xyz, vmdhs.location, vmdhs.time, vmdhs.sigma, vmdhs.mu, vmdhs.moment
        )
        print(
            "\n\nTesting Vertical Magnetic Dipole Halfspace Magnetic Flux Density Derivative dB\n"
        )

        db = vmdhs.magnetic_flux_time_derivative(xyz)
        np.testing.assert_equal(db_test, db)