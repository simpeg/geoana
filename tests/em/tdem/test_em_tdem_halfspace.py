import numpy as np

from geoana.em.tdem import VerticalMagneticDipoleHalfSpace
from geoana.em.tdem.base import theta as theta_func
from scipy.special import erf, ive

from geoana.em.tdem.reference import hp_from_vert_4_72, hz_from_vert_4_69a
from geoana.spatial import cylindrical_to_cartesian, cartesian_to_cylindrical


def H_from_Vertical(
    XYZ, loc, time, sigma, mu, moment
):
    theta = theta_func(time, sigma, mu)
    r_vec = XYZ - loc
    cyl_locs = cylindrical_to_cartesian(r_vec)
    hp = hp_from_vert_4_72(moment, theta, cyl_locs[..., 0])
    hz = hz_from_vert_4_69a(moment, theta, cyl_locs[..., 0])
    h_cyl = np.stack([hp, np.zeros_like(hp), hz], axis=-1)
    print(h_cyl.shape, r_vec.shape, hp.shape)
    return cartesian_to_cylindrical(cyl_locs, h_cyl)


def dH_from_Vertical(
    XYZ, loc, time, sigma, mu, moment
):
    XYZ = np.atleast_2d(XYZ)

    r_vec = XYZ - loc
    r = np.linalg.norm(r_vec[:, :2], axis=-1)
    x = r_vec[:, 0]
    y = r_vec[:, 1]
    tr = theta_func(time, sigma, mu) * r

    dhz_dt = 1 / (r ** 3 * time) * (
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
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

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
        z = np.linspace(-30, 0, 0)
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

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
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

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
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        db_test = dB_from_Vertical(
            xyz, vmdhs.location, vmdhs.time, vmdhs.sigma, vmdhs.mu, vmdhs.moment
        )
        print(
            "\n\nTesting Vertical Magnetic Dipole Halfspace Magnetic Flux Density Derivative dB\n"
        )

        db = vmdhs.magnetic_flux_time_derivative(xyz)
        np.testing.assert_equal(db_test, db)