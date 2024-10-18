import numpy as np
import numpy.testing as npt

from geoana.em.tdem import VerticalMagneticDipoleHalfSpace
from geoana.em.tdem.base import theta as theta_func

from geoana.em.tdem.reference import hp_from_vert_4_72, hz_from_vert_4_69a, dhz_from_vert_4_70, dhp_from_vert_4_74
from geoana.spatial import cylindrical_to_cartesian, cartesian_to_cylindrical


def H_from_Vertical(
    XYZ, loc, time, sigma, mu, moment
):
    theta = theta_func(time, sigma, mu)
    r_vec = XYZ - loc
    cyl_locs = cartesian_to_cylindrical(r_vec)
    hp = -hp_from_vert_4_72(moment, theta, cyl_locs[..., 0])
    hz = hz_from_vert_4_69a(moment, theta, cyl_locs[..., 0])
    h_cyl = np.stack([hp, np.zeros_like(hp), hz], axis=-1)
    return cylindrical_to_cartesian(cyl_locs, h_cyl)


def dH_from_Vertical(
    XYZ, loc, time, sigma, mu, moment
):
    XYZ = np.atleast_2d(XYZ)

    theta = theta_func(time, sigma, mu)
    r_vec = XYZ - loc
    cyl_locs = cartesian_to_cylindrical(r_vec)
    dhp = -dhp_from_vert_4_74(moment, theta, cyl_locs[..., 0], time)
    dhz = -dhz_from_vert_4_70(moment, theta, cyl_locs[..., 0], sigma, mu)
    h_cyl = np.stack([dhp, np.zeros_like(dhp), dhz], axis=-1)
    return cylindrical_to_cartesian(cyl_locs, h_cyl)


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
        time = 1E-3
        vmdhs = VerticalMagneticDipoleHalfSpace(time=time, moment=1E5)
        x = np.linspace(-200., 200., 50)
        y = np.linspace(-300., 300., 50)
        z = [0, ]
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1)

        htest = H_from_Vertical(
            xyz, vmdhs.location[:-2], vmdhs.time, vmdhs.sigma, vmdhs.mu, vmdhs.moment
        )
        print(
            "\n\nTesting Vertical Magnetic Dipole Halfspace Magnetic Field H\n"
        )

        h = vmdhs.magnetic_field(xyz)
        npt.assert_allclose(h, htest)

    def test_magnetic_flux(self):
        time = 1E-3
        vmdhs = VerticalMagneticDipoleHalfSpace(time=time)
        x = np.linspace(-200., 200., 50)
        y = np.linspace(-300., 300., 50)
        z = [0, ]
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        btest = B_from_Vertical(
            xyz, vmdhs.location, vmdhs.time, vmdhs.sigma, vmdhs.mu, vmdhs.moment
        )
        print(
            "\n\nTesting Vertical Magnetic Dipole Halfspace Magnetic Flux Density B\n"
        )

        b = vmdhs.magnetic_flux_density(xyz)
        npt.assert_allclose(b, btest)

    def test_magnetic_field_dt(self):
        time = 1E-3
        vmdhs = VerticalMagneticDipoleHalfSpace(time=time)
        x = np.linspace(-200., 200., 50)
        y = np.linspace(-300., 300., 50)
        z = [0, ]
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        dh_test = dH_from_Vertical(
            xyz, vmdhs.location, vmdhs.time, vmdhs.sigma, vmdhs.mu, vmdhs.moment
        )
        print(
            "\n\nTesting Vertical Magnetic Dipole Halfspace Magnetic Field Derivative dH\n"
        )

        dh = vmdhs.magnetic_field_time_derivative(xyz)
        npt.assert_allclose(dh_test, dh)

    def test_magnetic_flux_dt(self):
        time = 1E-3
        vmdhs = VerticalMagneticDipoleHalfSpace(time=time)
        x = np.linspace(-200., 200., 50)
        y = np.linspace(-300., 300., 50)
        z = [0, ]
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        db_test = dB_from_Vertical(
            xyz, vmdhs.location, vmdhs.time, vmdhs.sigma, vmdhs.mu, vmdhs.moment
        )
        print(
            "\n\nTesting Vertical Magnetic Dipole Halfspace Magnetic Flux Density Derivative dB\n"
        )

        db = vmdhs.magnetic_flux_time_derivative(xyz)
        npt.assert_allclose(db_test, db)