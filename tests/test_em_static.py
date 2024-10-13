import pytest
import unittest
import numpy as np
from scipy.constants import mu_0, epsilon_0
try:
    import discretize
except ImportError:
    discretize = None
from scipy.special import ellipk, ellipe

from geoana.em import static, fdem
from geoana import spatial

TOL = 0.1


class TestEM_Static(unittest.TestCase):

    def setUp(self):
        self.mdws = static.MagneticDipoleWholeSpace()
        self.mpws = static.MagneticPoleWholeSpace()
        self.clws = static.CircularLoopWholeSpace()
        self.lcfs = static.LineCurrentFreeSpace(nodes=np.c_[1., 1., 1.])

    def test_defaults(self):
        self.assertTrue(self.mdws.sigma == 1.0)
        self.assertTrue(self.mpws.sigma == 1.0)
        self.assertTrue(self.clws.sigma == 1.0)

        self.assertTrue(self.mdws.mu == mu_0)
        self.assertTrue(self.mpws.mu == mu_0)
        self.assertTrue(self.clws.mu == mu_0)

        self.assertTrue(self.mdws.epsilon == epsilon_0)
        self.assertTrue(self.mpws.epsilon == epsilon_0)
        self.assertTrue(self.clws.epsilon == epsilon_0)

        self.assertTrue(np.all(self.mdws.orientation == np.r_[1., 0., 0.]))
        self.assertTrue(np.all(self.mpws.orientation == np.r_[1., 0., 0.]))
        self.assertTrue(np.all(self.clws.orientation == np.r_[1., 0., 0.]))

        self.assertTrue(np.all(self.lcfs.nodes == np.c_[1., 1., 1.]))

        self.assertTrue(self.mdws.moment == 1.0)
        self.assertTrue(self.mpws.moment == 1.0)
        self.assertTrue(self.clws.current == 1.0)
        self.assertTrue(self.clws.radius == np.sqrt(1/np.pi))

    def test_errors(self):
        with pytest.raises(TypeError):
            self.mdws.mu = "box"
        with pytest.raises(ValueError):
            self.mdws.mu = -2
        with pytest.raises(TypeError):
            self.mdws.moment = "box"
        with pytest.raises(ValueError):
            self.mdws.orientation = [0, 1, 2, 3, 4]
        with pytest.raises(ValueError):
            self.mdws.orientation = [[0, 0], [0, 1]]
        with pytest.raises(TypeError):
            self.mdws.orientation = ["string"]
        with pytest.raises(ValueError):
            self.clws.location = [0, 1, 2, 3]
        with pytest.raises(ValueError):
            self.clws.location = [[0, 0], [0, 1]]
        with pytest.raises(TypeError):
            self.clws.location = ["string"]
        with pytest.raises(TypeError):
            self.clws.radius = "box"
        with pytest.raises(ValueError):
            self.clws.radius = -2
        with pytest.raises(TypeError):
            self.clws.current = "box"
        with pytest.raises(TypeError):
            x = np.linspace(-20., 20., 50)
            y = np.linspace(-30., 30., 50)
            z = np.linspace(-40., 40., 50)
            xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)
            self.clws.magnetic_flux_density(xyz, coordinates='square')
        with pytest.raises(ValueError):
            self.lcfs.nodes = [0, 1, 2, 3, 4, 5]
        with pytest.raises(ValueError):
            self.lcfs.nodes = [[0, 0], [0, 1]]
        with pytest.raises(TypeError):
            self.lcfs.nodes = ["string"]
        with pytest.raises(TypeError):
            self.lcfs.current = "box"

    def test_vector_potential(self):
        n = 50
        x = y = z = np.linspace(-25, 25, n)
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        for orientation in ["x", "y", "z"]:
            self.clws.orientation = orientation
            self.mdws.orientation = orientation

            inds = (
                (np.absolute(xyz[:, 0]) > 5) &
                (np.absolute(xyz[:, 1]) > 5) &
                (np.absolute(xyz[:, 2]) > 5)
            )

            a_clws = self.clws.vector_potential(xyz)[inds]
            a_mdws = self.mdws.vector_potential(xyz)[inds]

            self.assertTrue(isinstance(a_clws, np.ndarray))
            self.assertTrue(isinstance(a_mdws, np.ndarray))

            self.assertTrue(
                np.linalg.norm(a_clws - a_mdws) <
                0.5 * TOL * (np.linalg.norm(a_clws) + np.linalg.norm(a_mdws))
            )

    if discretize is not None:
        def test_magnetic_field_tensor(self):
            print("\n === Testing Tensor Mesh === \n")
            n = 30
            h = 2.
            mesh = discretize.TensorMesh(
                [np.full(n, h), np.full(n, h), np.full(n, h)],
                origin='CCC'
            )

            for radius in [0.5, 1, 1.5]:
                self.clws.radius = radius
                self.clws.current = 1./(np.pi * radius**2)

                fdem_dipole = fdem.MagneticDipoleWholeSpace(frequency=0)

                for location in [
                    np.r_[0, 0, 0], np.r_[10, 0, 0], np.r_[0, -10, 0],
                    np.r_[0, 0, 10], np.r_[10, 10, 10]
                ]:
                    self.clws.location = location
                    self.mdws.location = location
                    fdem_dipole.location = location

                    for orientation in ["x", "y", "z"]:
                        self.clws.orientation = orientation
                        self.mdws.orientation = orientation
                        fdem_dipole.orientation = orientation

                        a_clws = np.hstack([
                            self.clws.vector_potential(mesh.gridEx)[:, 0],
                            self.clws.vector_potential(mesh.gridEy)[:, 1],
                            self.clws.vector_potential(mesh.gridEz)[:, 2]
                        ])

                        a_mdws = np.hstack([
                            self.mdws.vector_potential(mesh.gridEx)[:, 0],
                            self.mdws.vector_potential(mesh.gridEy)[:, 1],
                            self.mdws.vector_potential(mesh.gridEz)[:, 2]
                        ])

                        b_clws = mesh.edge_curl * a_clws
                        b_mdws = mesh.edge_curl * a_mdws

                        b_clws_true = np.hstack([
                            self.clws.magnetic_flux_density(mesh.gridFx)[:, 0],
                            self.clws.magnetic_flux_density(mesh.gridFy)[:, 1],
                            self.clws.magnetic_flux_density(mesh.gridFz)[:, 2]
                        ])
                        b_mdws_true = np.hstack([
                            self.mdws.magnetic_flux_density(mesh.gridFx)[:, 0],
                            self.mdws.magnetic_flux_density(mesh.gridFy)[:, 1],
                            self.mdws.magnetic_flux_density(mesh.gridFz)[:, 2]
                        ])
                        b_fdem = np.hstack([
                            fdem_dipole.magnetic_flux_density(mesh.gridFx)[:, 0],
                            fdem_dipole.magnetic_flux_density(mesh.gridFy)[:, 1],
                            fdem_dipole.magnetic_flux_density(mesh.gridFz)[:, 2]
                        ])

                        self.assertTrue(isinstance(b_fdem, np.ndarray))

                        inds = (np.hstack([
                            (np.absolute(mesh.gridFx[:, 0]) > h*2 + location[0]) &
                            (np.absolute(mesh.gridFx[:, 1]) > h*2 + location[1]) &
                            (np.absolute(mesh.gridFx[:, 2]) > h*2 + location[2]),
                            (np.absolute(mesh.gridFy[:, 0]) > h*2 + location[0]) &
                            (np.absolute(mesh.gridFy[:, 1]) > h*2 + location[1]) &
                            (np.absolute(mesh.gridFy[:, 2]) > h*2 + location[2]),
                            (np.absolute(mesh.gridFz[:, 0]) > h*2 + location[0]) &
                            (np.absolute(mesh.gridFz[:, 1]) > h*2 + location[1]) &
                            (np.absolute(mesh.gridFz[:, 2]) > h*2 + location[2])
                        ]))

                        loop_passed_1 = (
                            np.linalg.norm(b_fdem[inds] - b_clws[inds]) <
                            0.5 * TOL * (
                                np.linalg.norm(b_fdem[inds]) +
                                np.linalg.norm(b_clws[inds])
                            )
                        )

                        loop_passed_2 = (
                            np.linalg.norm(b_fdem[inds] - b_clws_true[inds]) <
                            0.5 * TOL * (
                                np.linalg.norm(b_fdem[inds]) +
                                np.linalg.norm(b_clws_true[inds])
                            )
                        )

                        dipole_passed = (
                            np.linalg.norm(b_fdem[inds] - b_mdws[inds]) <
                            0.5 * TOL * (
                                np.linalg.norm(b_fdem[inds]) +
                                np.linalg.norm(b_mdws[inds])
                            )
                        )

                        print(
                            "Testing r = {}, loc = {}, orientation = {}".format(
                                radius, location, orientation
                            )
                        )
                        print(
                            "  fdem: {:1.4e}, loop: {:1.4e}, dipole: {:1.4e}"
                            " Passed? loop: {}, dipole: {} \n".format(
                                np.linalg.norm(b_fdem[inds]),
                                np.linalg.norm(b_clws[inds]),
                                np.linalg.norm(b_mdws[inds]),
                                loop_passed_1,
                                loop_passed_2,
                                dipole_passed
                            )
                        )
                        self.assertTrue(loop_passed_1)
                        self.assertTrue(loop_passed_2)
                        self.assertTrue(dipole_passed)

    def test_dipole_magnetic_flux_density(self):
        mdws = static.MagneticDipoleWholeSpace()
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)
        n_obs = xyz.shape[0]

        xyz_ = spatial.cylindrical_2_cartesian(xyz)
        r = mdws.vector_distance(xyz_)
        dxyz = spatial.repeat_scalar(mdws.distance(xyz_))
        m_vec = mdws.moment * np.atleast_2d(mdws.orientation).repeat(n_obs, axis=0)
        m_dot_r = (m_vec * r).sum(axis=1)
        m_dot_r = np.atleast_2d(m_dot_r).T.repeat(3, axis=1)

        b_test = (mu_0 / (4 * np.pi)) * ((3 * r * m_dot_r / (dxyz ** 5)) - m_vec / (dxyz ** 3))
        b_test = spatial.cartesian_2_cylindrical(xyz_, b_test)

        b = mdws.magnetic_flux_density(xyz, coordinates='cylindrical')
        np.testing.assert_equal(b_test, b)

    def test_pole_magnetic_flux_density(self):
        mpws = static.MagneticPoleWholeSpace()
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        xyz_ = spatial.cylindrical_2_cartesian(xyz)
        r = mpws.vector_distance(xyz_)
        dxyz = spatial.repeat_scalar(mpws.distance(xyz_))

        b_test = mpws.moment * mu_0 / (4 * np.pi * (dxyz ** 3)) * r
        b_test = spatial.cartesian_2_cylindrical(xyz_, b_test)

        b = mpws.magnetic_flux_density(xyz, coordinates='cylindrical')
        np.testing.assert_equal(b_test, b)

    def test_pole_magnetic_field(self):
        mpws = static.MagneticPoleWholeSpace()
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        xyz_ = spatial.cylindrical_2_cartesian(xyz)
        r = mpws.vector_distance(xyz_)
        dxyz = spatial.repeat_scalar(mpws.distance(xyz_))

        h_test = mpws.moment * mu_0 / (4 * np.pi * (dxyz ** 3)) * r
        h_test = spatial.cartesian_2_cylindrical(xyz_, h_test)
        h_test = h_test / mu_0

        h = mpws.magnetic_field(xyz, coordinates='cylindrical')
        np.testing.assert_equal(h_test, h)

    def test_loop_magnetic_flux_density(self):
        clws = static.CircularLoopWholeSpace()
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        xyz_ = np.atleast_2d(xyz)
        xyz_ = spatial.cylindrical_2_cartesian(xyz_)
        xyz_ = spatial.rotate_points_from_normals(xyz_, np.array(clws.orientation), np.r_[0, 0, 1],
                                                  x0=np.array(clws.location))
        dxyz = clws.vector_distance(xyz_)
        rho = np.linalg.norm(dxyz[:, :2], axis=-1)
        b = np.zeros((len(rho), 3))
        ind_axial = rho == 0
        b[ind_axial, -1] = mu_0 * clws.current * clws.radius ** 2 / (2 * (clws.radius ** 2 +
                                                                          dxyz[ind_axial, 2] ** 2) ** (1.5))
        alpha = rho[~ind_axial] / clws.radius
        beta = dxyz[~ind_axial, 2] / clws.radius
        gamma = dxyz[~ind_axial, 2] / rho[~ind_axial]
        q = ((1 + alpha) ** 2 + beta ** 2)
        k2 = 4 * alpha / q
        b[~ind_axial, -1] = mu_0 * clws.current / (2 * clws.radius * np.pi * np.sqrt(q)) * \
            (ellipe(k2) * (1 - alpha ** 2 - beta ** 2) / (q - 4 * alpha) + ellipk(k2))
        b_rad = mu_0 * clws.current * gamma / (2 * clws.radius * np.pi * np.sqrt(q)) * \
            (ellipe(k2) * (1 + alpha ** 2 + beta ** 2) / (q - 4 * alpha) - ellipk(k2))
        b[~ind_axial, 0] = b_rad * (dxyz[~ind_axial, 0] / rho[~ind_axial])
        b[~ind_axial, 1] = b_rad * (dxyz[~ind_axial, 1] / rho[~ind_axial])
        b = spatial.rotate_points_from_normals(b, np.r_[0, 0, 1], np.array(clws.orientation),)
        b = spatial.cartesian_2_cylindrical(xyz_, b)

        b_test = clws.magnetic_flux_density(xyz, coordinates='cylindrical')
        np.testing.assert_equal(b_test, b)

    if discretize is not None:
        def test_magnetic_field_3Dcyl(self):
            print("\n === Testing 3D Cyl Mesh === \n")
            n = 50
            ny = 10
            h = 2.
            mesh = discretize.CylindricalMesh(
                [h*np.ones(n), np.ones(ny) * 2 * np.pi / ny, h*np.ones(n)], x0="00C"
            )

            for radius in [0.5, 1, 1.5]:
                self.clws.radius = radius
                self.clws.current = 1./(np.pi * radius**2)  # So that all have moment of 1.0

                fdem_dipole = fdem.MagneticDipoleWholeSpace(frequency=0, quasistatic=True)

                for location in [
                    np.r_[0, 0, 0], np.r_[0, 4, 0], np.r_[4, 4., 0],
                    np.r_[4, 4, 4], np.r_[4, 0, -4]
                ]:
                    self.clws.location = location
                    self.mdws.location = location
                    fdem_dipole.location = location

                    for orientation in ["z"]:
                        self.clws.orientation = orientation
                        self.mdws.orientation = orientation
                        fdem_dipole.orientation = orientation

                        a_clws = np.hstack([
                            self.clws.vector_potential(
                                mesh.gridEx, coordinates="cylindrical"
                            )[:, 0],
                            self.clws.vector_potential(
                                mesh.gridEy, coordinates="cylindrical"
                            )[:, 1],
                            self.clws.vector_potential(
                                mesh.gridEz, coordinates="cylindrical"
                            )[:, 2]
                        ])

                        a_mdws = np.hstack([
                            self.mdws.vector_potential(
                                mesh.gridEx, coordinates="cylindrical"
                            )[:, 0],
                            self.mdws.vector_potential(
                                mesh.gridEy, coordinates="cylindrical"
                            )[:, 1],
                            self.mdws.vector_potential(
                                mesh.gridEz, coordinates="cylindrical"
                            )[:, 2]
                        ])

                        b_clws = mesh.edge_curl * a_clws
                        b_mdws = mesh.edge_curl * a_mdws

                        b_fdem = np.hstack([
                            spatial.cartesian_2_cylindrical(
                                spatial.cylindrical_2_cartesian(mesh.gridFx),
                                fdem_dipole.magnetic_flux_density(
                                    spatial.cylindrical_2_cartesian(mesh.gridFx)
                                )
                            )[:, 0],
                            spatial.cartesian_2_cylindrical(
                                spatial.cylindrical_2_cartesian(mesh.gridFy),
                                fdem_dipole.magnetic_flux_density(
                                    spatial.cylindrical_2_cartesian(mesh.gridFy)
                                )
                            )[:, 1],
                            spatial.cartesian_2_cylindrical(
                                spatial.cylindrical_2_cartesian(mesh.gridFz),
                                fdem_dipole.magnetic_flux_density(
                                    spatial.cylindrical_2_cartesian(mesh.gridFz)
                                )
                            )[:, 2]
                        ])


                        inds = (np.hstack([
                            (np.absolute(mesh.gridFx[:, 0]) > h*4 + location[0]) &
                            (np.absolute(mesh.gridFx[:, 2]) > h*4 + location[2]),
                            (np.absolute(mesh.gridFy[:, 0]) > h*4 + location[0]) &
                            (np.absolute(mesh.gridFy[:, 2]) > h*4 + location[2]),
                            (np.absolute(mesh.gridFz[:, 0]) > h*4 + location[0]) &
                            (np.absolute(mesh.gridFz[:, 2]) > h*4 + location[2])
                        ]))

                        loop_passed = (
                            np.linalg.norm(b_fdem[inds] - b_clws[inds]) <
                            0.5 * TOL * (
                                np.linalg.norm(b_fdem[inds]) +
                                np.linalg.norm(b_clws[inds])
                            )
                        )

                        dipole_passed = (
                            np.linalg.norm(b_fdem[inds] - b_mdws[inds]) <
                            0.5 * TOL * (
                                np.linalg.norm(b_fdem[inds]) +
                                np.linalg.norm(b_mdws[inds])
                            )
                        )

                        print(
                            "Testing r = {}, loc = {}, orientation = {}".format(
                                radius, location, orientation
                            )
                        )
                        print(
                            "  fdem: {:1.4e}, loop: {:1.4e}, dipole: {:1.4e}"
                            " Passed? loop: {}, dipole: {} \n".format(
                                np.linalg.norm(b_fdem[inds]),
                                np.linalg.norm(b_clws[inds]),
                                np.linalg.norm(b_mdws[inds]),
                                loop_passed,
                                dipole_passed
                            )
                        )
                        self.assertTrue(loop_passed)
                        self.assertTrue(dipole_passed)


def Vt_from_ESphere(
    XYZ, loc, sig_s, sig_b, radius, amp
):

    XYZ = np.atleast_2d(XYZ)

    sig_cur = (sig_s - sig_b) / (sig_s + 2 * sig_b)
    r_vec = XYZ - loc
    x = r_vec[:, 0]
    r = np.linalg.norm(r_vec, axis=-1)

    vt = np.zeros_like(r)
    ind0 = r > radius
    vt[ind0] = -amp[0] * x[ind0] * (1. - sig_cur * radius ** 3 / r[ind0] ** 3)
    vt[~ind0] = -amp[0] * x[~ind0] * 3. * sig_b / (sig_s + 2. * sig_b)
    return vt


def Vp_from_ESphere(
    XYZ, loc, sig_s, sig_b, radius, amp
):
    XYZ = np.atleast_2d(XYZ)

    r_vec = XYZ - loc
    x = r_vec[:, 0]

    vp = -amp[0] * x
    return vp


def Vs_from_ESphere(
    XYZ, loc, sig_s, sig_b, radius, amp
):

    vs = Vt_from_ESphere(XYZ, loc, sig_s, sig_b, radius, amp) - Vp_from_ESphere(XYZ, loc, sig_s, sig_b, radius, amp)
    return vs


def Et_from_ESphere(
    XYZ, loc, sig_s, sig_b, radius, amp
):

    XYZ = np.atleast_2d(XYZ)

    sig_cur = (sig_s - sig_b) / (sig_s + 2 * sig_b)
    r_vec = XYZ - loc
    x = r_vec[:, 0]
    y = r_vec[:, 1]
    z = r_vec[:, 2]
    r = np.linalg.norm(r_vec, axis=-1)

    et = np.zeros((*r.shape, 3))
    ind0 = r > radius
    et[ind0, 0] = amp[0] + amp[0] * radius ** 3. / (r[ind0] ** 5.) * sig_cur * (
                2. * x[ind0] ** 2. - y[ind0] ** 2. - z[ind0] ** 2.)
    et[ind0, 1] = amp[0] * radius ** 3. / (r[ind0] ** 5.) * 3. * x[ind0] * y[ind0] * sig_cur
    et[ind0, 2] = amp[0] * radius ** 3. / (r[ind0] ** 5.) * 3. * x[ind0] * z[ind0] * sig_cur
    et[~ind0, 0] = 3. * sig_b / (sig_s + 2. * sig_b) * amp[0]
    return et


def Ep_from_ESphere(
    XYZ, loc, sig_s, sig_b, radius, amp
):
    XYZ = np.atleast_2d(XYZ)

    r_vec = XYZ - loc
    r = np.linalg.norm(r_vec, axis=-1)

    ep = np.zeros((*r.shape, 3))
    ep[..., 0] = amp[0]
    return ep


def Es_from_ESphere(
    XYZ, loc, sig_s, sig_b, radius, amp
):

    es = Et_from_ESphere(XYZ, loc, sig_s, sig_b, radius, amp) - Ep_from_ESphere(XYZ, loc, sig_s, sig_b, radius, amp)
    return es


def Jt_from_ESphere(
    XYZ, loc, sig_s, sig_b, radius, amp
):

    XYZ = np.atleast_2d(XYZ)

    r_vec = XYZ - loc
    r = np.linalg.norm(r_vec, axis=-1)
    sigma = np.full(r.shape, sig_b)
    sigma[r <= radius] = sig_s

    jt = sigma[..., None] * Et_from_ESphere(XYZ, loc, sig_s, sig_b, radius, amp)
    return jt


def Jp_from_ESphere(
    XYZ, loc, sig_s, sig_b, radius, amp
):

    jp = sig_b * Ep_from_ESphere(XYZ, loc, sig_s, sig_b, radius, amp)
    return jp


def Js_from_ESphere(
    XYZ, loc, sig_s, sig_b, radius, amp
):

    js = Jt_from_ESphere(XYZ, loc, sig_s, sig_b, radius, amp) - Jp_from_ESphere(XYZ, loc, sig_s, sig_b, radius, amp)
    return js


def rho_from_ESphere(
    XYZ, dr, loc, sig_s, sig_b, radius, amp
):
    XYZ = np.atleast_2d(XYZ)

    sig_cur = (sig_s - sig_b) / (sig_s + 2 * sig_b)
    r_vec = XYZ - loc
    x = r_vec[:, 0]
    y = r_vec[:, 1]
    r = np.linalg.norm(r_vec, axis=-1)
    if dr is None:
        dr = 0.05 * radius

    ind = (r < radius + 0.5 * dr) & (r > radius - 0.5 * dr)
    rho = np.zeros_like(r)
    rho[ind] = epsilon_0 * 3 * Ep_from_ESphere(XYZ, loc, sig_s, sig_b, radius, amp)[ind, 0] * sig_cur * x[ind] / \
        (np.sqrt(x[ind]**2 + y[ind]**2))
    return rho


class TestElectroStaticSphere:

    def test_defaults(self):
        radius = 1.0
        sigma_sphere = 1.0
        sigma_background = 1.0
        ess = static.ElectrostaticSphere(radius, sigma_sphere, sigma_background)
        assert np.all(ess.primary_field == np.r_[1., 0., 0.])
        assert ess.radius == 1.0
        assert ess.sigma_sphere == 1.0
        assert ess.sigma_background == 1.0
        assert np.all(ess.location == np.r_[0., 0., 0.])

    def test_errors(self):
        ess = static.ElectrostaticSphere(primary_field=None, radius=1.0, sigma_sphere=1.0,
                                         sigma_background=1.0, location=None)
        with pytest.raises(ValueError):
            ess.sigma_sphere = -1
        with pytest.raises(ValueError):
            ess.sigma_background = -1
        with pytest.raises(ValueError):
            ess.radius = -2
        with pytest.raises(ValueError):
            ess.location = [0, 1, 2, 3, 4, 5]
        with pytest.raises(ValueError):
            ess.location = [[0, 0, 1, 4], [0, 1, 0, 3]]
        with pytest.raises(TypeError):
            ess.location = ["string"]
        with pytest.raises(ValueError):
            ess.primary_field = [0, 1, 2, 3, 4, 5]
        with pytest.raises(ValueError):
            ess.primary_field = [[0, 0, 1, 4], [0, 1, 0, 3]]
        with pytest.raises(TypeError):
            ess.primary_field = ["string"]

    def testV(self):
        radius = 1.0
        primary_field = None
        sig_s = 1.0
        sig_b = 1.0
        location = None
        ess = static.ElectrostaticSphere(
            radius=radius,
            primary_field=primary_field,
            sigma_background=sig_b,
            sigma_sphere=sig_s,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        vttest = Vt_from_ESphere(
            xyz, ess.location, ess.sigma_sphere, ess.sigma_background, ess.radius, ess.primary_field
        )
        vptest = Vp_from_ESphere(
            xyz, ess.location, ess.sigma_sphere, ess.sigma_background, ess.radius, ess.primary_field
        )
        vstest = Vs_from_ESphere(
            xyz, ess.location, ess.sigma_sphere, ess.sigma_background, ess.radius, ess.primary_field
        )
        print(
            "\n\nTesting Electric Potential V for Sphere\n"
        )

        vt = ess.potential(xyz, field='total')
        vp = ess.potential(xyz, field='primary')
        vs = ess.potential(xyz, field='secondary')
        np.testing.assert_equal(vttest, vt)
        np.testing.assert_equal(vptest, vp)
        np.testing.assert_equal(vstest, vs)

        vt, vp, vs = ess.potential(xyz, field='all')
        np.testing.assert_equal(vttest, vt)
        np.testing.assert_equal(vptest, vp)
        np.testing.assert_equal(vstest, vs)

    def testE(self):
        radius = 1.0
        primary_field = None
        sig_s = 1.0
        sig_b = 1.0
        location = None
        ess = static.ElectrostaticSphere(
            radius=radius,
            primary_field=primary_field,
            sigma_background=sig_b,
            sigma_sphere=sig_s,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        ettest = Et_from_ESphere(
            xyz, ess.location, ess.sigma_sphere, ess.sigma_background, ess.radius, ess.primary_field
        )
        eptest = Ep_from_ESphere(
            xyz, ess.location, ess.sigma_sphere, ess.sigma_background, ess.radius, ess.primary_field
        )
        estest = Es_from_ESphere(
            xyz, ess.location, ess.sigma_sphere, ess.sigma_background, ess.radius, ess.primary_field
        )
        print(
            "\n\nTesting Electric Potential V for Sphere\n"
        )

        et = ess.electric_field(xyz, field='total')
        ep = ess.electric_field(xyz, field='primary')
        es = ess.electric_field(xyz, field='secondary')
        np.testing.assert_equal(ettest, et)
        np.testing.assert_equal(eptest, ep)
        np.testing.assert_equal(estest, es)

        et, ep, es =ess.electric_field(xyz, field='all')
        np.testing.assert_equal(ettest, et)
        np.testing.assert_equal(eptest, ep)
        np.testing.assert_equal(estest, es)

    def testJ(self):
        radius = 1.0
        primary_field = None
        sig_s = 1.0
        sig_b = 1.0
        location = None
        ess = static.ElectrostaticSphere(
            radius=radius,
            primary_field=primary_field,
            sigma_background=sig_b,
            sigma_sphere=sig_s,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        jttest = Jt_from_ESphere(
            xyz, ess.location, ess.sigma_sphere, ess.sigma_background, ess.radius, ess.primary_field
        )
        jptest = Jp_from_ESphere(
            xyz, ess.location, ess.sigma_sphere, ess.sigma_background, ess.radius, ess.primary_field
        )
        jstest = Js_from_ESphere(
            xyz, ess.location, ess.sigma_sphere, ess.sigma_background, ess.radius, ess.primary_field
        )
        print(
            "\n\nTesting Current Density J for Sphere\n"
        )

        jt = ess.current_density(xyz, field='total')
        jp = ess.current_density(xyz, field='primary')
        js = ess.current_density(xyz, field='secondary')
        np.testing.assert_equal(jttest, jt)
        np.testing.assert_equal(jptest, jp)
        np.testing.assert_equal(jstest, js)

        jt, jp, js = ess.current_density(xyz, field='all')
        np.testing.assert_equal(jttest, jt)
        np.testing.assert_equal(jptest, jp)
        np.testing.assert_equal(jstest, js)

    def test_rho(self):
        radius = 1.0
        primary_field = None
        sig_s = 1.0
        sig_b = 1.0
        location = None
        ess = static.ElectrostaticSphere(
            radius=radius,
            primary_field=primary_field,
            sigma_background=sig_b,
            sigma_sphere=sig_s,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        rho_test = rho_from_ESphere(
            xyz, None, ess.location, ess.sigma_sphere, ess.sigma_background, ess.radius, ess.primary_field
        )

        rho = ess.charge_density(xyz)
        np.testing.assert_equal(rho_test, rho)


def Vt_from_Sphere(
    XYZ, loc, mu_s, mu_b, radius, amp
):

    XYZ = np.atleast_2d(XYZ)

    mu_cur = (mu_s - mu_b) / (mu_s + 2 * mu_b)
    r_vec = XYZ - loc
    x = r_vec[:, 0]
    r = np.linalg.norm(r_vec, axis=-1)

    vt = np.zeros_like(r)
    ind0 = r > radius
    vt[ind0] = -amp[0] * x[ind0] * (1. - mu_cur * radius ** 3 / r[ind0] ** 3)
    vt[~ind0] = -amp[0] * x[~ind0] * 3. * mu_b / (mu_s + 2. * mu_b)
    return vt


def Vp_from_Sphere(
    XYZ, loc, mu_s, mu_b, radius, amp
):
    XYZ = np.atleast_2d(XYZ)

    r_vec = XYZ - loc
    x = r_vec[:, 0]

    vp = -amp[0] * x
    return vp


def Vs_from_Sphere(
    XYZ, loc, mu_s, mu_b, radius, amp
):

    vs = Vt_from_Sphere(XYZ, loc, mu_s, mu_b, radius, amp) - Vp_from_Sphere(XYZ, loc, mu_s, mu_b, radius, amp)
    return vs


def Ht_from_Sphere(
    XYZ, loc, mu_s, mu_b, radius, amp
):

    XYZ = np.atleast_2d(XYZ)

    mu_cur = (mu_s - mu_b) / (mu_s + 2 * mu_b)
    r_vec = XYZ - loc
    x = r_vec[:, 0]
    y = r_vec[:, 1]
    z = r_vec[:, 2]
    r = np.linalg.norm(r_vec, axis=-1)

    ht = np.zeros((*r.shape, 3))
    ind0 = r > radius
    ht[ind0, 0] = amp[0] + amp[0] * radius ** 3. / (r[ind0] ** 5.) * mu_cur * (
                2. * x[ind0] ** 2. - y[ind0] ** 2. - z[ind0] ** 2.)
    ht[ind0, 1] = amp[0] * radius ** 3. / (r[ind0] ** 5.) * 3. * x[ind0] * y[ind0] * mu_cur
    ht[ind0, 2] = amp[0] * radius ** 3. / (r[ind0] ** 5.) * 3. * x[ind0] * z[ind0] * mu_cur
    ht[~ind0, 0] = 3. * mu_b / (mu_s + 2. * mu_b) * amp[0]
    return ht


def Hp_from_Sphere(
    XYZ, loc, mu_s, mu_b, radius, amp
):
    XYZ = np.atleast_2d(XYZ)

    r_vec = XYZ - loc
    x = r_vec[:, 0]

    hp = np.zeros((*x.shape, 3))
    hp[..., 0] = amp[0]
    return hp


def Hs_from_Sphere(
    XYZ, loc, mu_s, mu_b, radius, amp
):

    hs = Ht_from_Sphere(XYZ, loc, mu_s, mu_b, radius, amp) - Hp_from_Sphere(XYZ, loc, mu_s, mu_b, radius, amp)
    return hs


def Bt_from_Sphere(
    XYZ, loc, mu_s, mu_b, radius, amp
):

    XYZ = np.atleast_2d(XYZ)

    r_vec = XYZ - loc
    r = np.linalg.norm(r_vec, axis=-1)
    mu = np.full(r.shape, mu_b)
    mu[r <= radius] = mu_s

    bt = mu[..., None] * Et_from_ESphere(XYZ, loc, mu_s, mu_b, radius, amp)
    return bt


def Bp_from_Sphere(
    XYZ, loc, mu_s, mu_b, radius, amp
):

    bp = mu_b * Ep_from_ESphere(XYZ, loc, mu_s, mu_b, radius, amp)
    return bp


def Bs_from_Sphere(
    XYZ, loc, mu_s, mu_b, radius, amp
):

    bs = Bt_from_Sphere(XYZ, loc, mu_s, mu_b, radius, amp) - Bp_from_Sphere(XYZ, loc, mu_s, mu_b, radius, amp)
    return bs


class TestMagnetoStaticSphere:

    def test_defaults(self):
        radius = 1.0
        mu_sphere = 1.0
        mu_background = 1.0
        mss = static.MagnetostaticSphere(radius, mu_sphere, mu_background)
        assert np.all(mss.primary_field == np.r_[1., 0., 0.])
        assert mss.radius == 1.0
        assert mss.mu_sphere == 1.0
        assert mss.mu_background == 1.0
        assert np.all(mss.location == np.r_[0., 0., 0.])

    def test_errors(self):
        mss = static.MagnetostaticSphere(primary_field=None, radius=1.0, mu_sphere=1.0, mu_background=1.0,
                                         location=None)
        with pytest.raises(ValueError):
            mss.mu_sphere = -1
        with pytest.raises(ValueError):
            mss.mu_background = -1
        with pytest.raises(ValueError):
            mss.radius = -2
        with pytest.raises(ValueError):
            mss.location = [0, 1, 2, 3, 4, 5]
        with pytest.raises(ValueError):
            mss.location = [[0, 0, 1, 4], [0, 1, 0, 3]]
        with pytest.raises(TypeError):
            mss.location = ["string"]
        with pytest.raises(ValueError):
            mss.primary_field = [0, 1, 2, 3, 4, 5]
        with pytest.raises(ValueError):
            mss.primary_field = [[0, 0, 1, 4], [0, 1, 0, 3]]
        with pytest.raises(TypeError):
            mss.primary_field = ["string"]

    def testV(self):
        radius = 1.0
        primary_field = None
        mu_s = 1.0
        mu_b = 1.0
        location = None
        mss = static.MagnetostaticSphere(
            radius=radius,
            primary_field=primary_field,
            mu_background=mu_b,
            mu_sphere=mu_s,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        vttest = Vt_from_Sphere(
            xyz, mss.location, mss.mu_sphere, mss.mu_background, mss.radius, mss.primary_field
        )
        vptest = Vp_from_Sphere(
            xyz, mss.location, mss.mu_sphere, mss.mu_background, mss.radius, mss.primary_field
        )
        vstest = Vs_from_Sphere(
            xyz, mss.location, mss.mu_sphere, mss.mu_background, mss.radius, mss.primary_field
        )
        print(
            "\n\nTesting Magnetic Potential V for Sphere\n"
        )

        vt = mss.potential(xyz, field='total')
        vp = mss.potential(xyz, field='primary')
        vs = mss.potential(xyz, field='secondary')
        np.testing.assert_equal(vttest, vt)
        np.testing.assert_equal(vptest, vp)
        np.testing.assert_equal(vstest, vs)

        vt, vp, vs = mss.potential(xyz, field='all')
        np.testing.assert_equal(vttest, vt)
        np.testing.assert_equal(vptest, vp)
        np.testing.assert_equal(vstest, vs)

    def testH(self):
        radius = 1.0
        primary_field = None
        mu_s = 1.0
        mu_b = 1.0
        location = None
        mss = static.MagnetostaticSphere(
            radius=radius,
            primary_field=primary_field,
            mu_background=mu_b,
            mu_sphere=mu_s,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        httest = Ht_from_Sphere(
            xyz, mss.location, mss.mu_sphere, mss.mu_background, mss.radius, mss.primary_field
        )
        hptest = Hp_from_Sphere(
            xyz, mss.location, mss.mu_sphere, mss.mu_background, mss.radius, mss.primary_field
        )
        hstest = Hs_from_Sphere(
            xyz, mss.location, mss.mu_sphere, mss.mu_background, mss.radius, mss.primary_field
        )
        print(
            "\n\nTesting Magnetic Field H for Sphere\n"
        )

        ht = mss.magnetic_field(xyz, field='total')
        hp = mss.magnetic_field(xyz, field='primary')
        hs = mss.magnetic_field(xyz, field='secondary')
        np.testing.assert_equal(httest, ht)
        np.testing.assert_equal(hptest, hp)
        np.testing.assert_equal(hstest, hs)

        ht, hp, hs = mss.magnetic_field(xyz, field='all')
        np.testing.assert_equal(httest, ht)
        np.testing.assert_equal(hptest, hp)
        np.testing.assert_equal(hstest, hs)

    def testB(self):
        radius = 1.0
        primary_field = None
        mu_s = 1.0
        mu_b = 1.0
        location = None
        mss = static.MagnetostaticSphere(
            radius=radius,
            primary_field=primary_field,
            mu_background=mu_b,
            mu_sphere=mu_s,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        btest = Bt_from_Sphere(
            xyz, mss.location, mss.mu_sphere, mss.mu_background, mss.radius, mss.primary_field
        )
        bptest = Bp_from_Sphere(
            xyz, mss.location, mss.mu_sphere, mss.mu_background, mss.radius, mss.primary_field
        )
        bstest = Bs_from_Sphere(
            xyz, mss.location, mss.mu_sphere, mss.mu_background, mss.radius, mss.primary_field
        )
        print(
            "\n\nTesting Magnetic Flux Density B for Sphere\n"
        )

        bt = mss.magnetic_flux_density(xyz, field='total')
        bp = mss.magnetic_flux_density(xyz, field='primary')
        bs = mss.magnetic_flux_density(xyz, field='secondary')
        np.testing.assert_equal(btest, bt)
        np.testing.assert_equal(bptest, bp)
        np.testing.assert_equal(bstest, bs)

        bt, bp, bs = mss.magnetic_flux_density(xyz, field='all')
        np.testing.assert_equal(btest, bt)
        np.testing.assert_equal(bptest, bp)
        np.testing.assert_equal(bstest, bs)


def V_from_PointCurrentW(
    XYZ, loc, rho, cur
):

    XYZ = np.atleast_2d(XYZ)

    r_vec = XYZ - loc
    r = np.linalg.norm(r_vec, axis=-1)

    v = rho * cur / (4 * np.pi * r)
    return v


def J_from_PointCurrentW(
    XYZ, loc, rho, cur
):

    j = E_from_PointCurrentW(XYZ, loc, rho, cur) / rho
    return j


def E_from_PointCurrentW(
    XYZ, loc, rho, cur
):

    XYZ = np.atleast_2d(XYZ)

    r_vec = XYZ - loc
    r = np.linalg.norm(r_vec, axis=-1)

    e = rho * cur * r_vec / (4 * np.pi * r[..., None] ** 3)
    return e


class TestPointCurrentWholeSpace:

    def test_defaults(self):
        rho = 1.0
        pcws = static.PointCurrentWholeSpace(rho)
        assert pcws.rho == 1.0
        assert pcws.current == 1.0
        assert np.all(pcws.location == np.r_[0., 0., 0.])

    def test_error(self):
        pcws = static.PointCurrentWholeSpace(rho=1.0, current=1.0, location=None)

        with pytest.raises(TypeError):
            pcws.rho = "string"
        with pytest.raises(ValueError):
            pcws.rho = -1
        with pytest.raises(TypeError):
            pcws.current = "string"
        with pytest.raises(ValueError):
            pcws.location = [0, 1, 2, 3]
        with pytest.raises(ValueError):
            pcws.location = [[0, 0], [0, 1]]
        with pytest.raises(TypeError):
            pcws.location = ["string"]

    def test_potential(self):
        rho = 1.0
        current = 1.0
        location = None
        pcws = static.PointCurrentWholeSpace(
            current=current,
            rho=rho,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        vtest = V_from_PointCurrentW(
            xyz, pcws.location, pcws.rho, pcws.current
        )
        print(
            "\n\nTesting Electric Potential V for Point Current\n"
        )

        v = pcws.potential(xyz)
        np.testing.assert_equal(vtest, v)

    def test_current_density(self):
        rho = 1.0
        current = 1.0
        location = None
        pcws = static.PointCurrentWholeSpace(
            current=current,
            rho=rho,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        jtest = J_from_PointCurrentW(
            xyz, pcws.location, pcws.rho, pcws.current
        )
        print(
            "\n\nTesting Current Density J for Point Current\n"
        )

        j = pcws.current_density(xyz)
        np.testing.assert_equal(jtest, j)

    def test_electric_field(self):
        rho = 1.0
        current = 1.0
        location = None
        pcws = static.PointCurrentWholeSpace(
            current=current,
            rho=rho,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        etest = E_from_PointCurrentW(
            xyz, pcws.location, pcws.rho, pcws.current
        )
        print(
            "\n\nTesting Electric Field E for Point Current\n"
        )

        e = pcws.electric_field(xyz)
        np.testing.assert_equal(etest, e)


def V_from_PointCurrentH(
    XYZ, loc, rho, cur
):

    XYZ = np.atleast_2d(XYZ)

    r_vec = XYZ - loc
    r = np.linalg.norm(r_vec, axis=-1)

    v = rho * cur / (2 * np.pi * r)
    return v


def E_from_PointCurrentH(
    XYZ, loc, rho, cur
):

    XYZ = np.atleast_2d(XYZ)

    r_vec = XYZ - loc
    r = np.linalg.norm(r_vec, axis=-1)

    e = rho * cur * r_vec / (2 * np.pi * r[..., None] ** 3)
    return e


def J_from_PointCurrentH(
    XYZ, loc, rho, cur
):

    j = E_from_PointCurrentH(XYZ, loc, rho, cur) / rho
    return j


class TestPointCurrentHalfSpace:

    def test_defaults(self):
        rho = 1.0
        pchs = static.PointCurrentHalfSpace(rho)
        assert pchs.rho == 1.0
        assert pchs.current == 1.0
        assert np.all(pchs.location == np.r_[0., 0., 0.])

    def test_error(self):
        pchs = static.PointCurrentHalfSpace(rho=1.0, current=1.0, location=None)

        with pytest.raises(TypeError):
            pchs.rho = "box"
        with pytest.raises(ValueError):
            pchs.rho = -2
        with pytest.raises(TypeError):
            pchs.current = "box"
        with pytest.raises(ValueError):
            pchs.location = [0, 1, 2, 3]
        with pytest.raises(ValueError):
            pchs.location = [[0, 0], [0, 1]]
        with pytest.raises(TypeError):
            pchs.location = ["string"]
        with pytest.raises(ValueError):
            pchs.location = [0, 0, 1]

    def test_whole_space_objects(self):
        pchs = static.PointCurrentHalfSpace(rho=1.0, current=1.0, location=None)

        assert pchs._primary.rho == 1.0
        assert pchs._image.rho == 1.0
        assert pchs._primary.current == 1.0
        assert pchs._image.current == 1.0
        assert np.all(pchs._primary.location == np.r_[0., 0., 0.])
        assert np.all(pchs._image.location == np.r_[0., 0., 0.])

        pchs.rho = 2.0
        pchs.current = 2.0
        pchs.location = np.r_[1, 1, -1]

        assert pchs._primary.rho == 2.0
        assert pchs._image.rho == 2.0
        assert pchs._primary.current == 2.0
        assert pchs._image.current == 2.0
        assert np.all(pchs._primary.location == np.r_[1., 1., -1.])
        assert np.all(pchs._image.location == np.r_[1., 1., 1.])

    def test_potential(self):
        rho = 1.0
        current = 1.0
        location = np.r_[0., 0., 0.]
        pchs = static.PointCurrentHalfSpace(
            current=current,
            rho=rho,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 0., 50)
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        vtest = V_from_PointCurrentH(
            xyz, pchs.location, pchs.rho, pchs.current
        )
        print(
            "\n\nTesting Electric Potential V for Point Current in Halfspace\n"
        )

        v = pchs.potential(xyz)
        np.testing.assert_equal(vtest, v)

        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        with pytest.raises(ValueError):
            pchs.potential(xyz)

    def test_electric_field(self):
        rho = 1.0
        current = 1.0
        location = None
        pchs = static.PointCurrentHalfSpace(
            current=current,
            rho=rho,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 0., 50)
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        etest = E_from_PointCurrentH(
            xyz, pchs.location, pchs.rho, pchs.current
        )
        print(
            "\n\nTesting Electric Field E for Point Current in Halfspace\n"
        )

        e = pchs.electric_field(xyz)
        np.testing.assert_equal(etest, e)

        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        with pytest.raises(ValueError):
            pchs.electric_field(xyz)

    def test_current_density(self):
        rho = 1.0
        current = 1.0
        location = None
        pchs = static.PointCurrentHalfSpace(
            current=current,
            rho=rho,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 0., 50)
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        jtest = J_from_PointCurrentH(
            xyz, pchs.location, pchs.rho, pchs.current
        )
        print(
            "\n\nTesting Current Density J for Point Current in Halfspace\n"
        )

        j = pchs.current_density(xyz)
        np.testing.assert_equal(jtest, j)

        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        with pytest.raises(ValueError):
            pchs.current_density(xyz)


def V_from_Dipole1(
        XYZ_M, XYZ_N, rho, cur, loc_a, loc_b
):
    XYZ_M = np.atleast_2d(XYZ_M)

    r_vec1 = XYZ_M - loc_a
    r_vec2 = XYZ_M - loc_b
    r1 = np.linalg.norm(r_vec1, axis=-1)
    r2 = np.linalg.norm(r_vec2, axis=-1)

    v = rho * cur / (2 * np.pi * r1) - rho * cur / (2 * np.pi * r2)
    return v


def V_from_Dipole2(
        XYZ_M, XYZ_N, rho, cur, loc_a, loc_b
):
    XYZ_M = np.atleast_2d(XYZ_M)
    XYZ_N = np.atleast_2d(XYZ_N)

    r_vec1 = XYZ_M - loc_a
    r_vec2 = XYZ_M - loc_b
    r1 = np.linalg.norm(r_vec1, axis=-1)
    r2 = np.linalg.norm(r_vec2, axis=-1)

    r_vec3 = XYZ_N - loc_a
    r_vec4 = XYZ_N - loc_b
    r3 = np.linalg.norm(r_vec3, axis=-1)
    r4 = np.linalg.norm(r_vec4, axis=-1)

    vm = rho * cur / (2 * np.pi * r1) - rho * cur / (2 * np.pi * r2)
    vn = rho * cur / (2 * np.pi * r3) - rho * cur / (2 * np.pi * r4)
    v = vm - vn
    return v


def E_from_Dipole1(
        XYZ_M, XYZ_N, rho, cur, loc_a, loc_b
):
    XYZ_M = np.atleast_2d(XYZ_M)

    r_vec1 = XYZ_M - loc_a
    r_vec2 = XYZ_M - loc_b
    r1 = np.linalg.norm(r_vec1, axis=-1)
    r2 = np.linalg.norm(r_vec2, axis=-1)

    e = rho * cur * r_vec1 / (2 * np.pi * r1[..., None] ** 3) - rho * cur * r_vec2 / (2 * np.pi * r2[..., None] ** 3)
    return e


def E_from_Dipole2(
        XYZ_M, XYZ_N, rho, cur, loc_a, loc_b
):
    XYZ_M = np.atleast_2d(XYZ_M)
    XYZ_N = np.atleast_2d(XYZ_N)

    r_vec1 = XYZ_M - loc_a
    r_vec2 = XYZ_M - loc_b
    r1 = np.linalg.norm(r_vec1, axis=-1)
    r2 = np.linalg.norm(r_vec2, axis=-1)

    r_vec3 = XYZ_N - loc_a
    r_vec4 = XYZ_N - loc_b
    r3 = np.linalg.norm(r_vec3, axis=-1)
    r4 = np.linalg.norm(r_vec4, axis=-1)

    em = rho * cur * r_vec1 / (2 * np.pi * r1[..., None] ** 3) - rho * cur * r_vec2 / (2 * np.pi * r2[..., None] ** 3)
    en = rho * cur * r_vec3 / (2 * np.pi * r3[..., None] ** 3) - rho * cur * r_vec4 / (2 * np.pi * r4[..., None] ** 3)
    e = em - en
    return e


def J_from_Dipole1(
        XYZ_M, XYZ_N, rho, cur, loc_a, loc_b
):
    j = E_from_Dipole1(XYZ_M, None, rho, cur, loc_a, loc_b) / rho
    return j


def J_from_Dipole2(
        XYZ_M, XYZ_N, rho, cur, loc_a, loc_b
):
    j = E_from_Dipole2(XYZ_M, XYZ_N, rho, cur, loc_a, loc_b) / rho
    return j


class TestDipoleHalfSpace:

    def test_defaults(self):
        rho = 1.0
        dhs = static.DipoleHalfSpace(rho)
        assert dhs.rho == 1.0
        assert dhs.current == 1.0
        assert np.all(dhs.location_a == np.r_[-1, 0, 0])
        assert np.all(dhs.location_b == np.r_[1, 0, 0])

    def test_error(self):
        dhs = static.DipoleHalfSpace(rho=1.0, current=1.0, location_a=np.r_[-1, 0, 0], location_b=np.r_[1, 0, 0])

        with pytest.raises(TypeError):
            dhs.rho = "box"
        with pytest.raises(ValueError):
            dhs.rho = -2
        with pytest.raises(TypeError):
            dhs.current = "box"
        with pytest.raises(TypeError):
            dhs.location_a = ["string"]
        with pytest.raises(ValueError):
            dhs.location_a = [1, 1, -1, -1]
        with pytest.raises(ValueError):
            dhs.location_a = [0, 0, 1]
        with pytest.raises(TypeError):
            dhs.location_b = ["string"]
        with pytest.raises(ValueError):
            dhs.location_b = [1, 1, -1, -1]
        with pytest.raises(ValueError):
            dhs.location_b = [0, 0, 1]

    def test_half_space_objects(self):
        dhs = static.DipoleHalfSpace(rho=1.0, current=1.0, location_a=np.r_[-1, 0, 0], location_b=np.r_[1, 0, 0])

        assert dhs._a.rho == 1.0
        assert dhs._b.rho == 1.0
        assert dhs._a.current == 1.0
        assert dhs._b.current == 1.0
        assert np.all(dhs._a.location == np.r_[-1, 0, 0])
        assert np.all(dhs._b.location == np.r_[1, 0, 0])

        dhs.rho = 2.0
        dhs.current = 2.0
        dhs.location_a = np.r_[1, 0, 0]
        dhs.location_b = np.r_[-1, 0, 0]

        assert dhs._a.rho == 2.0
        assert dhs._b.rho == 2.0
        assert dhs._a.current == 2.0
        assert dhs._b.current == 2.0
        assert np.all(dhs._a.location == np.r_[1, 0, 0])
        assert np.all(dhs._b.location == np.r_[-1, 0, 0])

    def test_potential(self):
        dhs = static.DipoleHalfSpace(rho=1.0, current=1.0, location_a=np.r_[-1, 0, 0], location_b=np.r_[1, 0, 0])
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 0., 50)
        xyz1 = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        x = np.linspace(-30., 20., 50)
        y = np.linspace(-20., 30., 50)
        z = np.linspace(-30., 0., 50)
        xyz2 = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        vtest1 = V_from_Dipole1(
            xyz1, None, dhs.rho, dhs.current, dhs.location_a, dhs.location_b
        )

        vtest2 = V_from_Dipole2(
            xyz1, xyz2, dhs.rho, dhs.current, dhs.location_a, dhs.location_b
        )

        print(
            "\n\nTesting Electric Potential V for Dipole in Halfspace\n"
        )

        v1 = dhs.potential(xyz1)
        v2 = dhs.potential(xyz1, xyz2)
        np.testing.assert_equal(vtest1, v1)
        np.testing.assert_equal(vtest2, v2)

        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz3 = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)
        xyz4 = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        with pytest.raises(ValueError):
            dhs.potential(xyz3)
        with pytest.raises(ValueError):
            dhs.potential(xyz1, xyz4)

    def test_electric_field(self):
        dhs = static.DipoleHalfSpace(rho=1.0, current=1.0, location_a=np.r_[-1, 0, 0], location_b=np.r_[1, 0, 0])
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 0., 50)
        xyz1 = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        x = np.linspace(-30., 20., 50)
        y = np.linspace(-20., 30., 50)
        z = np.linspace(-30., 0., 50)
        xyz2 = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        etest1 = E_from_Dipole1(
            xyz1, None, dhs.rho, dhs.current, dhs.location_a, dhs.location_b
        )

        etest2 = E_from_Dipole2(
            xyz1, xyz2, dhs.rho, dhs.current, dhs.location_a, dhs.location_b
        )

        print(
            "\n\nTesting Electric Field V for Dipole in Halfspace\n"
        )

        e1 = dhs.electric_field(xyz1)
        e2 = dhs.electric_field(xyz1, xyz2)
        np.testing.assert_equal(etest1, e1)
        np.testing.assert_equal(etest2, e2)

        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz3 = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)
        xyz4 = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        with pytest.raises(ValueError):
            dhs.electric_field(xyz3)
        with pytest.raises(ValueError):
            dhs.electric_field(xyz1, xyz4)

    def test_current_density(self):
        dhs = static.DipoleHalfSpace(rho=1.0, current=1.0, location_a=np.r_[-1, 0, 0], location_b=np.r_[1, 0, 0])
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 0., 50)
        xyz1 = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        x = np.linspace(-30., 20., 50)
        y = np.linspace(-20., 30., 50)
        z = np.linspace(-30., 0., 50)
        xyz2 = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        jtest1 = J_from_Dipole1(
            xyz1, None, dhs.rho, dhs.current, dhs.location_a, dhs.location_b
        )

        jtest2 = J_from_Dipole2(
            xyz1, xyz2, dhs.rho, dhs.current, dhs.location_a, dhs.location_b
        )

        print(
            "\n\nTesting Current Density V for Dipole in Halfspace\n"
        )

        j1 = dhs.current_density(xyz1)
        j2 = dhs.current_density(xyz1, xyz2)
        np.testing.assert_equal(jtest1, j1)
        np.testing.assert_equal(jtest2, j2)

        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz3 = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)
        xyz4 = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)

        with pytest.raises(ValueError):
            dhs.current_density(xyz3)
        with pytest.raises(ValueError):
            dhs.current_density(xyz1, xyz4)

