import pytest
import unittest
import numpy as np
from scipy.constants import mu_0, epsilon_0
import discretize

from geoana.em import static, fdem
from geoana import spatial

TOL = 0.1


class TestEM_Static(unittest.TestCase):

    def setUp(self):
        self.mdws = static.MagneticDipoleWholeSpace()
        self.clws = static.CircularLoopWholeSpace()

    def test_defaults(self):
        self.assertTrue(self.mdws.sigma == 1.0)
        self.assertTrue(self.clws.sigma == 1.0)

        self.assertTrue(self.mdws.mu == mu_0)
        self.assertTrue(self.clws.mu == mu_0)

        self.assertTrue(self.mdws.epsilon == epsilon_0)
        self.assertTrue(self.clws.epsilon == epsilon_0)

        self.assertTrue(np.all(self.mdws.orientation == np.r_[1., 0., 0.]))
        self.assertTrue(np.all(self.clws.orientation == np.r_[1., 0., 0.]))

        self.assertTrue(self.mdws.moment == 1.0)
        self.assertTrue(self.clws.current == 1.0)
        self.assertTrue(self.clws.radius == np.sqrt(1/np.pi))

    def test_vector_potential(self):
        n = 50
        mesh = discretize.TensorMesh(
            [np.ones(n), np.ones(n), np.ones(n)], x0="CCC"
        )

        # radius = 1.0
        # self.clws.radius = radius
        # self.clws.current = 1./(np.pi * radius**2)

        for orientation in ["x", "y", "z"]:
            self.clws.orientation = orientation
            self.mdws.orientation = orientation

            inds = (
                (np.absolute(mesh.cell_centers[:, 0]) > 5) &
                (np.absolute(mesh.cell_centers[:, 1]) > 5) &
                (np.absolute(mesh.cell_centers[:, 2]) > 5)
            )

            a_clws = self.clws.vector_potential(mesh.cell_centers)[inds]
            a_mdws = self.mdws.vector_potential(mesh.cell_centers)[inds]

            self.assertTrue(isinstance(a_clws, np.ndarray))
            self.assertTrue(isinstance(a_mdws, np.ndarray))

            self.assertTrue(
                np.linalg.norm(a_clws - a_mdws) <
                0.5 * TOL * (np.linalg.norm(a_clws) + np.linalg.norm(a_mdws))
            )

    def test_magnetic_field_tensor(self):
        print("\n === Testing Tensor Mesh === \n")
        n = 30
        h = 2.
        mesh = discretize.TensorMesh(
            [h*np.ones(n), h*np.ones(n), h*np.ones(n)], x0="CCC"
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

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

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
    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

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

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

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
    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

    r_vec = XYZ - loc
    x = r_vec[:, 0]

    ep = np.zeros((*x.shape, 3))
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

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

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
        xyz = discretize.utils.ndgrid([x, y, z])

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
        xyz = discretize.utils.ndgrid([x, y, z])

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
        xyz = discretize.utils.ndgrid([x, y, z])

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


def Vt_from_Sphere(
    XYZ, loc, mu_s, mu_b, radius, amp
):

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

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
    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

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

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

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
    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

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

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

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
        xyz = discretize.utils.ndgrid([x, y, z])

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
        xyz = discretize.utils.ndgrid([x, y, z])

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
        xyz = discretize.utils.ndgrid([x, y, z])

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


def V_from_PointCurrentW(
    XYZ, loc, rho, cur
):

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

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

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

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
        xyz = discretize.utils.ndgrid([x, y, z])

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
        xyz = discretize.utils.ndgrid([x, y, z])

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
        xyz = discretize.utils.ndgrid([x, y, z])

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

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

    r_vec = XYZ - loc
    r = np.linalg.norm(r_vec, axis=-1)

    v = rho * cur / (2 * np.pi * r)
    return v


def J_from_PointCurrentH(
    XYZ, loc, rho, cur
):

    j = E_from_PointCurrentH(XYZ, loc, rho, cur) / rho
    return j


def E_from_PointCurrentH(
    XYZ, loc, rho, cur
):

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

    r_vec = XYZ - loc
    r = np.linalg.norm(r_vec, axis=-1)

    e = rho * cur * r_vec / (2 * np.pi * r[..., None] ** 3)
    return e


class TestPointCurrentHalfSpace:

    def test_defaults(self):
        rho = 1.0
        pcws = static.PointCurrentHalfSpace(rho)
        assert pcws.rho == 1.0
        assert pcws.current == 1.0
        assert np.all(pcws.location == np.r_[0., 0., 0.])

    def test_error(self):
        pcws = static.PointCurrentHalfSpace(rho=1.0, current=1.0, location=None)

        with pytest.raises(TypeError):
            pcws.rho = "box"
        with pytest.raises(ValueError):
            pcws.rho = -2
        with pytest.raises(TypeError):
            pcws.current = "box"
        with pytest.raises(ValueError):
            pcws.location = [0, 1, 2, 3]
        with pytest.raises(ValueError):
            pcws.location = [[0, 0], [0, 1]]
        with pytest.raises(TypeError):
            pcws.location = ["string"]
        with pytest.raises(ValueError):
            pcws.location = [0, 0, 1]

    def test_potential(self):
        rho = 1.0
        current = 1.0
        location = None
        pcws = static.PointCurrentHalfSpace(
            current=current,
            rho=rho,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])

        vtest = V_from_PointCurrentH(
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
        pcws = static.PointCurrentHalfSpace(
            current=current,
            rho=rho,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])

        jtest = J_from_PointCurrentH(
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
        pcws = static.PointCurrentHalfSpace(
            current=current,
            rho=rho,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])

        etest = E_from_PointCurrentH(
            xyz, pcws.location, pcws.rho, pcws.current
        )
        print(
            "\n\nTesting Electric Field E for Point Current\n"
        )

        e = pcws.electric_field(xyz)
        np.testing.assert_equal(etest, e)

