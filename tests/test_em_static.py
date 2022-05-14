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


class Test_StaticSphere(unittest.TestCase):

    def testV(self):
        x, y, z = np.mgrid[-10:10:5j, -10:10:5j, -10:10:5j]
        sphere = static.ElectrostaticSphere(3.4, 1E-1, 1E-4, 2.0)
        np.testing.assert_equal(sphere.location, np.r_[0, 0, 0])
        sphere.location = [0.1, 0.3, 0.5]
        XYZ = np.stack([x, y, z], axis=-1)

        Vt1 = sphere.potential((x, y, z), field='total')
        Vt2 = sphere.potential(XYZ, field='total')
        np.testing.assert_equal(Vt1, Vt2)

        with self.assertRaises(ValueError):
            Vt = sphere.potential((x[0:3], y, z))
        with self.assertRaises(TypeError):
            Vt = sphere.potential("xyzd")

        Vt1 = sphere.potential((x, y, z), field='total')
        Vp1 = sphere.potential((x, y, z), field='primary')
        Vs1 = sphere.potential((x, y, z), field='secondary')
        Vt2, Vp2, Vs2 = sphere.potential((x, y, z), field='all')
        np.testing.assert_equal(Vt1, Vt2)
        np.testing.assert_equal(Vp1, Vp2)
        np.testing.assert_equal(Vs1, Vs2)

    def testE(self):
        x, y, z = np.mgrid[-10:10:5j, -10:10:5j, -10:10:5j]
        sphere = static.ElectrostaticSphere(3.4, 1E-1, 1E-4, 2.0, [0.1, 0.3, 0.5])

        Vt1 = sphere.electric_field((x, y, z), field='total')
        Vp1 = sphere.electric_field((x, y, z), field='primary')
        Vs1 = sphere.electric_field((x, y, z), field='secondary')
        Vt2, Vp2, Vs2 = sphere.electric_field((x, y, z), field='all')
        np.testing.assert_equal(Vt1, Vt2)
        np.testing.assert_equal(Vp1, Vp2)
        np.testing.assert_equal(Vs1, Vs2)

    def testJ(self):
        x, y, z = np.mgrid[-10:10:5j, -10:10:5j, -10:10:5j]
        sphere = static.ElectrostaticSphere(3.4, 1E-1, 1E-4, 2.0, [0.1, 0.3, 0.5])

        Vt1 = sphere.current_density((x, y, z), field='total')
        Vp1 = sphere.current_density((x, y, z), field='primary')
        Vs1 = sphere.current_density((x, y, z), field='secondary')
        Vt2, Vp2, Vs2 = sphere.current_density((x, y, z), field='all')
        np.testing.assert_equal(Vt1, Vt2)
        np.testing.assert_equal(Vp1, Vp2)
        np.testing.assert_equal(Vs1, Vs2)

    def testQ(self):
        x, y, z = np.mgrid[-10:10:51j, -10:10:51j, -10:10:51j]
        sphere = static.ElectrostaticSphere(3.4, 1E-1, 1E-4, 2.0, [0.1, 0.3, 0.5])
        q = sphere.charge_density((x, y, z))
        print(np.sum(q))

    def test_errors(self):
        sphere = static.ElectrostaticSphere(3.4, 1E-1, 1E-4, 2.0, [0.1, 0.3, 0.5])
        with self.assertRaises(ValueError):
            sphere.location = [[0, 0, 1],[0, 1, 0]]
        with self.assertRaises(ValueError):
            sphere.location = [0, 1, 2, 3]
        with self.assertRaises(ValueError):
            sphere.radius = -1
        with self.assertRaises(ValueError):
            sphere.sigma_sphere = -1
        with self.assertRaises(ValueError):
            sphere.sigma_background = -1

if __name__ == '__main__':
    unittest.main()


def V_from_Sphere(
    XYZ, loc, mu_s, mu_b, radius, amp
):

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

    mu_cur = (mu_s - mu_b) / (mu_s + 2 * mu_b)
    r_vec = XYZ - loc
    x = r_vec[:, 0]
    r = np.linalg.norm(r_vec, axis=-1)

    v = np.zeros_like(r)
    ind0 = r > radius
    v[ind0] = -amp * x[ind0] * (1. - mu_cur * radius ** 3. / r[ind0] ** 3.)
    v[~ind0] = -amp * x[~ind0] * 3. * mu_b / (mu_s + 2. * mu_b)
    return v

def H_from_Sphere(
    XYZ, loc, mu_s, mu_b, radius, amp
):

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

    mu_cur = (mu_s - mu_b) / (mu_s + 2 * mu_b)
    r_vec = XYZ - loc
    x = r_vec[:, 0]
    y = r_vec[:, 1]
    z = r_vec[:, 2]
    r = np.linalg.norm(r_vec, axis=-1)

    h = np.zeros((*r.shape, 3))
    ind0 = r > radius
    h[ind0, 0] = amp * radius ** 3. * mu_cur * \
        (2. * x[ind0] ** 2. - y[ind0] ** 2. - z[ind0] ** 2.) / (r[ind0] ** 5.)
    h[ind0, 1] = amp * radius ** 3. * mu_cur * 3. * x[ind0] * y[ind0] / (r[ind0] ** 5.)
    h[ind0, 2] = amp * radius ** 3. * mu_cur * 3. * x[ind0] * z[ind0] / (r[ind0] ** 5.)
    h[~ind0] = 3. * mu_b / (mu_s + 2. * mu_b) * amp
    return h

def B_from_Sphere(
    XYZ, loc, mu_s, mu_b, radius, amp
):

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

    mu_cur = (mu_s - mu_b) / (mu_s + 2 * mu_b)
    r_vec = XYZ - loc
    x = r_vec[:, 0]
    y = r_vec[:, 1]
    z = r_vec[:, 2]
    r = np.linalg.norm(r_vec, axis=-1)

    h = np.zeros((*r.shape, 3))
    ind0 = r > radius
    h[ind0, 0] = amp * radius ** 3. * mu_cur * \
        (2. * x[ind0] ** 2. - y[ind0] ** 2. - z[ind0] ** 2.) / (r[ind0] ** 5.)
    h[ind0, 1] = amp * radius ** 3. * mu_cur * 3. * x[ind0] * y[ind0] / (r[ind0] ** 5.)
    h[ind0, 2] = amp * radius ** 3. * mu_cur * 3. * x[ind0] * z[ind0] / (r[ind0] ** 5.)
    h[~ind0] = 3. * mu_b / (mu_s + 2. * mu_b) * amp

    b = h * mu_0
    return b


class TestMagnetoStaticSphere:

    def test_defaults(self):
        radius = 1.0
        mu_sphere = 1.0
        mu_background = 1.0
        mss = static.MagnetostaticSphere(radius, mu_sphere, mu_background)
        assert mss.amplitude == 1.0
        assert mss.radius == 1.0
        assert mss.mu_sphere == 1.0
        assert mss.mu_background == 1.0
        assert np.all(mss.location == np.r_[0., 0., 0.])

    def test_errors(self):
        mss = static.MagnetostaticSphere(amplitude=1.0, radius=1.0, mu_sphere=1.0, mu_background=1.0, location=None)
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

    def testV(self):
        radius = 1.0
        amplitude = 1.0
        mu_s = 1.0
        mu_b = 1.0
        location = [0., 0., 0.]
        mss = static.MagnetostaticSphere(
            radius=radius,
            amplitude=amplitude,
            mu_background=mu_b,
            mu_sphere=mu_s,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])

        vtest = V_from_Sphere(
            xyz, mss.location, mss.mu_sphere, mss.mu_background, mss.radius, mss.amplitude
        )
        print(
            "\n\nTesting Magnetic Potential V for Sphere\n"
        )

        vt, vp, vs = mss.potential(xyz, field='all')
        np.testing.assert_equal(vtest, vt)

    def testH(self):
        radius = 1.0
        amplitude = 1.0
        mu_s = 1.0
        mu_b = 1.0
        location = [0., 0., 0.]
        mss = static.MagnetostaticSphere(
            radius=radius,
            amplitude=amplitude,
            mu_background=mu_b,
            mu_sphere=mu_s,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])

        htest = H_from_Sphere(
            xyz, mss.location, mss.mu_sphere, mss.mu_background, mss.radius, mss.amplitude
        )
        print(
            "\n\nTesting Magnetic Field H for Sphere\n"
        )

        ht, hp, hs = mss.magnetic_field(xyz, field='all')
        np.testing.assert_equal(htest, ht)

    def testB(self):
        radius = 1.0
        amplitude = 1.0
        mu_s = 1.0
        mu_b = 1.0
        location = [0., 0., 0.]
        mss = static.MagnetostaticSphere(
            radius=radius,
            amplitude=amplitude,
            mu_background=mu_b,
            mu_sphere=mu_s,
            location=location
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])

        btest = B_from_Sphere(
            xyz, mss.location, mss.mu_sphere, mss.mu_background, mss.radius, mss.amplitude
        )
        print(
            "\n\nTesting Magnetic Flux Density B for Sphere\n"
        )

        bt, bp, bs = mss.magnetic_flux_density(xyz, field='all')
        np.testing.assert_equal(btest, bt)
