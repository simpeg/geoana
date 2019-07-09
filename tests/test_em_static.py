from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
from scipy.constants import mu_0, epsilon_0
import discretize
import properties

from geoana.electromagnetics import static, fdem
from geoana import spatial

TOL = 0.1

class TestEM_Static(unittest.TestCase):

    def setUp(self):
        self.mdws = static.MagneticDipoleWholeSpace()
        self.clws = static.CircularLoopWholeSpace()

    def test_defaults(self):
        self.assertTrue(self.mdws.sigma == 1)
        self.assertTrue(self.clws.sigma == 1)

        self.assertTrue(self.mdws.mu == mu_0)
        self.assertTrue(self.clws.mu == mu_0)

        self.assertTrue(self.mdws.epsilon == epsilon_0)
        self.assertTrue(self.clws.epsilon == epsilon_0)

        self.assertTrue(np.all(self.mdws.orientation == np.r_[1., 0., 0.]))
        self.assertTrue(np.all(self.clws.orientation == np.r_[1., 0., 0.]))

        self.assertTrue(self.mdws.moment == 1)
        self.assertTrue(self.clws.current == 1)
        self.assertTrue(self.clws.radius == 1)

    def test_vector_potential(self):
        n = 50
        mesh = discretize.TensorMesh(
            [np.ones(n), np.ones(n), np.ones(n)], x0="CCC"
        )

        radius = 1
        self.clws.radius = radius
        self.clws.current = 1./(np.pi * radius**2)

        for orientation in ["x", "y", "z"]:
            self.clws.orientation = orientation
            self.mdws.orientation = orientation

            inds = (
                (np.absolute(mesh.gridCC[:, 0]) > 5) &
                (np.absolute(mesh.gridCC[:, 1]) > 5) &
                (np.absolute(mesh.gridCC[:, 2]) > 5)
            )

            a_clws = self.clws.vector_potential(mesh.gridCC)[inds]
            a_mdws = self.mdws.vector_potential(mesh.gridCC)[inds]

            self.assertTrue(isinstance(a_clws, np.ndarray))
            self.assertFalse(isinstance(a_clws, properties.Vector3))

            self.assertTrue(isinstance(a_mdws, np.ndarray))
            self.assertFalse(isinstance(a_mdws, properties.Vector3))

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

                    b_clws = mesh.edgeCurl * a_clws
                    b_mdws = mesh.edgeCurl * a_mdws

                    b_fdem = np.hstack([
                        fdem_dipole.magnetic_flux_density(mesh.gridFx)[:, 0],
                        fdem_dipole.magnetic_flux_density(mesh.gridFy)[:, 1],
                        fdem_dipole.magnetic_flux_density(mesh.gridFz)[:, 2]
                    ])

                    self.assertTrue(isinstance(b_fdem, np.ndarray))
                    self.assertFalse(isinstance(b_fdem, properties.Vector3))

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

    def test_magnetic_field_3Dcyl(self):
        print("\n === Testing 3D Cyl Mesh === \n")
        n = 50
        ny = 10
        h = 2.
        mesh = discretize.CylMesh(
            [h*np.ones(n), np.ones(ny) * 2 * np.pi / ny, h*np.ones(n)], x0="00C"
        )

        for radius in [0.5, 1, 1.5]:
            self.clws.radius = radius
            self.clws.current = 1./(np.pi * radius**2)

            fdem_dipole = fdem.MagneticDipoleWholeSpace(frequency=0)

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

                    b_clws = mesh.edgeCurl * a_clws
                    b_mdws = mesh.edgeCurl * a_mdws

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

if __name__ == '__main__':
    unittest.main()
