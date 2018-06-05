from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
from scipy.constants import mu_0, epsilon_0
import discretize

from geoana.em import static, fdem
from geoana import spatial


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
        n = 100
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

            self.assertTrue(np.allclose(
                self.clws.vector_potential(mesh.gridCC)[inds],
                self.mdws.vector_potential(mesh.gridCC)[inds],
            ))

    def test_magnetic_field_tensor(self):
        n = 50
        mesh = discretize.TensorMesh(
            [np.ones(n), np.ones(n), np.ones(n)], x0="CCC"
        )

        radius = 1
        self.clws.radius = radius
        self.clws.current = 1./(np.pi * radius**2)

        fdem_dipole = fdem.MagneticDipoleWholeSpace(frequency=0)

        for location in [
            np.r_[0, 0, 0], np.r_[10, 0, 0], np.r_[0, -10, 0], np.r_[0, 0, 10],
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

                inds = (np.hstack([
                    (np.absolute(mesh.gridFx[:, 0]) > 5 + location[0]) &
                    (np.absolute(mesh.gridFx[:, 1]) > 5 + location[1]) &
                    (np.absolute(mesh.gridFx[:, 2]) > 5 + location[2]),
                    (np.absolute(mesh.gridFy[:, 0]) > 5 + location[0]) &
                    (np.absolute(mesh.gridFy[:, 1]) > 5 + location[1]) &
                    (np.absolute(mesh.gridFy[:, 2]) > 5 + location[2]),
                    (np.absolute(mesh.gridFz[:, 0]) > 5 + location[0]) &
                    (np.absolute(mesh.gridFz[:, 1]) > 5 + location[1]) &
                    (np.absolute(mesh.gridFz[:, 2]) > 5 + location[2])
                ]))

                self.assertTrue(np.allclose(b_fdem[inds], b_clws[inds]))
                self.assertTrue(np.allclose(b_fdem[inds], b_mdws[inds]))

    def test_magnetic_field_3Dcyl(self):
        n = 50
        ny = 10
        mesh = discretize.CylMesh(
            [np.ones(n), np.ones(ny) * 2 * np.pi / ny, np.ones(n)], x0="00C"
        )

        radius = 1
        self.clws.radius = radius
        self.clws.current = 1./(np.pi * radius**2)

        fdem_dipole = fdem.MagneticDipoleWholeSpace(frequency=0)

        for location in [
            np.r_[0, 0, 0], np.r_[10, 0, 0], np.r_[10, np.pi/2., 0],
            np.r_[0, 0, 10]
        ]:
            self.clws.location = location
            self.mdws.location = location
            fdem_dipole.location = location

            for orientation in ["x", "y", "z"]:
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
                    (np.absolute(mesh.gridFx[:, 0]) > 5 + location[0]) &
                    (np.absolute(mesh.gridFx[:, 1]) > 0 + location[1]) &
                    (np.absolute(mesh.gridFx[:, 2]) > 5 + location[2]),
                    (np.absolute(mesh.gridFy[:, 0]) > 5 + location[0]) &
                    (np.absolute(mesh.gridFy[:, 1]) > 0 + location[1]) &
                    (np.absolute(mesh.gridFy[:, 2]) > 5 + location[2]),
                    (np.absolute(mesh.gridFz[:, 0]) > 5 + location[0]) &
                    (np.absolute(mesh.gridFz[:, 1]) > 0 + location[1]) &
                    (np.absolute(mesh.gridFz[:, 2]) > 5 + location[2])
                ]))

                self.assertTrue(np.allclose(b_fdem[inds], b_clws[inds]))
                self.assertTrue(np.allclose(b_fdem[inds], b_mdws[inds]))

if __name__ == '__main__':
    unittest.main()
