import unittest
import pytest

import numpy as np
from geoana import utils
from geoana import spatial


def test_errors():
    x, y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
    xy = np.stack((x, y), axis=-1)
    with pytest.raises(AssertionError):
        spatial.rotate_points_from_normals(xy, np.r_[1, 1, 1], np.r_[1, 1, 0])
    with pytest.raises(Exception):
        spatial.vector_dot(xy, np.r_[1, 1])
    x, y = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
    xy = np.stack((x, y), axis=-1)
    with pytest.raises(Exception):
        spatial.vector_distance(xy, np.r_[1, 2])


class TestCoordinates(unittest.TestCase):

    def test_rotate_vec_cyl2cart(self):
        vec = np.r_[1., 0, 0].reshape(1, 3)
        grid = np.r_[1., np.pi/4, 0].reshape(1, 3)
        self.assertTrue(np.allclose(
            spatial.cylindrical_to_cartesian(grid, vec),
            np.sqrt(2)/2 * np.r_[1, 1, 0]
        ))
        self.assertTrue(np.allclose(
            spatial.cylindrical_to_cartesian(grid),
            np.sqrt(2)/2 * np.r_[1, 1, 0]
        ))
        self.assertTrue(np.allclose(
            spatial.cartesian_to_cylindrical(
                np.sqrt(2)/2 * np.r_[1, 1, 0]
            ),
            grid
        ))

        self.assertTrue(np.allclose(
            spatial.cartesian_to_cylindrical(
                spatial.cylindrical_to_cartesian(grid)
            ),
            grid
        ))

        self.assertTrue(np.allclose(
            spatial.cartesian_to_cylindrical(
                np.sqrt(2)/2 * np.r_[1, 1, 0],
                spatial.cylindrical_to_cartesian(grid, vec)
            ),
            vec
        ))

        vec = np.r_[0, 1, 2].reshape(1, 3)
        grid = np.r_[1, np.pi/4, 0].reshape(1, 3)
        self.assertTrue(np.allclose(
            spatial.cylindrical_to_cartesian(grid, vec),
            np.r_[-np.sqrt(2)/2, np.sqrt(2)/2, 2]
        ))

        vec = np.r_[1., 0]
        grid = np.r_[1., np.pi/4].reshape(1, 2)
        self.assertTrue(np.allclose(
            spatial.cylindrical_to_cartesian(grid, vec),
            np.sqrt(2)/2 * np.r_[1, 1]
        ))

    def test_cartesian_to_cylindrical(self):
        vec = np.r_[1., 0, 0]
        grid = np.r_[1., np.pi / 4, 0]
        grid_ = np.atleast_2d(grid)
        vec_ = vec.reshape(grid_.shape, order='F')
        theta = np.arctan2(grid_[:, 1], grid_[:, 0])
        c2c_test = np.hstack([utils.mkvc(np.cos(theta) * vec_[:, 0] + np.sin(theta) * vec_[:, 1], 2),
                              utils.mkvc(-np.sin(theta) * vec_[:, 0] + np.cos(theta) * vec_[:, 1], 2),
                              utils.mkvc(vec_[:, 2], 2)])
        c2c = spatial.cartesian_to_cylindrical(grid, vec)
        np.testing.assert_equal(c2c_test, c2c)

    def test_vector_dot(self):
        xyz = utils.ndgrid(np.linspace(-1, 1, 20), np.array([0]), np.linspace(-1, 1, 20))
        vector = np.array([1, 2, 3])
        vd = spatial.vector_dot(xyz, vector)
        vd_test = 1 * xyz[:, 0] + 3 * xyz[:, 2]
        np.testing.assert_equal(vd_test, vd)

    def test_distance(self):
        xyz = utils.ndgrid(np.linspace(-1, 1, 20), np.array([0]), np.linspace(-1, 1, 20))
        dxyz = spatial.vector_distance(xyz)
        v = np.atleast_2d(dxyz)
        vm = np.sqrt((v ** 2).sum(axis=1))
        distance = spatial.distance(xyz)
        np.testing.assert_equal(vm, distance)

    def test_aliases(self):
        vec = np.r_[1., 0, 0].reshape(1, 3)
        grid = np.r_[1., np.pi / 4, 0].reshape(1, 3)
        s2c = spatial.spherical_2_cartesian(grid, vec)
        c2s = spatial.cartesian_2_spherical(grid, vec)
        np.testing.assert_equal(s2c, spatial.spherical_to_cartesian(grid, vec))
        np.testing.assert_equal(c2s, spatial.cartesian_to_spherical(grid, vec))

        s2c = spatial.spherical_2_cartesian(grid)
        c2s = spatial.cartesian_2_spherical(grid)
        np.testing.assert_equal(s2c, spatial.spherical_to_cartesian(grid))
        np.testing.assert_equal(c2s, spatial.cartesian_to_spherical(grid))

        vec = np.r_[1, 0, 0]
        s2c = spatial.spherical_2_cartesian(grid, vec)
        c2s = spatial.cartesian_2_spherical(grid, vec)
        np.testing.assert_equal(s2c, spatial.spherical_to_cartesian(grid, vec))
        np.testing.assert_equal(c2s, spatial.cartesian_to_spherical(grid, vec))


if __name__ == '__main__':
    unittest.main()
