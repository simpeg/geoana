import unittest
import pytest

import numpy as np
import numpy.testing as npt
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

    def test_cartesian_to_cylindrical(self):
        cart_points = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
            [1, 1, 0],
        ])

        cyl_points = np.array([
            [1, 0, 0],
            [1, np.pi/2, 0],
            [0, 0, 1],
            [1, np.pi, 0],
            [1, -np.pi/2, 0],
            [0, 0, -1],
            [np.sqrt(2), np.pi / 4, 0]
        ])

        out_cyl = spatial.cartesian_to_cylindrical(cart_points)
        np.testing.assert_allclose(out_cyl, cyl_points, atol=1E-15)

        out_cart = spatial.cylindrical_to_cartesian(cyl_points)
        np.testing.assert_allclose(out_cart, cart_points, atol=1E-15)

    def test_cartesian_to_cylindrical_vector(self):
        cart_points = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
            [1, 1, 0],
        ])

        cyl_vecs = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, -1],
            [np.sqrt(2), 0, 0]
        ])

        cart_vecs2 = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, -1],
            [-1, 1, 0],
        ])

        cyl_vecs2 = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, np.sqrt(2), 0],
        ])

        out_cyl = spatial.cartesian_to_cylindrical(cart_points, cart_points)
        np.testing.assert_allclose(out_cyl, cyl_vecs, atol=1E-15)

        out_cyl = spatial.cartesian_to_cylindrical(cart_points, cart_vecs2)
        np.testing.assert_allclose(out_cyl, cyl_vecs2, atol=1E-15)

        cyl_points = spatial.cartesian_to_cylindrical(cart_points)

        out_cart = spatial.cylindrical_to_cartesian(cyl_points, cyl_vecs)
        np.testing.assert_allclose(out_cart, cart_points, atol=1E-15)

        out_cart = spatial.cylindrical_to_cartesian(cyl_points, cyl_vecs2)
        np.testing.assert_allclose(out_cart, cart_vecs2, atol=1E-15)

    def test_cartesian_to_spherical(self):
        test_points = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1],
            [0, -1, 0],
            [-1, 0, 0],
        ])

        sph_points = np.array([
            [1, 0, 0],
            [1, np.pi/2, np.pi/2],
            [1, 0, np.pi/2],
            [1, 0, np.pi],
            [1, -np.pi/2, np.pi/2],
            [1, np.pi, np.pi/2],
        ])

        test_sph = spatial.cartesian_to_spherical(test_points)

        npt.assert_allclose(test_sph, sph_points, atol=1E-15)

        undo_test = spatial.spherical_2_cartesian(sph_points)

        npt.assert_allclose(undo_test, test_points, atol=1E-15)

    def test_cartesian_to_spherical_vector(self):
        cart_points = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1],
            [0, -1, 0],
            [-1, 0, 0],
        ])

        cart_vec2 = np.array([
            [1, 0, 0],
            [-1, 0, 0],
            [0, -1, 0],
            [-1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ])

        sphere_vecs2 = np.array([
            [0, 0, 1],
            [0, 1, 0],   # @ y=1, vec = -x_hat -> +phi_hat
            [0, -1, 0],  # @ x=1, vec = -y_hat -> -phi_hat
            [0, 0, 1],
            [0, 1, 0],   # @ y=-1, vec=x-hat -> + phi_hat
            [0, -1, 0],   # @ x=-1, vec=y_hat -> - phi_hat
        ])

        # If we use these test points as the vectors to transform
        # it should give us back vectors with only a radial component.
        sphere_vecs = np.zeros_like(cart_points)
        sphere_vecs[:, 0] = 1

        test_vecs = spatial.cartesian_to_spherical(cart_points, cart_points)
        npt.assert_allclose(test_vecs, sphere_vecs, atol=1E-15)

        test_vecs = spatial.cartesian_to_spherical(cart_points, cart_vec2)
        npt.assert_allclose(test_vecs, sphere_vecs2, atol=1E-15)

        sphere_points = spatial.cartesian_to_spherical(cart_points)
        out_cart = spatial.spherical_to_cartesian(sphere_points, sphere_vecs)
        npt.assert_allclose(out_cart, cart_points, atol=1E-15)

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
