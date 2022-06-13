import unittest
import pytest

import numpy as np
from geoana import spatial


def test_errors():
    x, y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
    xyz = np.stack((x, y), axis=-1)
    with pytest.raises(AssertionError):
        spatial.rotate_points_from_normals(xyz, np.r_[1, 1, 1], np.r_[1, 1, 0])
    with pytest.raises(Exception):
        spatial.vector_dot(xyz, np.r_[1, 1])

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


if __name__ == '__main__':
    unittest.main()
