import numpy as np
import geoana
from geoana.utils import check_xyz_dim, mkvc, ndgrid
import pytest

try:
    from numpy.exceptions import ComplexWarning
except ImportError:
    from numpy import ComplexWarning


def test_config_info():
    info = geoana.show_config()
    assert info['version'] == geoana.__version__


def test_mkvc():
    x = np.random.rand(3, 2)
    x_test = np.concatenate((x[:, 0], x[:, 1]), axis=None)
    x_new = mkvc(x)
    np.testing.assert_equal(x_test, x_new)


def test_nd_grid():
    x = np.array([1])
    x_test = np.array([1])
    x_new = ndgrid(x)
    np.testing.assert_equal(x_test, x_new)

    x = np.array([[1, 2, 3]])
    x_test = np.array([[1, 2, 3]])
    x_new = ndgrid(x)
    np.testing.assert_equal(x_test, x_new)

    x = np.array([[1, 2, 3]])
    y = np.array([[4, 5]])
    xy_test_1 = np.array([[1, 4], [2, 4], [3, 4], [1, 5], [2, 5], [3, 5]])
    xy_new_1 = ndgrid(x, y)
    x_test = np.array([[1, 1], [2, 2], [3, 3]])
    y_test = np.array([[4, 5], [4, 5], [4, 5]])
    x_new, y_new = ndgrid(x, y, vector=False)
    np.testing.assert_equal(xy_test_1, xy_new_1)
    np.testing.assert_equal(x_test, x_new)
    np.testing.assert_equal(y_test, y_new)

    x = np.array([[1, 2]])
    y = np.array([[3, 4]])
    z = np.array([[5, 6]])
    xy_test_1 = np.array([[1, 3, 5], [2, 3, 5], [1, 4, 5], [2, 4, 5], [1, 3, 6], [2, 3, 6], [1, 4, 6], [2, 4, 6]])
    xy_new_1 = ndgrid(x, y, z)
    x_test = np.array([[[1, 1], [1, 1]], [[2, 2], [2, 2]]])
    y_test = np.array([[[3, 3], [4, 4]], [[3, 3], [4, 4]]])
    z_test = np.array([[[5, 6], [5, 6]], [[5, 6], [5, 6]]])
    x_new, y_new, z_new = ndgrid(x, y, z, vector=False)
    np.testing.assert_equal(xy_test_1, xy_new_1)
    np.testing.assert_equal(x_test, x_new)
    np.testing.assert_equal(y_test, y_new)
    np.testing.assert_equal(z_test, z_new)


def test_x_y_z_stack():
    xyz = np.random.rand(10, 4, 3)
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    xyz2 = check_xyz_dim((x, y, z))
    np.testing.assert_equal(xyz, xyz2)
    assert xyz2.ndim == 3
    assert xyz2.shape == (10, 4, 3)


def test_good_pass_through():
    xyz = np.random.rand(20, 2, 3)
    xyz2 = check_xyz_dim(xyz)
    assert xyz is xyz2


def test_bad_stack():
    x = np.random.rand(10, 2)
    y = np.random.rand(12, 3)
    z = np.random.rand(20, 2)

    with pytest.raises(ValueError):
        check_xyz_dim((x, y, z))


def test_dtype_cast_good():
    xyz_int = np.random.randint(0, 10, (10, 4, 3))

    xyz2 = check_xyz_dim(xyz_int, dtype=float)
    assert np.issubdtype(xyz2.dtype, float)
    assert xyz_int is not xyz2


def test_bad_dtype_cast():
    xyz = np.random.rand(10, 3) + 1j* np.random.rand(10, 3)
    with pytest.warns(ComplexWarning):
        xyz2 = check_xyz_dim(xyz)
    xyz = [['0', 'O', 'o']]
    with pytest.raises(ValueError):
        xyz2 = check_xyz_dim(xyz)


def test_bad_dim():
    xyz = np.random.rand(10, 3, 2)
    with pytest.raises(ValueError):
        xyz = check_xyz_dim(xyz)

