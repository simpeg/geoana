import numpy as np
from geoana.utils import check_xyz_dim, mkvc
import pytest


def test_mkvc():
    x = np.random.rand(3, 2)
    x_test = np.concatenate((x[:, 0], x[:, 1]), axis=None)
    x_new = mkvc(x)
    np.testing.assert_equal(x_test, x_new)

    y = np.matrix('1 2; 3 4')
    y_test = np.array([[1, 2], [3, 4]])
    y_test = np.concatenate((y_test[:, 0], y_test[:, 1]), axis=None)
    y_new = mkvc(y)
    np.testing.assert_equal(y_test, y_new)


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
    with pytest.warns(np.ComplexWarning):
        xyz2 = check_xyz_dim(xyz)
    xyz = [['0', 'O', 'o']]
    with pytest.raises(ValueError):
        xyz2 = check_xyz_dim(xyz)


def test_bad_dim():
    xyz = np.random.rand(10, 3, 2)
    with pytest.raises(ValueError):
        xyz = check_xyz_dim(xyz)
