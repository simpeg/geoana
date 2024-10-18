import pytest
import numpy as np

from scipy.constants import mu_0, epsilon_0
from geoana.em import fdem


def test_skin_depth():
    w = fdem.omega(1)
    qsd = np.sqrt(2 / (w * 1 * mu_0))
    sd = np.sqrt((mu_0 * epsilon_0 / 2) * (np.sqrt(1 + 1 ** 2 / (w * epsilon_0) ** 2) - 1)) / w
    qsd_test = fdem.skin_depth(1, 1, quasistatic=True)
    sd_test = fdem.skin_depth(1, 1, quasistatic=False)

    np.testing.assert_equal(qsd_test, qsd)
    np.testing.assert_equal(sd_test, sd)


def test_sigma_hat():
    qsh = fdem.sigma_hat(1, 1, quasistatic=True)
    sh = fdem.sigma_hat(1, 1, quasistatic=False)
    sh_test = 1 + 1j * fdem.omega(1) * epsilon_0

    np.testing.assert_equal(1, qsh)
    np.testing.assert_equal(sh_test, sh)


def test_base_fdem():
    # gets frequency and quasistatic properties.
    with pytest.raises(TypeError, match="frequencies are not a valid type"):
        fdem.BaseFDEM("string")
    with pytest.raises(ValueError, match="All frequencies must be greater than 0"):
        fdem.BaseFDEM(-1)
    with pytest.raises(TypeError, match="frequencies must have at most 1 dimension."):
        fdem.BaseFDEM(np.array([[1, 2], [3, 4]]))

    fd = fdem.BaseFDEM(frequency=np.logspace(1, 4, 3), sigma=1, quasistatic=True)
    fds = fd.sigma_hat
    np.testing.assert_equal(1, fds)

    assert fd.quasistatic is True



