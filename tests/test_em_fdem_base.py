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
    edws = fdem.ElectricDipoleWholeSpace(1)
    with pytest.raises(TypeError):
        edws.frequency = "string"
    with pytest.raises(ValueError):
        edws.frequency = -1
    with pytest.raises(TypeError):
        edws.frequency = np.array([[1, 2], [3, 4]])
