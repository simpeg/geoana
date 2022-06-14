import pytest
import numpy as np

from scipy.constants import mu_0
from geoana.em import tdem


def test_peak_time():
    z = 2
    z_array = np.linspace(-1, 1, 20)
    sigma = 2
    sigma_array = np.linspace(-1, 1, 20)
    mu = 2
    mu_array = np.linspace(-1, 1, 20)

    peak_time = (mu * sigma * z ** 2) / 6
    peak_time_array = (mu_array * sigma_array * z_array ** 2) / 6
    peak_time_default = (mu_0 * sigma * z ** 2) / 6

    peak_time_test = tdem.peak_time(z, sigma, mu)
    peak_time_array_test = tdem.peak_time(z_array, sigma_array, mu_array)
    peak_time_default_test = tdem.peak_time(z, sigma)

    np.testing.assert_equal(peak_time, peak_time_test)
    np.testing.assert_equal(peak_time_array, peak_time_array_test)
    np.testing.assert_equal(peak_time_default, peak_time_default_test)


def test_diffusion_distance():
    time = 2
    time_array = np.linspace(-1, 1, 20)
    sigma = 2
    sigma_array = np.linspace(-1, 1, 20)
    mu = 2
    mu_array = np.linspace(-1, 1, 20)

    diffusion_distance = np.sqrt(2 * time / (mu * sigma))
    diffusion_distance_array = np.sqrt(2 * time_array / (mu_array * sigma_array))
    diffusion_distance_default = np.sqrt(2 * time / (mu_0 * sigma))

    diffusion_distance_test = tdem.diffusion_distance(time, sigma, mu)
    diffusion_distance_array_test = tdem.diffusion_distance(time_array, sigma_array, mu_array)
    diffusion_distance_default_test = tdem.diffusion_distance(time, sigma)

    np.testing.assert_equal(diffusion_distance, diffusion_distance_test)
    np.testing.assert_equal(diffusion_distance_array, diffusion_distance_array_test)
    np.testing.assert_equal(diffusion_distance_default, diffusion_distance_default_test)


def test_base_tdem():
    with pytest.raises(TypeError):
        tdem.VerticalMagneticDipoleHalfSpace(time="string")
    with pytest.raises(ValueError):
        tdem.VerticalMagneticDipoleHalfSpace(time=-1)
    with pytest.raises(TypeError):
        tdem.VerticalMagneticDipoleHalfSpace(time=np.array([[1, 2], [3, 4]]))

