import pytest
from scipy.constants import mu_0, epsilon_0

from geoana.em import BaseEM, BaseDipole, BaseElectricDipole, BaseMagneticDipole

import numpy as np
import numpy.testing as npt

def test_base_defaults():
    base_em = BaseEM()
    assert base_em.sigma == 1
    assert base_em.mu == mu_0
    assert base_em.epsilon == epsilon_0

def test_base_errors():
    with pytest.raises(TypeError, match="sigma must be a number"):
        BaseEM(sigma='box')
    with pytest.raises(ValueError, match="sigma must be greater than 0"):
        BaseEM(sigma = -2)
    with pytest.raises(TypeError, match="epsilon must be a number.*"):
        BaseEM(epsilon = "box")
    with pytest.raises(ValueError, match="Invalid epsilon.*"):
        BaseEM(epsilon = -2)
    with pytest.raises(TypeError, match="mu must be a number.*"):
        BaseEM(mu = "box")
    with pytest.raises(ValueError, match="mu must be greater than 0"):
        BaseEM(mu = -2)

def test_dipole_defaults():
    base_dipole = BaseDipole()
    npt.assert_equal(base_dipole.location, [0, 0, 0])
    npt.assert_equal(base_dipole.orientation, [1, 0, 0])

    base_dipole.orientation = "Y"
    npt.assert_equal(base_dipole.orientation, [0, 1, 0])

    base_dipole.orientation = "Z"
    npt.assert_equal(base_dipole.orientation, [0, 0, 1])

    base_dipole.orientation = [1, 1, 1]
    npt.assert_allclose(base_dipole.orientation, np.sqrt([1/3, 1/3, 1/3]))

def test_dipole_errors():

    with pytest.raises(TypeError, match="location must be array_like of float.*"):
        BaseDipole(location="box")
    with pytest.raises(ValueError, match="location must be array_like with shape.*"):
        BaseDipole(location=[0, 0])
    with pytest.raises(ValueError, match="Orientation must be one of {'X','Y','Z'}"):
        BaseDipole(orientation="box")
    with pytest.raises(TypeError, match="orientation must be str or array_like.*"):
        BaseDipole(orientation=["a"])
    with pytest.raises(ValueError, match="orientation must be array_like with shape.*"):
        BaseDipole(orientation=[0, 0])

def test_e_dipole_defaults():
    base_dipole = BaseElectricDipole()
    assert base_dipole.length == 1
    assert base_dipole.current == 1

def test_e_dipole_errors():

    with pytest.raises(TypeError, match="length must be a number, got.*"):
        BaseElectricDipole(length="box")
    with pytest.raises(ValueError, match="length must be greater than 0"):
        BaseElectricDipole(length=0)
    with pytest.raises(TypeError, match="current must be a number.*"):
        BaseElectricDipole(current="box")

def test_h_dipole_defaults():
    base_dipole = BaseMagneticDipole()
    assert base_dipole.moment == 1

def test_h_dipole_errors():
    with pytest.raises(TypeError, match="moment must be a number, got.*"):
        BaseMagneticDipole(moment="box")

