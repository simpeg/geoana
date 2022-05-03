import pytest
import numpy as np
from scipy.constants import G

from geoana import gravity


class TestPointMass:

    def test_defaults(self):
        pm = gravity.PointMass()
        assert pm.mass == 1
        assert np.allclose(pm.location, np.array([0, 0, 0]))


