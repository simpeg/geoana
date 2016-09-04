from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np

import geoana
import geoana.traits as tr


class Simple1(geoana.BaseAnalytic):
        orientation1 = tr.Vector(
            help='Orientation of the dipole',
            normalize=True,
            default_value='down'
        )

        def hello(self):
            return 'world'


class Simple2(Simple1):
        orientation2 = tr.Vector(
            help='Orientation of the dipole',
            normalize=True,
            default_value='up'
        )


class SimpleOther(geoana.BaseAnalytic):
        blah = tr.Vector(
            help='Orientation of the dipole',
            normalize=True,
            default_value='up'
        )


class Simple3(Simple2, SimpleOther):
        orientation1 = tr.Vector(
            help='Orientation of the dipole',
            normalize=True,
            default_value='up'
        )


class TestTraits(unittest.TestCase):

    def test_vector(self):

        two = Simple2()
        assert np.allclose(two.orientation1, [0, 0, -1])
        assert np.allclose(two.orientation2, [0, 0, 1])
        assert two.hello() == 'world'

        three = Simple3()
        assert np.allclose(three.orientation1, [0, 0, 1])
        assert np.allclose(three.orientation2, [0, 0, 1])
        assert np.allclose(three.blah, [0, 0, 1])

        three.orientation2 = 'west'
        assert np.allclose(three.orientation2, [-1, 0, 0])
        assert np.allclose(two.orientation2, [0, 0, 1])  # unchanged


if __name__ == '__main__':
    unittest.main()
