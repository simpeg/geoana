from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import geoana
import geoana.traits as tr


class SimpleAnalytic(geoana.BaseAnalytic):
        orientation = tr.Vector(
            help='Orientation of the dipole',
            normalize=True,
            default_value='up'
        )


class TestTraits(unittest.TestCase):

    def test_vector(self):

        v = SimpleAnalytic()
        print(v.orientation)
        v.orientation = [1, 2, 2.]
        print(v.orientation)

        print(v.__doc__)

        v.orientation = 'up'
        print(v.orientation)


if __name__ == '__main__':
    unittest.main()
