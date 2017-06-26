from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np

from scipy.constants import mu_0, epsilon_0
from geoana.em import fdem
from discretize.utils import ndgrid, asArray_N_x_Dim


def E_from_EDWS(
    XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', kappa=0.,
    epsr=1., t=0.
):
    """E_from_EDWS
    Computing the analytic electric fields (E) from an electrical dipole in
    a wholespace
    - You have the option of computing E for multiple frequencies at a single
    reciever location
      or a single frequency at multiple locations

    :param numpy.array XYZ: reciever locations at which to evaluate E
    :param float epsr: relative permitivitty value (unitless),  default is 1.0
    :rtype: numpy.array
    :return: Ex, Ey, Ez: arrays containing all 3 components of E evaluated at the specified locations and frequencies.
    """

    mu = mu_0*(1+kappa)
    epsilon = epsilon_0*epsr
    sig_hat = sig + 1j*fdem.omega(f)*epsilon

    XYZ = asArray_N_x_Dim(XYZ, 3)

    dx = XYZ[:, 0] - srcLoc[0]
    dy = XYZ[:, 1] - srcLoc[1]
    dz = XYZ[:, 2] - srcLoc[2]

    r = np.sqrt(dx**2. + dy**2. + dz**2.)
    # k  = np.sqrt( -1j*2.*pi*f*mu*sig )
    k = np.sqrt(
        fdem.omega(f)**2. * mu * epsilon - 1j * fdem.omega(f) * mu * sig
    )

    front = current * length / (4.*np.pi*sig_hat * r**3) * np.exp(-1j*k*r)
    mid = -k**2 * r**2 + 3*1j*k*r + 3

    if orientation.upper() == 'X':
        Ex = front*((dx**2 / r**2)*mid + (k**2 * r**2 - 1j*k*r-1.))
        Ey = front*(dx*dy / r**2)*mid
        Ez = front*(dx*dz / r**2)*mid
        return Ex, Ey, Ez

    elif orientation.upper() == 'Y':
        #  x--> y, y--> z, z-->x
        Ey = front*((dy**2 / r**2)*mid + (k**2 * r**2 - 1j*k*r-1.))
        Ez = front*(dy*dz / r**2)*mid
        Ex = front*(dy*dx / r**2)*mid
        return Ex, Ey, Ez

    elif orientation.upper() == 'Z':
        # x --> z, y --> x, z --> y
        Ez = front*((dz**2 / r**2)*mid + (k**2 * r**2 - 1j*k*r-1.))
        Ex = front*(dz*dx / r**2)*mid
        Ey = front*(dz*dy / r**2)*mid
        return Ex, Ey, Ez


class TestFDEMdipole(unittest.TestCase):

    def test_defaults(self):
        edws = fdem.ElectricDipoleWholeSpace()
        assert(edws.sigma == 1)
        assert(edws.mu == mu_0)
        assert(edws.epsilon == epsilon_0)
        assert(np.all(edws.orientation == np.r_[1., 0., 0.]))
        assert(edws.length == 1.)
        assert(np.all(edws.location == np.r_[0., 0., 0.]))
        assert(edws.frequency == 1.)

    def compare_fields(name, field, ftest):

        def check_component(name, f, ftest):
            geoana_norm = np.linalg.norm(f)
            test_norm = np.linalg.norm(ftest)
            diff = np.linalg.norm(f-ftest)
            passed = np.allclose(f, ftest)
            print(
                "Testing {} ... geoana: {:1.4e}, compare: {:1.4e}, "
                "diff: {:1.4e}, passed?: {}".format(
                    name, geoana_norm, test_norm, diff, passed
                )
            )
            return passed

        for i, orientation in enumerate(['x', 'y', 'z']):
            for component in ['real', 'imag']:
                check_component(
                    orientation + '_' + component,
                    getattr(field[:, i], component),
                    getattr(ftest[:, i], component)
                )

    def electric_dipole_e(self, orientation):
        sigma = np.exp(np.random.randn(1))
        frequency = np.random.rand(1)*1e6
        edws = fdem.ElectricDipoleWholeSpace(
            orientation=orientation,
            sigma=sigma,
            frequency=frequency
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = ndgrid([x, y, z])

        extest, eytest, eztest = E_from_EDWS(
            xyz, edws.location, edws.sigma, edws.frequency,
            orientation=orientation.upper()
        )

        e = edws.electric_field(xyz)
        print(
            "\n\nTesting Electric Dipole {} orientation\n".format(orientation)
        )

        self.compare_fields(e, np.vstack([extest, eytest, eztest]).T)

    def test_electric_dipole_x_e(self):
        self.electric_dipole_e("x")

    def test_electric_dipole_y_e(self):
        self.electric_dipole_e("y")

    def test_electric_dipole_z_e(self):
        self.electric_dipole_e("z")

    def test_electric_dipole_tilted_e(self):

        orientation = np.random.rand(3)
        orientation = orientation / np.linalg.norm(orientation)

        edws = fdem.ElectricDipoleWholeSpace(
            orientation=orientation
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)

        xyz = ndgrid([x, y, z])

        extest0, eytest0, eztest0 = E_from_EDWS(
            xyz, edws.location, edws.sigma, edws.frequency,
            orientation='X'
        )
        extest1, eytest1, eztest1 = E_from_EDWS(
            xyz, edws.location, edws.sigma, edws.frequency,
            orientation='Y'
        )
        extest2, eytest2, eztest2 = E_from_EDWS(
            xyz, edws.location, edws.sigma, edws.frequency,
            orientation='Z'
        )

        extest = (
            orientation[0]*extest0 + orientation[1]*extest1 + orientation[2]*extest2
        )
        eytest = (
            orientation[0]*eytest0 + orientation[1]*eytest1 + orientation[2]*eytest2
        )
        eztest = (
            orientation[0]*eztest0 + orientation[1]*eztest1 + orientation[2]*eztest2
        )

        e = edws.electric_field(xyz)
        print(
            "\n\nTesting Electric Dipole {} orientation\n".format("45 degree")
        )

        self.compare_fields(e, np.vstack([extest, eytest, eztest]).T)



if __name__ == '__main__':
    unittest.main()
