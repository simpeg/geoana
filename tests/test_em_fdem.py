from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

from geoana.em import fdem


def E_from_EDWS(XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', kappa=0., epsr=1., t=0.):
    """E_from_EDWS
    Computing the analytic electric fields (E) from an electrical dipole in a wholespace
    - You have the option of computing E for multiple frequencies at a single reciever location
      or a single frequency at multiple locations

    :param numpy.array XYZ: reciever locations at which to evaluate E
    :param float epsr: relative permitivitty value (unitless),  default is 1.0
    :rtype: numpy.array
    :return: Ex, Ey, Ez: arrays containing all 3 components of E evaluated at the specified locations and frequencies.
    """

    mu = mu_0*(1+kappa)
    epsilon = epsilon_0*epsr
    sig_hat = sig + 1j*omega(f)*epsilon

    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
    # Check
    if XYZ.shape[0] > 1 & f.shape[0] > 1:
        raise Exception('I/O type error: For multiple field locations only a single frequency can be specified.')

    dx = XYZ[:, 0] - srcLoc[0]
    dy = XYZ[:, 1] - srcLoc[1]
    dz = XYZ[:, 2] - srcLoc[2]

    r = np.sqrt(dx**2. + dy**2. + dz**2.)
    # k  = np.sqrt( -1j*2.*pi*f*mu*sig )
    k = np.sqrt(omega(f)**2. * mu * epsilon - 1j * omega(f) * mu * sig)

    front = current * length / (4.*pi*sig_hat* r**3) * np.exp(-1j*k*r)
    mid   = -k**2 * r**2 + 3*1j*k*r + 3

    if orientation.upper() == 'X':
        Ex = front*((dx**2 / r**2)*mid + (k**2 * r**2 -1j*k*r-1.))
        Ey = front*(dx*dy  / r**2)*mid
        Ez = front*(dx*dz  / r**2)*mid
        return Ex, Ey, Ez

    elif orientation.upper() == 'Y':
        #  x--> y, y--> z, z-->x
        Ey = front*((dy**2 / r**2)*mid + (k**2 * r**2 -1j*k*r-1.))
        Ez = front*(dy*dz  / r**2)*mid
        Ex = front*(dy*dx  / r**2)*mid
        return Ex, Ey, Ez

    elif orientation.upper() == 'Z':
        # x --> z, y --> x, z --> y
        Ez = front*((dz**2 / r**2)*mid + (k**2 * r**2 -1j*k*r-1.))
        Ex = front*(dz*dx  / r**2)*mid
        Ey = front*(dz*dy  / r**2)*mid
        return Ex, Ey, Ez


class TestFDEM(unittest.TestCase):

    def test_vector(self):
        edws = fdem.ElectricDipole_WholeSpace()




if __name__ == '__main__':
    unittest.main()
