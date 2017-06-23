from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from scipy.constants import mu_0, pi, epsilon_0
import numpy as np
import warnings

from .base import BaseElectricDipole, BaseMagneticDipole, BaseEM


###############################################################################
#                                                                             #
#                           Utility Functions                                 #
#                                                                             #
###############################################################################

def omega(frequency):
    """
    Angular frequency

    .. math::

        \omega = 2 \pi f

    :param frequency float: frequency (Hz)
    """
    return 2*np.pi*frequency


def wave_number(
    frequency, sigma, mu=mu_0, epsilon=epsilon_0, quasi_static=False
):
    """
    Wavenumber of an electromagnetic wave in a medium with constant physical
    properties

    .. math::

        k = \sqrt{\omega**2 \mu \varepsilon - i \omega \mu \sigma}

    :param (float, numpy.ndarray) frequency: frequency (Hz)
    :param float sigma: electrical conductivity (S/m)
    :param float mu: magnetic permeability (H/m). Default: :math:`\mu_0 = 4\pi \times 10^{-7}` H/m
    :param float epsilon: dielectric permittivity (F/m). Default: :math:`\epsilon_0 = 8.85 \times 10^{-12}` F/m
    :param bool quasi_static: use the quasi-static assumption? Default: False
    """
    omega = omega(frequency)
    if quasi_static is True:
        return np.sqrt(-1j * omega * mu * sigma)
    return np.sqrt(omega**2. * mu * epsilon - 1j * omega * mu * sigma)


def skin_depth(frequency, sigma, mu=mu_0):
    """
    Distance at which an em wave has decayed by a factor of :math:`1/e` in a
    medium with constant physical properties

    .. math::

        \sqrt{\\frac{2}{\omega \sigma \mu}}

    :param float frequency: frequency (Hz)
    :param float sigma: electrical conductivity (S/m)
    :param float mu: magnetic permeability (H/m). Default: :math:`\mu_0 = 4\pi \times 10^{-7}` H/m
    """
    omega = omega(frequency)
    return np.sqrt(2./(omega*sigma*mu))


def sigma_hat(frequency, sigma, epsilon=epsilon_0, quasi_static=False):
    """
    conductivity with displacement current contribution

    .. math::

        \hat{\sigma} = \sigma + i \omega \varepsilon

    :param (float, numpy.array) frequency: frequency (Hz)
    :param float sigma: electrical conductivity (S/m)
    :param float epsilon: dielectric permittivity. Default :math:`\varepsilon_0`
    :param bool quasi_static: use the quasi-static assumption? Default: False
    """
    if quasi_static is True:
        return sigma
    return sigma + 1j*omega(frequency)*epsilon


###############################################################################
#                                                                             #
#                                  Classes                                    #
#                                                                             #
###############################################################################

class BaseFDEM(BaseEM):
    """
    Base frequency domain electromagnetic class
    """
    frequency = properties.Float(
        "Source frequency (Hz)",
        default=1e2,
        min=0.0
    )

    quasi_static = properties.Bool(
        "Use the quasi-static approximation and ignore displacement current?",
        default=False
    )

    @property
    def omega(self):
        """
        Angular frequency

        .. math::

            \omega = 2\pi f
        """
        return omega(self.frequency)

    @property
    def sigma_hat(self):
        """
        conductivity with displacement current contribution

        .. math::

            \hat{\sigma} = \sigma + i \omega \varepsilon

        """
        return sigma_hat(
            self.frequency, self.sigma, self.epsilon, self.quasi_static
        )

    @property
    def wave_number(self):
        """
        Wavenumber of an electromagnetic wave in a medium with constant physical
        properties

        .. math::

        k = \sqrt{\omega**2 \mu \varepsilon - i \omega \mu \sigma}        """
        return wave_number(self.frequency, self.sigma, self.mu)

    @property
    def skin_depth(self):
        """
        Distance at which an em wave has decayed by a factor of :math:`1/e` in a
        medium with constant physical properties

        .. math::

            \sqrt{\\frac{2}{\omega \sigma \mu}}
        """
        return skin_depth(self.frequency, self.sigma, self.mu)


class ElectricDipoleWholeSpace(BaseElectricDipole, BaseFDEM):
    """
    Harmonic electric dipole in a whole space. The source is
    (c.f. Ward and Hohmann, 1988 page 173). The source current
    density for a dipole located at :math:`\mathbf{r}_s` with orientation
    :math:`\hat{\mathbf{r}}_s`

    .. math::

        \mathbf{J}(\mathbf{r}) = I ds \delta(\mathbf{r} - \mathbf{r}_s)
    """

    def electric_field(self, xyz, **kwargs):
        pass

    def current_density(self, xyz, **kwargs):
        pass

    def magnetic_field(self, xyz, **kwargs):
        pass

    def magnetic_flux_density(self, xyz, **kwargs):
        pass


class MagneticDipoleWholeSpace(BaseMagneticDipole, BaseFDEM)

    def electric_field(self, xyz, **kwargs):
        pass

    def current_density(self, xyz, **kwargs):
        pass

    def magnetic_field(self, xyz, **kwargs):
        pass

    def magnetic_flux_density(self, xyz, **kwargs):
        pass


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


def MagneticDipoleFields(srcLoc, obsLoc, component, orientation='Z', moment=1., mu=mu_0):
    """
        Calculate the vector potential of a set of magnetic dipoles
        at given locations 'ref. <http://en.wikipedia.org/wiki/Dipole#Magnetic_vector_potential>'

        .. math::

            B = \frac{\mu_0}{4 \pi r^3} \left( \frac{3 \vec{r} (\vec{m} \cdot
                                                                \vec{r})}{r^2})
                                                - \vec{m}
                                        \right) \cdot{\hat{rx}}

        :param numpy.ndarray srcLoc: Location of the source(s) (x, y, z)
        :param numpy.ndarray obsLoc: Where the potentials will be calculated
                                     (x, y, z)
        :param str component: The component to calculate - 'x', 'y', or 'z'
        :param numpy.ndarray moment: The vector dipole moment (vertical)
        :rtype: numpy.ndarray
        :return: The vector potential each dipole at each observation location
    """

    if isinstance(orientation, str):
        assert orientation.upper() in ['X', 'Y', 'Z'], (
            "orientation must be 'x', 'y', or 'z' or a vector not {}"
            .format(orientation)
        )
    elif (not np.allclose(np.r_[1., 0., 0.], orientation) or
          not np.allclose(np.r_[0., 1., 0.], orientation) or
          not np.allclose(np.r_[0., 0., 1.], orientation)):
        warnings.warn(
            'Arbitrary trasnmitter orientations ({}) not thouroughly tested '
            'Pull request on a test anyone? bueller?'.format(orientation)
        )

    if isinstance(component, str):
        assert component.upper() in ['X', 'Y', 'Z'], (
            "component must be 'x', 'y', or 'z' or a vector not {}"
            .format(component)
        )
    elif (not np.allclose(np.r_[1., 0., 0.], component) or
          not np.allclose(np.r_[0., 1., 0.], component) or
          not np.allclose(np.r_[0., 0., 1.], component)):
        warnings.warn(
            'Arbitrary receiver orientations ({}) not thouroughly tested '
            'Pull request on a test anyone? bueller?'
        ).format(component)

    if isinstance(orientation, str):
        orientation = orientationDict[orientation.upper()]

    if isinstance(component, str):
        component = orientationDict[component.upper()]

    assert np.linalg.norm(orientation, 2) == 1., (
        "orientation must be a unit vector. "
        "Use 'moment=X to scale source fields"
    )

    if np.linalg.norm(component, 2) != 1.:
        warnings.warn(
            'The magnitude of the receiver component vector is > 1, '
            ' it is {}. The receiver fields will be scaled.'
            .format(np.linalg.norm(component, 2))
        )

    srcLoc = np.atleast_2d(srcLoc)
    component = np.atleast_2d(component)
    obsLoc = np.atleast_2d(obsLoc)
    orientation = np.atleast_2d(orientation)

    nObs = obsLoc.shape[0]
    nSrc = int(srcLoc.size / 3.)

    # use outer product to construct an array of [x_src, y_src, z_src]

    m = moment*orientation.repeat(nObs, axis=0)
    B = []

    for i in range(nSrc):
        srcLoc = srcLoc[i, np.newaxis].repeat(nObs, axis=0)
        rx = component.repeat(nObs, axis=0)
        dR = obsLoc - srcLoc
        r = np.sqrt((dR**2).sum(axis=1))

        # mult each element and sum along the axis (vector dot product)
        m_dot_dR_div_r2 = (m * dR).sum(axis=1) / (r**2)

        # multiply the scalar m_dot_dR by the 3D vector r
        rvec_m_dot_dR_div_r2 = np.vstack([m_dot_dR_div_r2 * dR[:, i] for
                                          i in range(3)]).T
        inside = (3. * rvec_m_dot_dR_div_r2) - m

        # dot product with rx orientation
        inside_dot_rx = (inside * rx).sum(axis=1)
        front = (mu/(4.* pi * r**3))

        B.append(Utils.mkvc(front * inside_dot_rx))

    return np.vstack(B).T
