import numpy as np
from scipy.constants import mu_0

def _rTE_forward(frequencies, lamb, sigma, mu, thicknesses):
    """Compute reflection coefficients for Transverse Electric (TE) mode.

    The first layer is considered to be the top most layer. The last
    layer is considered to have infinite thickness. All physical properties 
    are defined starting from the top most layer. 

    Parameters
    ----------
    frequencies : float, numpy.ndarray
        Frequency (Hz); shape = (n_frequency, ).
    lamb : float, numpy.ndarray
        Spatial wavenumber (1/m), shape = (n_filter, ).
    sigma: complex, numpy.ndarray
        Conductivity (S/m), shape = (n_layer, n_frequency).
    mu: complex, numpy.ndarray
        Magnetic permeability (H/m), shape = (n_layer, n_frequency).
    thicknesses: float, numpy.ndarray
        Thickness (m) of each layer, shape = (n_layer-1,).

    Returns
    -------
    rTE: complex numpy.ndarray
        Reflection coefficients, shape = (n_frequency, n_filter)
    """
    n_layer = len(thicknesses)+1

    omega = 2*np.pi*frequencies
    l2 = lamb**2

    k2 = -1j * omega * mu * sigma
    u = np.sqrt(l2 - k2[:, :, None])
    Y = u/(1j*omega*mu)[:, :, None]
    tanh = np.tanh(u[:-1]*thicknesses[:, None, None])
    Yh = Y[-1]
    for k in range(n_layer-2, -1, -1):
        Yh = Y[k] * (Yh + Y[k]*tanh[k])/(Y[k] + Yh*tanh[k])
    # conductivity and susceptibility in air layer is 0
    Y0 = lamb/(1j*omega*mu_0)[:, None]
    TE = (Y0 - Yh)/(Y0 + Yh)

    return TE

def _rTE_gradient(frequencies, lamb, sigma, mu, thicknesses):
    """Compute reflection coefficients for Transverse Electric (TE) mode.

    The first layer is considered to be the top most layer. The last
    layer is considered to have infinite thickness. All physical properties 
    are defined starting from the top most layer. 

    Parameters
    ----------
    frequencies : float, numpy.ndarray
        Frequency (Hz); shape = (n_frequency, ).
    lamb : float, numpy.ndarray
        Spatial wavenumber (1/m), shape = (n_filter, ).
    sigma: complex, numpy.ndarray
        Conductivity (S/m), shape = (n_layer, n_frequency).
    mu: complex, numpy.ndarray
        Magnetic permeability (H/m), shape = (n_layer, n_frequency).
    thicknesses: float, numpy.ndarray
        Thickness (m) of each layer, shape = (n_layer-1,).

    Returns
    -------
    rTE_dsigma: complex numpy.ndarray
        Reflection coefficients gradient w.r.t. conductivity
        shape = (n_layer, n_frequency, n_filter)
    rTE_dh: complex numpy.ndarray
        Reflection coefficients gradient w.r.t. thicknesses
        shape = (n_layer-1, n_frequency, n_filter)
    rTE_dmu: complex numpy.ndarray
        Reflection coefficients gradient w.r.t. magnetic permeability
        shape = (n_layer, n_frequency, n_filter)
    """
    n_frequency = len(frequencies)
    n_filter = len(lamb)
    n_layer = len(thicknesses)+1

    omega = 2*np.pi*frequencies
    l2 = lamb**2

    k2 = -1j * omega * mu * sigma
    u = np.sqrt(l2 - k2[:, :, None])
    Y = u/(1j*omega*mu)[:, :, None]
    tanh = np.tanh(u[:-1]*thicknesses[:, None, None])
    Yh = np.empty((n_layer, n_frequency, n_filter), dtype=np.complex128)
    Yh[-1] = Y[-1]
    for k in range(n_layer-2, -1, -1):
        Yh[k] = Y[k] * (Yh[k+1] + Y[k]*tanh[k])/(Y[k] + Yh[k+1]*tanh[k])
    Y0 = lamb/(1j*omega*mu_0)[:, None]
    # TE = (Y0 - Yh[0])/(Y0 + Yh[0])

    rTE_dsigma = np.empty((n_layer, n_frequency, n_filter), dtype=np.complex128)
    rTE_dh = np.empty((n_layer-1, n_frequency, n_filter), dtype=np.complex128)
    rTE_dmu = np.empty((n_layer, n_frequency, n_filter), dtype=np.complex128)

    gyh0 = -2.0*Y0/((Y0 + Yh[0])*(Y0 + Yh[0]))
    for k in range(n_layer-1):
        bot = (Y[k] + Yh[k+1]*tanh[k])*(Y[k] + Yh[k+1]*tanh[k])
        gy = gyh0 * tanh[k]*(2.0*tanh[k]*Y[k]*Yh[k+1] + Y[k]*Y[k] + Yh[k+1]*Yh[k+1])/bot
        gtanh = gyh0 * (Y[k]*Y[k]*Y[k] - Y[k]*Yh[k+1]*Yh[k+1])/bot
        gyh0 = gyh0 * -(tanh[k]*tanh[k] - 1.0)*Y[k]*Y[k]/bot

        rTE_dh[k] = gtanh * u[k] * (1.0 - tanh[k]*tanh[k])
        gu = gtanh * thicknesses[k] * (1.0 - tanh[k]*tanh[k])

        gu += gy/(1j * omega * mu[k])[:, None]

        gmu = gy * (-Y[k]/mu[k][:, None])
        gk2 = gu * -0.5 / u[k]

        gmu -= gk2 * (1j * omega * sigma[k])[:, None]
        rTE_dsigma[k] = gk2 * (-1j * omega * mu[k])[:, None]
        rTE_dmu[k] = gmu
    gu = gyh0 / (1j * omega * mu[-1])[:, None]
    gmu = gyh0 * (-Yh[-1]/mu[-1][:, None])
    gk2 = gu * -0.5 / u[-1]
    gmu += gk2 * (-1j * omega * sigma[-1])[:, None]

    rTE_dsigma[-1] = gk2 * (-1j * omega * mu[-1])[:, None]
    rTE_dmu[-1] = gmu

    return rTE_dsigma, rTE_dh, rTE_dmu


try:
    from geoana.kernels._extensions.rTE import rTE_forward, rTE_gradient
except ImportError:
    # Store the above as the kernels
    rTE_forward = _rTE_forward
    rTE_gradient = _rTE_gradient
