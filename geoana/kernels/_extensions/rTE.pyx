# distutils: language=c++
# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from libcpp.complex cimport complex, sqrt, exp
from libcpp cimport bool

ctypedef np.float64_t REAL_t
ctypedef np.intp_t SIZE_t
ctypedef np.complex128_t COMPLEX_t
ctypedef complex[double] complex_t

cdef extern from "_rTE.h" namespace "funcs":
    void rTE(
        complex_t *TE,
        double *frequencies,
        double *lambdas,
        complex_t *sigmas,
        double *mus,
        double *thicks,
        SIZE_t n_frequency,
        SIZE_t n_filter,
        SIZE_t n_layers
        ) nogil

    void rTEgrad(
        complex_t * TE_dsigma,
        complex_t * TE_dmu,
        complex_t * TE_dh,
        double * frequencies,
        double * lambdas,
        complex_t * sigmas,
        double * mus,
        double * h,
        SIZE_t n_frequency,
        SIZE_t n_filter,
        SIZE_t n_layers
    ) nogil

def rTE_forward(frequencies, lamb, sigma, mu, thicknesses):
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
    # Sigma and mu must be fortran contiguous
    sigma = np.require(sigma, dtype=np.complex128, requirements="F")
    mu = np.require(mu, dtype=np.float64, requirements="F")

    # These just must be contiguous
    lamb = np.require(lamb, dtype=np.float64, requirements="C")
    frequencies = np.require(frequencies, dtype=np.float64, requirements="C")
    thicknesses = np.require(thicknesses, dtype=np.float64, requirements="C")

    # Dimension checking
    if frequencies.shape[0] != sigma.shape[1]:
        raise ValueError(
            f"sigma array's last dimension must be the same as the frequency arrays first. Got {sigma.shape} and {frequencies.shape}"
        )
    if sigma.shape[0] != thicknesses.shape[0]+1:
        raise ValueError(
            f"sigma array's first dimension must match thickness array length+1. Got {sigma.shape[0]} and {thicknesses.shape[0]}"
        )
    if mu.shape != sigma.shape:
        raise ValueError(
            f"mu array must match sigma array shape. Got {mu.shape} and {sigma.shape}"
        )

    cdef:
        REAL_t[:] f = frequencies
        REAL_t[:] lam = lamb
        COMPLEX_t[:, :] sig = sigma
        REAL_t[:, :] c_mu = mu
        REAL_t[:] hs = thicknesses

    cdef:
        SIZE_t n_frequency, n_filter, n_layers


    n_frequency = f.shape[0]
    n_filter = lam.shape[0]
    n_layers = sig.shape[0]

    cdef COMPLEX_t[:, :] out = np.empty((n_frequency, n_filter), dtype=np.complex128, order="F")

    cdef:
        # Types are same size and both pairs of numbers (r, i), so can cast one to the other
        complex_t *out_p = <complex_t *> &out[0, 0]
        complex_t *sig_p = <complex_t *> &sig[0, 0]
        REAL_t *h_p = NULL
    if n_layers > 1:
        h_p = &hs[0]

    rTE(out_p, &f[0], &lam[0], sig_p, &c_mu[0, 0], h_p,
          n_frequency, n_filter, n_layers)

    return np.array(out)

def rTE_gradient(frequencies, lamb, sigma, mu, thicknesses):
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
    # Require that they are all the same ordering as sigma
    sigma = np.require(sigma, dtype=np.complex128, requirements="F")
    mu = np.require(mu, dtype=np.float64, requirements="F")
    lamb = np.require(lamb, dtype=np.float64, requirements="C")
    frequencies = np.require(frequencies, dtype=np.float64, requirements="C")
    thicknesses = np.require(thicknesses, dtype=np.float64, requirements="C")

    # Dimension checking
    if frequencies.shape[0] != sigma.shape[1]:
        raise ValueError(
            f"sigma array's last dimension must be the same as the frequency arrays first. Got {sigma.shape} and {frequencies.shape}"
        )
    if sigma.shape[0] != thicknesses.shape[0]+1:
        raise ValueError(
            f"sigma array's first dimension must match thickness array length+1. Got {sigma.shape[0]} and {thicknesses.shape[0]}"
        )
    if mu.shape != sigma.shape:
        raise ValueError(
            f"mu array must match sigma array shape. Got {mu.shape} and {sigma.shape}"
        )

    cdef:
        REAL_t[:] f = frequencies
        REAL_t[:] lam = lamb
        COMPLEX_t[:, :] sig = sigma
        REAL_t[:, :] c_mu = mu
        REAL_t[:] hs = thicknesses

    cdef:
        SIZE_t n_frequency, n_filter, n_layers


    n_frequency = f.shape[0]
    n_filter = lam.shape[0]
    n_layers = sig.shape[0]

    cdef COMPLEX_t[:, :, :] gsig = np.empty((n_layers, n_frequency, n_filter), dtype=np.complex128, order="F")
    cdef COMPLEX_t[:, :, :] gh = np.empty((n_layers-1, n_frequency, n_filter), dtype=np.complex128, order="F")
    cdef COMPLEX_t[:, :, :] gmu = np.empty((n_layers, n_frequency, n_filter), dtype=np.complex128, order="F")

    cdef:
        # Types are same size and both pairs of numbers (r, i), so can cast one to the other
        complex_t *sig_p = <complex_t *> &sig[0, 0]
        complex_t *gsig_p = <complex_t *> &gsig[0, 0, 0]
        complex_t *gmu_p = <complex_t *> &gmu[0, 0, 0]
        complex_t *gh_p = NULL
        REAL_t *h_p = NULL
    if n_layers > 1:
        h_p = &hs[0]
        gh_p = <complex_t *> &gh[0, 0, 0]

    rTEgrad(gsig_p, gmu_p, gh_p, &f[0], &lam[0], sig_p, &c_mu[0, 0], h_p,
          n_frequency, n_filter, n_layers)

    return np.array(gsig), np.array(gh), np.array(gmu)
