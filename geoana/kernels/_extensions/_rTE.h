#ifndef __TE_H
#define __TE_H

#include <complex>
#include <vector>

typedef std::complex<double> complex_t;
typedef std::size_t size_t;
typedef std::vector<complex_t> vec_complex_t;

namespace funcs {
    void rTE(
        complex_t *TE,
        double *frequencies,
        double *lambdas,
        complex_t *sigmas,
        double *mus,
        double *depths,
        size_t n_frequency,
        size_t n_filter,
        size_t n_layers
    );

    void rTEgrad(
        complex_t * TE_dsigma,
        complex_t * TE_dmu,
        complex_t * TE_dh,
        double * frequencies,
        double * lambdas,
        complex_t * sigmas,
        double * mus,
        double * h,
        size_t n_frequency,
        size_t n_filter,
        size_t n_layers
    );
}

#endif
