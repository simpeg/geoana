#define _USE_MATH_DEFINES
#include <cmath>
#include "_rTE.h"
#include <complex>
#include <vector>

void funcs::rTE(
    complex_t * TE,
    double * frequencies,
    double * lambdas,
    complex_t * sigmas,
    double * mus,
    double * h,
    size_t n_frequency,
    size_t n_filter,
    size_t n_layers
){
    double mu_0 = 4.0E-7*M_PI;
    complex_t j(0.0, 1.0);

    complex_t k2, u, Yh, Y, tanhuh, Y0;
    complex_t *sigi;
    double *mui;
    double omega, mu_c, l2;

    for(size_t i_filt=0, i=0; i_filt<n_filter; ++i_filt){
        l2 = lambdas[i_filt] * lambdas[i_filt];
        for(size_t i_freq=0; i_freq<n_frequency; ++i_freq, ++i){
            sigi = sigmas + i_freq*n_layers;
            mui = mus + i_freq*n_layers;
            omega = 2.0*M_PI*frequencies[i_freq];
            mu_c = mui[n_layers-1];

            k2 = -j * omega * mu_c * sigi[n_layers-1];
            u = std::sqrt(l2 - k2);
            Yh = u/(j * omega * mu_c);
            for(size_t k=n_layers-2; k<n_layers-1; --k){
                mu_c = mui[k];
                k2 = -j * omega * mu_c * sigi[k];
                u = std::sqrt(l2 - k2);
                Y = u / (j * omega * mu_c);
                tanhuh = std::tanh(u * h[k]);
                Yh = Y * (Yh + Y*tanhuh)/(Y + Yh*tanhuh);
            }
            Y0 = lambdas[i_filt]/(j*omega*mu_0);
            TE[i] = (Y0 - Yh)/(Y0 + Yh);
        }
    }
}

void funcs::rTEgrad(
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
){
    double mu_0 = 4.0E-7*M_PI;
    complex_t j(0.0, 1.0);

    //complex_t k2, u, Yh, Y, tanhuh, Y0;
    double omega, l2;

    // storing intermediate values allows us to accelerate reverse mode.
    // only really "need" to store the recursive variable,
    // the others could be recalculated (with extra computing cost)
    // but want to avoid calculating sqrt and tanh wherever possible
    vec_complex_t k2s(n_layers), us(n_layers), Yhs(n_layers);
    vec_complex_t Ys(n_layers-1), tanhs(n_layers-1);

    double *mui;
    complex_t *sigi, *TE_dhi, *TE_dmui, *TE_dsigmai;
    complex_t gyh0, bot, gy, gtanh, Y0;
    complex_t gu, gmu, gk2;

    for(size_t i_filt=0, i=0; i_filt<n_filter; ++i_filt){
        l2 = lambdas[i_filt] * lambdas[i_filt];
        for(size_t i_freq=0; i_freq<n_frequency; ++i_freq, ++i){
            sigi = sigmas + i_freq*n_layers;
            mui = mus + i_freq*n_layers;
            if (TE_dh != NULL){
                TE_dhi = TE_dh + i*(n_layers-1);
            }
            TE_dmui = TE_dmu + i*n_layers;
            TE_dsigmai = TE_dsigma + i*n_layers;
            omega = 2.0*M_PI*frequencies[i_freq];

            k2s[n_layers-1] = -j * omega * mui[n_layers-1] * sigi[n_layers-1];
            us[n_layers-1] = std::sqrt(l2 - k2s[n_layers-1]);
            Yhs[n_layers-1] = us[n_layers-1]/(j * omega * mui[n_layers-1]);

            for(size_t k=n_layers-2; k<n_layers-1; --k){
                k2s[k] = -j * omega * mui[k] * sigi[k];
                us[k] = std::sqrt(l2 - k2s[k]);
                Ys[k] = us[k] / (j * omega * mui[k]);
                tanhs[k] = std::tanh(us[k] * h[k]);
                Yhs[k] = Ys[k] * (Yhs[k+1] + Ys[k]*tanhs[k])/(Ys[k] + Yhs[k+1]*tanhs[k]);
            }
            Y0 = lambdas[i_filt]/(j*omega*mu_0);
            // TE = (Y0 - Yhs[0])/(Y0 + Yhs[0]);

            // reverse through to back propagate the derivatives
            // general formula for an operation like v4 = v1 + v2*v3
            // gv1 = gv4 * d(v4)/d(v1)
            // gv2 = gv4 * d(v4)/d(v2)
            // gv3 = gv4 * d(v4)/d(v3)
            // and so on
            // if a variable appears in the right hand side of more than one operation
            // just accumulate onto that variables "g"
            // starting with gTE = 1.0;
            gyh0 = -2.0*Y0/((Y0 + Yhs[0])*(Y0 + Yhs[0])); // * gTE
            for(size_t k=0; k<n_layers-1; ++k){
                bot = (Ys[k] + Yhs[k+1]*tanhs[k])*(Ys[k] + Yhs[k+1]*tanhs[k]);
                gy = gyh0 * tanhs[k]*(2.0*tanhs[k]*Ys[k]*Yhs[k+1] + Ys[k]*Ys[k] + Yhs[k+1]*Yhs[k+1])/bot;
                gtanh = gyh0 * (Ys[k]*Ys[k]*Ys[k] - Ys[k]*Yhs[k+1]*Yhs[k+1])/bot;
                gyh0 = gyh0 * -(tanhs[k]*tanhs[k] - 1.0)*Ys[k]*Ys[k]/bot;

                TE_dhi[k] = gtanh * us[k] * (1.0 - tanhs[k]*tanhs[k]);
                gu = gtanh * h[k] * (1.0 - tanhs[k]*tanhs[k]);

                gu += gy/(j * omega * mui[k]);

                gmu = gy * (-Ys[k]/mui[k]);
                gk2 = gu * -0.5 / us[k];

                gmu -= gk2 * (j * omega * sigi[k]);
                TE_dsigmai[k] = gk2 * (-j * omega * mui[k]);
                TE_dmui[k] = gmu;
            }
            gu = gyh0 / (j * omega * mui[n_layers-1]);
            gmu = gyh0 *(-Yhs[n_layers-1]/mui[n_layers-1]);
            gk2 = gu * -0.5 / us[n_layers-1];
            gmu += gk2 * -j * omega * sigi[n_layers-1];

            TE_dsigmai[n_layers-1] = gk2 * -j * omega * mui[n_layers-1];
            TE_dmui[n_layers-1] = gmu;
        }
    }
}
