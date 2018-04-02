"""
Electric Dipole in a Whole Space: Frequency Domain
==================================================

In this example, we plot electric and magnetic flux density due to an electric
dipole in a whole space. Note that you can also examine the current density
and magnetic field.

We can vary the conductivity, magnetic permeability and dielectric permittivity
of the wholespace, the frequency of the source and whether or not the
quasistatic assumption is imposed.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.constants import mu_0, epsilon_0

from geoana import utils, spatial
from geoana.em import fdem

# define frequencies that we want to look at
frequencies = np.logspace(0, 4, 3)

# Build the electric dipole object
edipole = fdem.ElectricDipoleWholeSpace(
    sigma=1.,  # conductivity of 1 S/m
    mu=mu_0,  # permeability of free space (this is the default)
    epsilon=epsilon_0,  # permittivity of free space (this is the default)
    location=np.r_[0., 0., 0.],  # location of the dipole
    orientation='Z',  # vertical dipole (can also be a unit-vector)
    quasistatic=False  # don't use the quasistatic assumption
)

# construct a grid where we want to plot electric fields
x = np.linspace(-50, 50, 100)
z = np.linspace(-50, 50, 100)
xyz = utils.ndgrid([x, np.r_[0], z])


# plot amplitude
def plot_amplitude(ax, v):
    v = spatial.vector_magnitude(v)
    plt.colorbar(
        ax.pcolormesh(
            x, z, v.reshape(len(x), len(z), order='F'), norm=LogNorm()
        ), ax=ax
    )
    ax.axis('square')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')


# plot streamlines
def plot_streamlines(ax, v):
    vx = v[:, 0].reshape(len(x), len(z), order='F')
    vz = v[:, 2].reshape(len(x), len(z), order='F')
    ax.streamplot(x, z, vx.T, vz.T, color='k')


# create fig, ax for electric fields and magnetic flux
fig_e, ax_e = plt.subplots(
    2, len(frequencies), figsize=(5*len(frequencies), 7)
)
fig_b, ax_b = plt.subplots(
    2, len(frequencies), figsize=(5*len(frequencies), 7)
)

# loop over frequencies and plot
for i, frequency in enumerate(frequencies):

    # set the frequency of the dipole
    edipole.frequency = frequency

    # evaluate the electric field and magnetic flux density
    electric_field = edipole.electric_field(xyz)
    magnetic_flux_density = edipole.magnetic_flux_density(xyz)

    # plot amplitude of electric field
    for ax, reim in zip(ax_e[:, i], ['real', 'imag']):
        # grab real or imag component
        e_plot = getattr(electric_field, reim)

        # plot both amplitude and streamlines
        plot_amplitude(ax, e_plot)
        plot_streamlines(ax, e_plot)

        # set the title
        ax.set_title(
            'E {} at {:1.1e} Hz'.format(reim, frequency)
        )

    # plot the amplitude of the magnetic field (note the magnetic field is into
    # and out of the page in this geometry, so we don't plot vectors)
    for ax, reim in zip(ax_b[:, i], ['real', 'imag']):
        # grab real or imag component
        b_plot = getattr(magnetic_flux_density, reim)

        # plot amplitude
        plot_amplitude(ax, b_plot)

        # set the title
        ax.set_title(
            'B {} at {:1.1e} Hz'.format(reim, frequency)
        )

# format so text doesn't overlap
fig_e.tight_layout()
fig_b.tight_layout()
plt.show()


