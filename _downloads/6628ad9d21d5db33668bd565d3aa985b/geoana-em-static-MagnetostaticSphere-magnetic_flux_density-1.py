# Here, we define a sphere with permeability mu_sphere in a uniform magnetostatic field with permeability
# mu_background and plot the total and secondary magnetic flux densities.
#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from geoana.em.static import MagnetostaticSphere
#
# Define the sphere.
#
mu_sphere = 10. ** -1
mu_background = 10. ** -3
radius = 1.0
simulation = MagnetostaticSphere(
    location=None, mu_sphere=mu_sphere, mu_background=mu_background, radius=radius, primary_field=None
)
#
# Now we create a set of gridded locations and compute the magnetic flux densities.
#
X, Y = np.meshgrid(np.linspace(-2*radius, 2*radius, 20), np.linspace(-2*radius, 2*radius, 20))
Z = np.zeros_like(X) + 0.25
xyz = np.stack((X, Y, Z), axis=-1)
bt = simulation.magnetic_flux_density(xyz, field='total')
bs = simulation.magnetic_flux_density(xyz, field='secondary')
#
# Finally, we plot the total and secondary magnetic flux densities.
#
fig, axs = plt.subplots(1, 2, figsize=(18,12))
titles = ['Total Magnetic Flux Density', 'Secondary Magnetic Flux Density']
for ax, B, title in zip(axs.flatten(), [bt, bs], titles):
    B_amp = np.linalg.norm(B, axis=-1)
    im = ax.pcolor(X, Y, B_amp, shading='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(label= 'Amplitude (T)')
    ax.streamplot(X, Y, B[..., 0], B[..., 1], density=0.75)
    ax.add_patch(patches.Circle((0, 0), radius, fill=False, linestyle='--'))
    ax.set_ylabel('Y coordinate ($m$)')
    ax.set_xlabel('X coordinate ($m$)')
    ax.set_aspect('equal')
    ax.set_title(title)
plt.tight_layout()
plt.show()
