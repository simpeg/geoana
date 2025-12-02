# Here, we define a sphere with permeability mu_sphere in a uniform magnetostatic field with permeability
# mu_background and plot the total and secondary magnetic fields.
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
# Now we create a set of gridded locations and compute the magnetic fields.
#
X, Y = np.meshgrid(np.linspace(-2*radius, 2*radius, 20), np.linspace(-2*radius, 2*radius, 20))
Z = np.zeros_like(X) + 0.25
xyz = np.stack((X, Y, Z), axis=-1)
ht = simulation.magnetic_field(xyz, field='total')
hs = simulation.magnetic_field(xyz, field='secondary')
#
# Finally, we plot the total and secondary magnetic fields.
#
fig, axs = plt.subplots(1, 2, figsize=(18,12))
titles = ['Total Magnetic Field', 'Secondary Magnetic Field']
for ax, H, title in zip(axs.flatten(), [ht, hs], titles):
    H_amp = np.linalg.norm(H, axis=-1)
    im = ax.pcolor(X, Y, H_amp, shading='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(label= 'Amplitude ($A/m$)')
    ax.streamplot(X, Y, H[..., 0], H[..., 1], density=0.75)
    ax.add_patch(patches.Circle((0, 0), radius, fill=False, linestyle='--'))
    ax.set_ylabel('Y coordinate ($m$)')
    ax.set_xlabel('X coordinate ($m$)')
    ax.set_aspect('equal')
    ax.set_title(title)
plt.tight_layout()
plt.show()
