# Here, we define a sphere with conductivity sigma_sphere in a uniform electrostatic field with conductivity
# sigma_background and plot the total and secondary electric potentials.
#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from geoana.em.static import ElectrostaticSphere
#
# Define the sphere.
#
sigma_sphere = 10. ** -1
sigma_background = 10. ** -3
radius = 1.0
simulation = ElectrostaticSphere(
    location=None, sigma_sphere=sigma_sphere, sigma_background=sigma_background, radius=radius, primary_field=None
)
#
# Now we create a set of gridded locations and compute the electric potentials.
#
X, Y = np.meshgrid(np.linspace(-2*radius, 2*radius, 20), np.linspace(-2*radius, 2*radius, 20))
Z = np.zeros_like(X) + 0.25
xyz = np.stack((X, Y, Z), axis=-1)
vt = simulation.potential(xyz, field='total')
vs = simulation.potential(xyz, field='secondary')
#
# Finally, we plot the total and secondary electric potentials.
#
fig, axs = plt.subplots(1, 2, figsize=(18,12))
titles = ['Total Potential', 'Secondary Potential']
for ax, V, title in zip(axs.flatten(), [vt, vs], titles):
    im = ax.pcolor(X, Y, V, shading='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(label= 'Potential (V)')
    ax.add_patch(patches.Circle((0, 0), radius, fill=False, linestyle='--'))
    ax.set_ylabel('Y coordinate ($m$)')
    ax.set_xlabel('X coordinate ($m$)')
    ax.set_title(title)
    ax.set_aspect('equal')
plt.tight_layout()
plt.show()