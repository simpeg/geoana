# Here, we define a dipole source in a halfspace to compute potential.
#
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from geoana.em.static import DipoleHalfSpace
#
# Define the dipole source.
#
rho = 1.0
current = 1.0
location_a = np.r_[-1, 0, 0]
location_b = np.r_[1, 0, 0]
simulation = DipoleHalfSpace(
    current=current, rho=rho, location_a=location_a, location_b=location_b
)
#
# Now we create a set of gridded locations and compute the electric potential.
#
X, Y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
Z = np.zeros_like(X)
xyz = np.stack((X, Y, Z), axis=-1)
v1 = simulation.potential(xyz)
v2 = simulation.potential(xyz - np.r_[2, 0, 0], xyz + np.r_[2, 0, 0])
#
# Finally, we plot the electric potential.
#
fig, axs = plt.subplots(1, 2, figsize=(18,12))
titles = ['3 Electrodes', '4 Electrodes']
for ax, V, title in zip(axs.flatten(), [v1, v2], titles):
    im = ax.pcolor(X, Y, V, shading='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(label= 'Potential (V)')
    ax.set_ylabel('Y coordinate ($m$)')
    ax.set_xlabel('X coordinate ($m$)')
    ax.set_aspect('equal')
    ax.set_title(title)
plt.tight_layout()
plt.show()
