# Here, we define a dipole source in a halfspace to compute electric field.
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
# Now we create a set of gridded locations and compute the electric field.
#
X, Y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
Z = np.zeros_like(X)
xyz = np.stack((X, Y, Z), axis=-1)
e1 = simulation.electric_field(xyz)
e2 = simulation.electric_field(xyz - np.r_[2, 0, 0], xyz + np.r_[2, 0, 0])
#
# Finally, we plot the electric field.
#
fig, axs = plt.subplots(1, 2, figsize=(18,12))
titles = ['3 Electrodes', '4 Electrodes']
for ax, E, title in zip(axs.flatten(), [e1, e2], titles):
    E_amp = np.linalg.norm(E, axis=-1)
    im = ax.pcolor(X, Y, E_amp, shading='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(label= 'Amplitude ($V/m$)')
    ax.streamplot(X, Y, E[..., 0], E[..., 1], density=0.75)
    ax.set_ylabel('Y coordinate ($m$)')
    ax.set_xlabel('X coordinate ($m$)')
    ax.set_aspect('equal')
    ax.set_title(title)
#
# Finally, we plot the electric field.
#
E_amp = np.linalg.norm(e1, axis=-1)
plt.pcolor(X, Y, E_amp, shading='auto')
cb = plt.colorbar()
cb.set_label(label= 'Electric Field ($V/m$)')
plt.streamplot(X, Y, e1[..., 0], e1[..., 1], density=0.75)
plt.ylabel('Y coordinate ($m$)')
plt.xlabel('X coordinate ($m$)')
plt.title('Electric Field from Dipole using 3 Electrodes')
#
plt.tight_layout()
plt.show()
