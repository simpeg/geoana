# Here, we define a point current with current=1A in a halfspace and plot the electric
# field lines in the xy-plane.
#
import numpy as np
import matplotlib.pyplot as plt
from geoana.em.static import PointCurrentHalfSpace
#
# Define the point current.
#
rho = 1.0
current = 1.0
simulation = PointCurrentHalfSpace(
    current=current, rho=rho, location=None
)
#
# Now we create a set of gridded locations and compute the electric field.
#
X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
Z = np.zeros_like(X)
xyz = np.stack((X, Y, Z), axis=-1)
e = simulation.electric_field(xyz)
#
# Finally, we plot the electric field lines.
#
e_amp = np.linalg.norm(e, axis=-1)
plt.pcolor(X, Y, e_amp)
cb = plt.colorbar()
cb.set_label(label= 'Amplitude ($V/m$)')
plt.streamplot(X, Y, e[..., 0], e[..., 1], density=0.50)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Electric Field Lines for a Point Current in a Halfspace')
plt.show()
