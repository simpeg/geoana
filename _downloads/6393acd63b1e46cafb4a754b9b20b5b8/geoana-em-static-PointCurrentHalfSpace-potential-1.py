# Here, we define a point current with current=1A in a halfspace and plot the electric
# potential.
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
# Now we create a set of gridded locations and compute the electric potential.
#
X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
Z = np.zeros_like(X)
xyz = np.stack((X, Y, Z), axis=-1)
v = simulation.potential(xyz)
#
# Finally, we plot the electric potential.
#
plt.pcolor(X, Y, v)
cb1 = plt.colorbar()
cb1.set_label(label= 'Potential (V)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Electric Potential from Point Current in a Halfspace')
plt.show()
