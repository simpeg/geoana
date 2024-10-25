# Reproducing part of Figure 4.4 and 4.5 from Ward and Hohmann 1988
#
import numpy as np
import matplotlib.pyplot as plt
from geoana.em.tdem import magnetic_field_vertical_magnetic_dipole
#
# Calculate the field at the time given, and 100 m along the x-axis,
#
times = np.logspace(-8, 0, 200)
xy = np.array([[100, 0, 0]])
h = magnetic_field_vertical_magnetic_dipole(times, xy, sigma=1E-2)
#
# Match the vertical magnetic field plot
#
plt.loglog(times*1E3, h[:,0, 2], c='C0', label='$h_z$')
plt.loglog(times*1E3, -h[:,0, 2], '--', c='C0')
plt.loglog(times*1E3, h[:,0, 0], c='C1', label='$h_x$')
plt.loglog(times*1E3, -h[:,0, 0], '--', c='C1')
plt.xlabel('time (ms)')
plt.ylabel('h (A/m)')
plt.legend()
plt.show()
