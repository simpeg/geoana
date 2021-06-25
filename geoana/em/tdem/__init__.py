from geoana.em.tdem.base import peak_time, diffusion_distance, theta, BaseTDEM

from geoana.em.tdem.wholespace import ElectricDipoleWholeSpace

from geoana.em.tdem.halfspace import VerticalMagneticDipoleHalfSpace

from geoana.em.tdem.simple_functions import (
    vertical_magnetic_field_horizontal_loop,
    vertical_magnetic_flux_horizontal_loop,
    vertical_magnetic_field_time_deriv_horizontal_loop,
    vertical_magnetic_flux_time_deriv_horizontal_loop,
    magnetic_field_vertical_magnetic_dipole,
    magnetic_field_time_deriv_magnetic_dipole,
    magnetic_flux_vertical_magnetic_dipole,
    magnetic_flux_time_deriv_magnetic_dipole,
)
