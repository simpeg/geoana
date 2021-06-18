from geoana.em.fdem.base import (
    omega, wavenumber, skin_depth, sigma_hat, BaseFDEM
)

from geoana.em.fdem.wholespace import (
    ElectricDipoleWholeSpace, MagneticDipoleWholeSpace
)

from geoana.em.fdem.halfspace import MagneticDipoleHalfSpace

from geoana.em.fdem.layered import MagneticDipoleLayeredHalfSpace

from geoana.em.fdem.simple_functions import (
    vertical_magnetic_field_horizontal_loop, vertical_magnetic_flux_horizontal_loop
)
