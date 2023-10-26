"""This module contains kernals for (semi-)analytic geophysical responses
"""
from geoana.kernels.tranverse_electric_reflections import rTE_forward, rTE_gradient
from geoana.kernels.potential_field_prism import (
    prism_f,
    prism_fz,
    prism_fzz,
    prism_fzx,
    prism_fzy,
    prism_fzzz,
    prism_fxxy,
    prism_fxxz,
    prism_fxyz,
)

try:
    from numba.extending import get_cython_function_address
    import ctypes

    def __as_ctypes_func(module, function, argument_types):
        func_address = get_cython_function_address(module, function)
        func_type = ctypes.CFUNCTYPE(*argument_types)
        return func_type(func_address)

    c_prism = __as_ctypes_func(
        'geoana.kernels._extensions.potential_field_prism',
        'prism_f',
        (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
    )
    c_prism_fz = __as_ctypes_func(
        'geoana.kernels._extensions.potential_field_prism',
        'prism_fz',
        (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
    )
    c_prism_fzz = __as_ctypes_func(
        'geoana.kernels._extensions.potential_field_prism',
        'prism_fzz',
        (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
    )
    c_prism_fzx = __as_ctypes_func(
        'geoana.kernels._extensions.potential_field_prism',
        'prism_fzx',
        (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
    )
    c_prism_fzy = __as_ctypes_func(
        'geoana.kernels._extensions.potential_field_prism',
        'prism_fzy',
        (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
    )
    c_prism_fzzz = __as_ctypes_func(
        'geoana.kernels._extensions.potential_field_prism',
        'prism_fzzz',
        (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
    )
    c_prism_fxxy = __as_ctypes_func(
        'geoana.kernels._extensions.potential_field_prism',
        'prism_fxxy',
        (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
    )
    c_prism_fxxz = __as_ctypes_func(
        'geoana.kernels._extensions.potential_field_prism',
        'prism_fxxz',
        (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
    )
    c_prism_fxyz = __as_ctypes_func(
        'geoana.kernels._extensions.potential_field_prism',
        'prism_fxyz',
        (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
    )

except ImportError:
    pass
