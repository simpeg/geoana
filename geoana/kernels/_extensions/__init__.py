try:
    # register numba jitable versions of the prism functions
    # if numba is available (and this module is installed).
    from numba.extending import (
        overload,
        get_cython_function_address
    )
    from numba import types
    import ctypes

    from .potential_field_prism import (
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
    funcs = [
        prism_f,
        prism_fz,
        prism_fzz,
        prism_fzx,
        prism_fzy,
        prism_fzzz,
        prism_fxxy,
        prism_fxxz,
        prism_fxyz,
    ]

    def _numba_register_prism_func(prism_func):
        module = 'geoana.kernels._extensions.potential_field_prism'
        name = prism_func.__name__

        func_address = get_cython_function_address(module, name)
        func_type = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
        c_func = func_type(func_address)

        @overload(prism_func)
        def numba_func(x, y, z):
            if isinstance(x, types.Float):
                if isinstance(y, types.Float):
                    if isinstance(z, types.Float):
                        def f(x, y, z):
                            return c_func(x, y, z)
                        return f
    for func in funcs:
        _numba_register_prism_func(func)

except ImportError as err:
    pass