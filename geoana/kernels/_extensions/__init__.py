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

    from .potential_field_polyhedron import (
        polyhedron_edge_log,
        triangle_solid_angle,
        face_edge_solid_angle,
    )

    def _numba_register_polyhedron_func(func, n_args):
        module = 'geoana.kernels._extensions.potential_field_polyhedron'
        func_address = get_cython_function_address(module, func.__name__)
        func_type = ctypes.CFUNCTYPE(ctypes.c_double, *([ctypes.c_double] * n_args))
        c_func = func_type(func_address)

        @overload(func)
        def numba_func(*args):
            if len(args) == n_args and all(isinstance(a, types.Float) for a in args):
                def f(*args):
                    return c_func(*args)
                return f

    _numba_register_polyhedron_func(polyhedron_edge_log, 6)
    _numba_register_polyhedron_func(triangle_solid_angle, 9)
    _numba_register_polyhedron_func(face_edge_solid_angle, 10)

except ImportError as err:
    pass