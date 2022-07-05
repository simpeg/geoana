#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include <math.h>

static PyMethodDef PrismMethods[] = {
    {NULL, NULL, 0, NULL}
};

/* The loop definition must precede the PyMODINIT_FUNC. */

static void double_prism_f(char **args, const npy_intp *dimensions,
                         const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *inx = args[0], *iny = args[1], *inz = args[2];
    char *out = args[3];
    npy_intp inx_step = steps[0], iny_step = steps[1], inz_step = steps[2];
    npy_intp out_step = steps[3];

    double x, y, z, v, r;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        x = *(double *)inx;
        y = *(double *)iny;
        z = *(double *)inz;
        v = 0.0;
        r = sqrt(x * x + y * y + z * z);
        if (x != 0.0){
            if (y != 0.0){
                v -= x * y * log(z + r);
            }
            v += 0.5 * x * x * atan( y * z / (x * r));
        }
        if (y != 0.0){
            if (z != 0.0){
                v -= y *z * log(x + r);
            }
            v += 0.5 * y * y * atan(z * x / (y * r));
        }
        if (z != 0.0){
            if (x != 0.0){
                v -= z * x * log(y + r);
            }
            v += 0.5 * z * z * atan(x * y / (z * r));
        }
        *((double *)out) = v;
        /* END main ufunc computation */

        inx += inx_step;
        iny += iny_step;
        inz += inz_step;
        out += out_step;
    }
}

/* This a pointer to the above function */
PyUFuncGenericFunction funcs_prism_f[1] = {&double_prism_f};

static void double_prism_fz(char **args, const npy_intp *dimensions,
                         const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *inx = args[0], *iny = args[1], *inz = args[2];
    char *out = args[3];
    npy_intp inx_step = steps[0], iny_step = steps[1], inz_step = steps[2];
    npy_intp out_step = steps[3];

    double x, y, z, v, r;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        x = *(double *)inx;
        y = *(double *)iny;
        z = *(double *)inz;
        v = 0.0;
        r = sqrt(x * x + y * y + z * z);
        if (x != 0.0){
            v += x * log(y + r);
        }
        if (y != 0.0){
            v += y * log(x + r);
        }
        if (z != 0.0){
            v -= z * atan(x * y / (z * r));
        }
        *((double *)out) = v;
        /* END main ufunc computation */

        inx += inx_step;
        iny += iny_step;
        inz += inz_step;
        out += out_step;
    }
}

/* This a pointer to the above function */
PyUFuncGenericFunction funcs_prism_fz[1] = {&double_prism_fz};

static void double_prism_fzz(char **args, const npy_intp *dimensions,
                         const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *inx = args[0], *iny = args[1], *inz = args[2];
    char *out = args[3];
    npy_intp inx_step = steps[0], iny_step = steps[1], inz_step = steps[2];
    npy_intp out_step = steps[3];

    double x, y, z, v, r;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        x = *(double *)inx;
        y = *(double *)iny;
        z = *(double *)inz;
        if (z != 0.0){
            r = sqrt(x * x + y * y + z * z);
            v = atan(x * y / (z * r));
        }else{
            v = 0.0;
        }
        *((double *)out) = v;
        /* END main ufunc computation */

        inx += inx_step;
        iny += iny_step;
        inz += inz_step;
        out += out_step;
    }
}

/* This a pointer to the above function */
PyUFuncGenericFunction funcs_prism_fzz[1] = {&double_prism_fzz};

static void double_prism_fzx(char **args, const npy_intp *dimensions,
                         const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *inx = args[0], *iny = args[1], *inz = args[2];
    char *out = args[3];
    npy_intp inx_step = steps[0], iny_step = steps[1], inz_step = steps[2];
    npy_intp out_step = steps[3];

    double x, y, z, v, r;
    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        x = *(double *)inx;
        y = *(double *)iny;
        z = *(double *)inz;
        r = sqrt(x * x + y * y + z * z);
        v = y + r;
        if (v == 0.0){
            if (y < 0){
                v = log(-2 * y);
            }
        }else{
            v = -log(v);
        }
        *((double *)out) = v;
        /* END main ufunc computation */

        inx += inx_step;
        iny += iny_step;
        inz += inz_step;
        out += out_step;
    }
}

/* This a pointer to the above function */
PyUFuncGenericFunction funcs_prism_fzx[1] = {&double_prism_fzx};

static void double_prism_fzy(char **args, const npy_intp *dimensions,
                         const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *inx = args[0], *iny = args[1], *inz = args[2];
    char *out = args[3];
    npy_intp inx_step = steps[0], iny_step = steps[1], inz_step = steps[2];
    npy_intp out_step = steps[3];

    double x, y, z, v, r;
    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        x = *(double *)inx;
        y = *(double *)iny;
        z = *(double *)inz;
        r = sqrt(x * x + y * y + z * z);
        v = x + r;
        if (v == 0.0){
            if (x < 0){
                v = log(-2 * x);
            }
        }else{
            v = -log(v);
        }
        *((double *)out) = v;
        /* END main ufunc computation */

        inx += inx_step;
        iny += iny_step;
        inz += inz_step;
        out += out_step;
    }
}

/* This a pointer to the above function */
PyUFuncGenericFunction funcs_prism_fzy[1] = {&double_prism_fzy};

static void double_prism_fzzz(char **args, const npy_intp *dimensions,
                         const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *inx = args[0], *iny = args[1], *inz = args[2];
    char *out = args[3];
    npy_intp inx_step = steps[0], iny_step = steps[1], inz_step = steps[2];
    npy_intp out_step = steps[3];

    double x, y, z, v, r, v1, v2;
    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        x = *(double *)inx;
        y = *(double *)iny;
        z = *(double *)inz;
        r = sqrt(x * x + y * y + z * z);
        v1 = x * x + z * z;
        v2 = y * y + z * z;
        v = 0.0;
        if (v1 != 0.0){
            v += 1.0/v1;
        }
        if (v2 != 0.0){
            v += 1.0/v2;
        }
        if (r != 0.0){
            v *= x * y / r;
        }

        *((double *)out) = v;
        /* END main ufunc computation */

        inx += inx_step;
        iny += iny_step;
        inz += inz_step;
        out += out_step;
    }
}

/* This a pointer to the above function */
PyUFuncGenericFunction funcs_prism_fzzz[1] = {&double_prism_fzzz};

static void double_prism_fxxy(char **args, const npy_intp *dimensions,
                         const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *inx = args[0], *iny = args[1], *inz = args[2];
    char *out = args[3];
    npy_intp inx_step = steps[0], iny_step = steps[1], inz_step = steps[2];
    npy_intp out_step = steps[3];

    double x, y, z, v, r;
    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        x = *(double *)inx;
        y = *(double *)iny;
        z = *(double *)inz;
        if (x != 0.0){
            v = x * x + y * y;
            r = sqrt(x * x + y * y + z * z);
            v = - x * z / (v * r);
        }else{
            v = 0.0;
        }
        *((double *)out) = v;
        /* END main ufunc computation */

        inx += inx_step;
        iny += iny_step;
        inz += inz_step;
        out += out_step;
    }
}

/* This a pointer to the above function */
PyUFuncGenericFunction funcs_prism_fxxy[1] = {&double_prism_fxxy};

static void double_prism_fxxz(char **args, const npy_intp *dimensions,
                         const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *inx = args[0], *iny = args[1], *inz = args[2];
    char *out = args[3];
    npy_intp inx_step = steps[0], iny_step = steps[1], inz_step = steps[2];
    npy_intp out_step = steps[3];

    double x, y, z, v, r;
    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        x = *(double *)inx;
        y = *(double *)iny;
        z = *(double *)inz;
        if (x != 0.0){
            v = x * x + z * z;
            r = sqrt(x * x + y * y + z * z);
            v = - x * y / (v * r);
        }else{
            v = 0.0;
        }
        *((double *)out) = v;
        /* END main ufunc computation */

        inx += inx_step;
        iny += iny_step;
        inz += inz_step;
        out += out_step;
    }
}

/* This a pointer to the above function */
PyUFuncGenericFunction funcs_prism_fxxz[1] = {&double_prism_fxxz};

static void double_prism_fxyz(char **args, const npy_intp *dimensions,
                         const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *inx = args[0], *iny = args[1], *inz = args[2];
    char *out = args[3];
    npy_intp inx_step = steps[0], iny_step = steps[1], inz_step = steps[2];
    npy_intp out_step = steps[3];

    double x, y, z, v, r;
    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        x = *(double *)inx;
        y = *(double *)iny;
        z = *(double *)inz;
        r = sqrt(x * x + y * y + z * z);
        if (r != 0.0){
            v = 1.0/r;
        }else{
            v = 0.0;
        }
        *((double *)out) = v;
        /* END main ufunc computation */

        inx += inx_step;
        iny += iny_step;
        inz += inz_step;
        out += out_step;
    }
}

/* This a pointer to the above function */
PyUFuncGenericFunction funcs_prism_fxyz[1] = {&double_prism_fxyz};

/* These are the input and return dtypes of the prism functions.*/
static char types[4] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "potential_field_prism",
    NULL,
    -1,
    PrismMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

static void *data[1] = {NULL,};

PyMODINIT_FUNC PyInit_potential_field_prism(void)
{
    PyObject *m, *d;
    PyObject *prism_f, *prism_fz, *prism_fzz, *prism_fzx, *prism_fzy;
    PyObject *prism_fzzz, *prism_fxxy, *prism_fxxz, *prism_fxyz;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    prism_f = PyUFunc_FromFuncAndData(funcs_prism_f, data, types, 1, 3, 1,
                                    PyUFunc_None, "prism_f",
"Evaluates the indefinite volume integral for the 1/r kernel.\n\n"
"This is used to evaluate the gravitational potential of dense prisms.\n\n"
"Parameters\n"
"----------\n"
"x, y, z : (...) numpy.ndarray\n"
"    The nodal locations to evaluate the function at\n\n"
"Returns\n"
"-------\n"
"(...) numpy.ndarray", 0);

    prism_fz = PyUFunc_FromFuncAndData(funcs_prism_fz, data, types, 1, 3, 1,
                                    PyUFunc_None, "prism_fz",
"Evaluates the indefinite volume integral for the d/dz * 1/r kernel.\n\n"
"This is used to evaluate the gravitational field of dense prisms.\n\n"
"Parameters\n"
"----------\n"
"x, y, z : (...) numpy.ndarray\n"
"    The nodal locations to evaluate the function at\n\n"
"Returns\n"
"-------\n"
"(...) numpy.ndarray\n\n"
"Notes\n"
"-----\n"
"Can be used to compute other components by cycling the inputs", 0);

    prism_fzz = PyUFunc_FromFuncAndData(funcs_prism_fzz, data, types, 1, 3, 1,
                                    PyUFunc_None, "prism_fzz",
"Evaluates the indefinite volume integral for the d**2/dz**2 * 1/r kernel.\n\n"
"This is used to evaluate the gravitational potential of dense prisms.\n\n"
"Parameters\n"
"----------\n"
"x, y, z : (...) numpy.ndarray\n"
"    The nodal locations to evaluate the function at\n\n"
"Returns\n"
"-------\n"
"  (...) numpy.ndarray\n\n"
"Notes\n"
"-----\n"
"Can be used to compute other components by cycling the inputs", 0);

    prism_fzx = PyUFunc_FromFuncAndData(funcs_prism_fzx, data, types, 1, 3, 1,
                                    PyUFunc_None, "prism_fzx",
"Evaluates the indefinite volume integral for the d**2/(dz*dx) * 1/r kernel.\n\n"
"This is used to evaluate the gravitational potential of dense prisms.\n\n"
"Parameters\n"
"----------\n"
"x, y, z : (...) numpy.ndarray\n"
"    The nodal locations to evaluate the function at\n\n"
"Returns\n"
"-------\n"
"(...) numpy.ndarray\n\n"
"Notes\n"
"-----\n"
"Can be used to compute other components by cycling the inputs", 0);

    prism_fzy = PyUFunc_FromFuncAndData(funcs_prism_fzy, data, types, 1, 3, 1,
                                    PyUFunc_None, "prism_fzy",
"Evaluates the indefinite volume integral for the d**2/(dz*dx) * 1/r kernel.\n\n"
"This is used to evaluate the gravitational potential of dense prisms.\n\n"
"Parameters\n"
"----------\n"
"x, y, z : (...) numpy.ndarray\n"
"    The nodal locations to evaluate the function at\n\n"
"Returns\n"
"-------\n"
"(...) numpy.ndarray\n\n"
"Notes\n"
"-----\n"
"Can be used to compute other components by cycling the inputs", 0);

    prism_fzzz = PyUFunc_FromFuncAndData(funcs_prism_fzzz, data, types, 1, 3, 1,
                                    PyUFunc_None, "prism_fzzz",
"Evaluates the indefinite volume integral for the d**3/(dz**3) * 1/r kernel.\n\n"
"This is used to evaluate the magnetic gradient of susceptible prisms.\n\n"
"Parameters\n"
"----------\n"
"x, y, z : (...) numpy.ndarray\n"
"    The nodal locations to evaluate the function at\n\n"
"Returns\n"
"-------\n"
"(...) numpy.ndarray\n\n"
"Notes\n"
"-----\n"
"Can be used to compute other components by cycling the inputs", 0);

    prism_fxxy = PyUFunc_FromFuncAndData(funcs_prism_fxxy, data, types, 1, 3, 1,
                                    PyUFunc_None, "prism_fxxy",
"Evaluates the indefinite volume integral for the d**3/(dx**2 * dy) * 1/r kernel.\n\n"
"This is used to evaluate the magnetic gradient of susceptible prisms.\n\n"
"Parameters\n"
"----------\n"
"x, y, z : (...) numpy.ndarray\n"
"    The nodal locations to evaluate the function at\n\n"
"Returns\n"
"-------\n"
"(...) numpy.ndarray\n\n"
"Notes\n"
"-----\n"
"Can be used to compute other components by cycling the inputs", 0);

    prism_fxxz = PyUFunc_FromFuncAndData(funcs_prism_fxxz, data, types, 1, 3, 1,
                                    PyUFunc_None, "prism_fxxz",
"Evaluates the indefinite volume integral for the d**3/(dx**2 * dz) * 1/r kernel.\n\n"
"This is used to evaluate the magnetic gradient of susceptible prisms.\n\n"
"Parameters\n"
"----------\n"
"x, y, z : (...) numpy.ndarray\n"
"    The nodal locations to evaluate the function at\n\n"
"Returns\n"
"-------\n"
"(...) numpy.ndarray\n\n"
"Notes\n"
"-----\n"
"Can be used to compute other components by cycling the inputs", 0);

    prism_fxyz = PyUFunc_FromFuncAndData(funcs_prism_fxyz, data, types, 1, 3, 1,
                                    PyUFunc_None, "prism_fxyz",
"Evaluates the indefinite volume integral for the d**3/(dx * dy * dz) * 1/r kernel.\n\n"
"This is used to evaluate the magnetic gradient of susceptible prisms.\n\n"
"Parameters\n"
"----------\n"
"x, y, z : (...) numpy.ndarray\n"
"    The nodal locations to evaluate the function at\n\n"
"Returns\n"
"-------\n"
"(...) numpy.ndarray\n\n"
"Notes\n"
"-----\n"
"Can be used to compute other components by cycling the inputs", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "prism_f", prism_f);
    Py_DECREF(prism_f);

    PyDict_SetItemString(d, "prism_fz", prism_fz);
    Py_DECREF(prism_fz);

    PyDict_SetItemString(d, "prism_fzz", prism_fzz);
    Py_DECREF(prism_fzz);

    PyDict_SetItemString(d, "prism_fzx", prism_fzx);
    Py_DECREF(prism_fzx);

    PyDict_SetItemString(d, "prism_fzy", prism_fzy);
    Py_DECREF(prism_fzy);

    PyDict_SetItemString(d, "prism_fzzz", prism_fzzz);
    Py_DECREF(prism_fzzz);

    PyDict_SetItemString(d, "prism_fxxy", prism_fxxy);
    Py_DECREF(prism_fxxy);

    PyDict_SetItemString(d, "prism_fxxz", prism_fxxz);
    Py_DECREF(prism_fxxz);

    PyDict_SetItemString(d, "prism_fxyz", prism_fxyz);
    Py_DECREF(prism_fxyz);

    return m;
}
