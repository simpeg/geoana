#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include <math.h>
#include <stdio.h>

static PyMethodDef PrismMethods[] = {
    {NULL, NULL, 0, NULL}
};

static void double_fnode(char **args, const npy_intp *dimensions,
                             const npy_intp *steps, void *data)
{
    printf("Here1");
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1], *in3 = args[2];
    char *out = args[3];
    npy_intp in1_step = steps[0], in2_step = steps[1], in3_step = steps[2];
    npy_intp out_step = steps[3];
    printf("Here2");

    double r;
    double x;
    double y;
    double z;
    double v;
    printf("Here3");

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        x = *(double *)in1;
        y = *(double *)in2;
        z = *(double *)in3;
        r = sqrt(x * x + y * y + z * z);

        v = 0.0;
        if(x != 0.0) {
            if(y != 0.0){
                v -= x * y * log(z + r);
            }
            v += 0.5 * x * x * atan(y * z / (x * r));
        }
        if(y != 0.0) {
            if(z != 0.0){
                v -= y * z * log(x + r);
            }
            v += 0.5 * y * y * atan(z * x / (y * r));
        }
        if(z != 0.0) {
            if(x != 0.0){
                v -= z * x * log(y + r);
            }
            v += 0.5 * z * z * atan(x * y / (z * r));
        }
        *((double *)out) = v;
        /* END main ufunc computation */

        in1 += in1_step;
        in2 += in2_step;
        in3 += in3_step;
        out += out_step;
    }
};

/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&double_fnode};

/* These are the input and return dtypes of fnode.*/

static char types[4] = {NPY_DOUBLE, NPY_DOUBLE,
                        NPY_DOUBLE, NPY_DOUBLE};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "potential_prism",
    NULL,
    -1,
    PrismMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_potential_prism(void)
{
    PyObject *m, *prism_f, *d;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    prism_f = PyUFunc_FromFuncAndData(funcs, NULL, types, 1, 3, 1,
                                    PyUFunc_None, "prism_f",
                                    "prism_f_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "prism_f", prism_f);
    Py_DECREF(prism_f);

    return m;
}
