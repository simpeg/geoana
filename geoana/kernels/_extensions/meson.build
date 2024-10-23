
numpy_nodepr_api = ['-DNPY_NO_DEPRECATED_API=NPY_1_22_API_VERSION']

# NumPy include directory
np_dep = dependency('numpy')

# Deal with M_PI & friends; add `use_math_defines` to c_args or cpp_args
# Cython doesn't always get this right itself (see, e.g., gh-16800), so
# explicitly add the define as a compiler flag for Cython-generated code.
is_windows = host_machine.system() == 'windows'
if is_windows
  use_math_defines = ['-D_USE_MATH_DEFINES']
else
  use_math_defines = []
endif

c_undefined_ok = ['-Wno-maybe-uninitialized']

cython_args = []
if cy.version().version_compare('>=3.1.0')
  cython_args += ['-Xfreethreading_compatible=True']
endif

cython_c_args = [numpy_nodepr_api, use_math_defines]

cython_cpp_args = cython_c_args

module_path = 'geoana/kernels/_extensions'

py.extension_module(
    'potential_field_prism',
    'potential_field_prism.pyx',
    cython_args: cython_args,
    c_args: cython_c_args,
    install: true,
    subdir: module_path,
    dependencies : [py_dep, np_dep],
)

py.extension_module(
    'rTE',
    ['rTE.pyx', '_rTE.cpp'],
    cython_args: cython_args,
    cpp_args: cython_cpp_args,
    install: true,
    subdir: module_path,
    dependencies : [py_dep, np_dep],
    override_options : ['cython_language=cpp'],
)

python_sources = [
  '__init__.py',
]

py.install_sources(
  python_sources,
  subdir: module_path
)