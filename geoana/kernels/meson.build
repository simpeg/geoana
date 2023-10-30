
python_sources = [
  '__init__.py',
  'potential_field_prism.py',
  'tranverse_electric_reflections.py',
]

py.install_sources(
  python_sources,
  subdir: 'geoana/kernels'
)

pure_python = get_option('pure_python')
if not pure_python
    subdir('_extensions')
endif