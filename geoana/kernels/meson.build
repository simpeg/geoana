
python_sources = [
  '__init__.py',
  'potential_field_prism.py',
  'tranverse_electric_reflections.py',
]

py.install_sources(
  python_sources,
  subdir: 'geoana/kernels'
)


if with_extensions
    subdir('_extensions')
endif