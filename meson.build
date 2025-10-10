project(
  'geoana',
  # Note that the git commit hash cannot be added dynamically here
  # (it is dynamically generated though setuptools_scm)
  version: run_command('python',
    [
      '-c',
      '''
from setuptools_scm import get_version
print(get_version())'''
    ],
    check: true
).stdout().strip(),
  license: 'MIT',
  meson_version: '>= 1.4.0',
  default_options: [
    'buildtype=debugoptimized',
    'b_ndebug=if-release',
    'cpp_std=c++17',
  ],
)

with_extensions = get_option('with_extensions')

# https://mesonbuild.com/Python-module.html
py_mod = import('python')
py = py_mod.find_installation(pure: not with_extensions)

subdir('geoana')