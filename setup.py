try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

# set the version number
with open('pypmc/_version.py') as f:
    exec(f.read())

n = 'pypmc'

extra_compile_args=["-Wno-unused-but-set-variable",
                    "-Wno-unused-function",
                    "-O3"]

extensions = [ Extension('*', ['pypmc/*/*.pyx' ],
                                                  extra_compile_args=extra_compile_args,
                                                  include_dirs=[numpy.get_include()])
             ]

setup(
    name=n,
    packages=[n],
    version=__version__,
    author='Frederik Beaujean, Stephan Jahn',
    author_email='Frederik.Beaujean@lmu.de, stephan.jahn@mytum.de',
    license='GPLv2',
    install_requires=['numpy', 'scipy', 'cython'],
    extras_require={'testing': ['nose'], 'plotting': ['matplotlib'], 'parallelization': ['mpi4py']},
    ext_modules=cythonize(extensions,
                          compiler_directives=dict(profile=False, boundscheck=False,
                                                   wraparound=False, cdivision=True),
                          )
    )
