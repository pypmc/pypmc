from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages, Extension
from codecs import open # To use a consistent encoding
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
    packages=find_packages(),
    version=__version__,
    author='Frederik Beaujean, Stephan Jahn',
    author_email='Frederik.Beaujean@lmu.de, stephan.jahn@mytum.de',
    license='GPLv2',
    install_requires=['numpy', 'scipy', 'cython', 'setuptools>=3.3'],
    extras_require={'testing': ['nose'], 'plotting': ['matplotlib'], 'parallelization': ['mpi4py']},
    ext_modules=cythonize(extensions,
                          compiler_directives=dict(boundscheck=False, cdivision=True,
                                                   embedsignature=True,
                                                   profile=False, wraparound=False),
                          )
    )
