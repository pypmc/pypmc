# bootstrap: download setuptools 3.3 if needed
from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages, Extension
import numpy
import os

package_name = 'pypmc'

# set the version number
with open('pypmc/_version.py') as f:
    exec(f.read())

def find_files(directory, pattern):
    '''Generate file names in a directory and its subdirectories matching ``pattern``.'''
    import fnmatch

    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

extra_compile_args=["-Wno-unused-but-set-variable",
                    "-Wno-unused-function",
                    "-O3"]
include_dirs = [numpy.get_include()]

# Call cython if *.pyx available
try:
    from Cython.Build import cythonize

    extensions = [Extension('*', ['pypmc/*/*.pyx'],
                            extra_compile_args=extra_compile_args,
                            include_dirs=include_dirs)]

    compiler_directives = dict(boundscheck=False, cdivision=True,
                               embedsignature=True,
                               profile=False, wraparound=False)
    ext_modules = cythonize(extensions, compiler_directives=compiler_directives)

except ImportError:
    ext_modules = []

# either cython not available or we are in a source distribution
# todo isn't there a less cumbersome way to emulate cythonize?
if not ext_modules:
    ext_modules = [Extension(os.path.splitext(f)[0].replace('/', '.'), # tmp/file.c -> tmp.file
                             [f],
                             extra_compile_args=extra_compile_args,
                             include_dirs=include_dirs) for f in find_files(package_name, '*.c')]

# the long description is unavailable in a source distribution and not essential to build
try:
    with open('doc/abstract.txt') as f:
        long_description = f.read()
except:
    long_description = ''

setup(
    name=package_name,
    packages=find_packages(),
    ext_modules=ext_modules,
    version=__version__,
    author='Frederik Beaujean, Stephan Jahn',
    author_email='Frederik.Beaujean@lmu.de, stephan.jahn@mytum.de',
    url='https://github.com/fredRos/pypmc',
    description='A toolkit for adaptive importance sampling featuring implementations of variational Bayes and population Monte Carlo.',
    long_description=long_description,
    license='GPLv2',
    install_requires=['numpy', 'scipy', 'setuptools>=3.3'],
    extras_require={'testing': ['nose'], 'plotting': ['matplotlib'], 'parallelization': ['mpi4py']},
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',
                 'Operating System :: Unix',
                 'Programming Language :: Cython',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Mathematics',
                 ],
    platforms=['Unix'],
    )
