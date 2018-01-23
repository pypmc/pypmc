from __future__ import print_function

# bootstrap: download setuptools 3.3 if needed
from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages, Extension
import os
import sys

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


def get_extensions():
    import numpy

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

    except Exception as error:
        print('WARNING: Could not cythonize:', repr(error))
        ext_modules = []

    # either cython not available or we are in a source distribution
    # TODO isn't there a less cumbersome way to emulate cythonize?
    if not ext_modules:
        ext_modules = [Extension(os.path.splitext(f)[0].replace('/', '.'), # tmp/file.c -> tmp.file
                                 [f],
                                 extra_compile_args=extra_compile_args,
                                 include_dirs=include_dirs) for f in find_files(package_name, '*.c')]

    return ext_modules


def setup_package():
    # Figure out whether to add ``*_requires = ['numpy']``.
    setup_requires = []
    try:
        import numpy
    except ImportError:
        setup_requires.append('numpy>=1.6, <2.0')

    # the long description is unavailable in a source distribution and not essential to build
    try:
        with open('doc/abstract.txt') as f:
            long_description = f.read()
    except:
        long_description = ''

    setup_args = dict(
        name=package_name,
        packages=find_packages(),
        version=__version__,
        author='Frederik Beaujean, Stephan Jahn',
        author_email='Frederik.Beaujean@lmu.de, stephan.jahn@mytum.de',
        url='https://github.com/fredRos/pypmc',
        description='A toolkit for adaptive importance sampling featuring implementations of variational Bayes, population Monte Carlo, and Markov chains.',
        long_description=long_description,
        license='GPLv2',
        setup_requires=setup_requires,
        install_requires=['numpy', 'scipy'],
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

    if len(sys.argv) >= 2 and (
            '--help' in sys.argv[1:] or
            sys.argv[1] in ('--help-commands', 'egg_info', '--version',
                            'clean')):
        # For these actions, NumPy is not required.
        pass
    else:
        setup_args['packages'] = find_packages()
        setup_args['ext_modules'] = get_extensions()

    setup(**setup_args)


if __name__ == '__main__':
    setup_package()
