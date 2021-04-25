from setuptools import setup, find_packages, Extension
import os
import sys

package_name = 'pypmc'

# set the version number
with open('pypmc/_version.py') as f:
    exec(f.read())


def get_extensions():
    import numpy

    extra_compile_args=["-Wno-unused-but-set-variable",
                        "-Wno-unused-function",
                        "-O3"]
    include_dirs = [numpy.get_include()]

    from Cython.Build import cythonize

    extensions = [Extension('*', ['pypmc/*/*.pyx'],
                            extra_compile_args=extra_compile_args,
                            include_dirs=include_dirs)]

    compiler_directives = dict(boundscheck=False, cdivision=True,
                               embedsignature=True,
                               profile=False, wraparound=False,
                               # needed to make cython>0.29 happy
                               language_level=2)
    ext_modules = cythonize(extensions, compiler_directives=compiler_directives)

    return ext_modules


def setup_package():
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
        author_email='beaujean@mytum.de, stephan.jahn@mytum.de',
        url='https://github.com/pypmc/pypmc',
        description='A toolkit for adaptive importance sampling featuring implementations of variational Bayes, population Monte Carlo, and Markov chains.',
        long_description=long_description,
        license='GPLv2',
        install_requires=['numpy>=1.6, <2.0', 'scipy'],
        extras_require={'testing': ['nose'], 'plotting': ['matplotlib'], 'parallelization': ['mpi4py']},
        classifiers=['Development Status :: 5 - Production/Stable',
                     'Intended Audience :: Developers',
                     'Intended Audience :: Science/Research',
                     'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',
                     'Operating System :: Unix',
                     'Programming Language :: Cython',
                     'Programming Language :: Python',
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
        # For these actions, dependencies are not required.
        pass
    else:
        setup_args['packages'] = find_packages()
        setup_args['ext_modules'] = get_extensions()

    setup(**setup_args)


if __name__ == '__main__':
    setup_package()
