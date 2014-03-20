try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from Cython.Build import cythonize

# set the version number
with open('pypmc/_version.py') as f:
    exec(f.read())

n = 'pypmc'

setup(
    name=n,
    packages=[n],
    version=__version__,
    author='Frederik Beaujean, Stephan Jahn',
    author_email='Frederik.Beaujean@lmu.de, stephan.jahn@mytum.de',
    license='GPLv2',
    install_requires=['numpy', 'scipy', 'cython'],
    ext_modules=cythonize('pypmc/mix_adapt/*.pyx')
    )
