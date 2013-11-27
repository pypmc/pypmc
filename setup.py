try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

# set the version number
with open('pypmc/version.py') as f:
    exec(f.read())

n = 'pypmc'

setup(
    name=n,
    packages=[n],
    version=__version__,
    author='Frederik Beaujean',
    author_email='Frederik.Beaujean@lmu.de',
    license='GPLv2',
    install_requires=['numpy', 'scipy'],
    )
