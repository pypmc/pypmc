#! /bin/bash

# Print every line, if it fails, the entire build fails.
set -e

# setup environment with conda and matching python version
wget https://repo.continuum.io/miniconda/Miniconda-latest-${CONDA_OS}-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda config --add channels conda-forge
conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION cython matplotlib mpi4py nomkl nose numpy scipy sphinx
# set the backend to work in headless env
mkdir -p ~/.config/matplotlib && cp doc/matplotlibrc ~/.config/matplotlib
source activate test-environment

# install and run unit tests
if [[ $TRAVIS_PYTHON_VERSION == 2.7 ]]; then
    export NOSETESTS2=nosetests;
    make install2;
    make check2;
    make check2mpi;
fi
if [[ $TRAVIS_PYTHON_VERSION == 3.6 ]]; then
    export NOSETESTS3=nosetests;
    make install3;
    make check3;
    make check3mpi;
    # # build and deploy docs only for one python version and only for a new release
    # if [[ $TRAVIS_TAG ]]; then
    #      pip install doctr;
    #      make doc;
    #      doctr deploy . --built-docs doc/_build/;
    #   fi
fi
