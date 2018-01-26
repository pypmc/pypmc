#! /bin/bash

# If any line in this section fails, build fails.
set -e

# prepare cibuild environment
export CIBW_BEFORE_BUILD="{pip} install Cython numpy scipy"
export CIBW_TEST_REQUIRES="nose"
export CIBW_TEST_COMMAND="nosetests {project}"
# numpy not supported anymore on python3.3
export CIBW_SKIP="cp33-*"

# build wheels and deploy
$PIP install cibuildwheel==0.7.0 twine;
cibuildwheel --output-dir wheelhouse;
# TODO reactivate check once it works
# if [[ $TRAVIS_TAG ]]; then
# python -m twine upload wheelhouse/*.whl;
# fi
