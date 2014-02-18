rm -rf /tmp/pypmc_coverage /tmp/pypmc_coverage3
nosetests  pypmc --with-coverage --cover-package=pypmc --cover-html --cover-html-dir=/tmp/pypmc_coverage &
nosetests3 pypmc --with-coverage --cover-package=pypmc --cover-html --cover-html-dir=/tmp/pypmc_coverage3
firefox /tmp/pypmc_coverage/index.html /tmp/pypmc_coverage3/index.html
