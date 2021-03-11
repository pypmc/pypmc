# This Makefile implements common tasks needed by developers
# A list of implemented rules can be obtained by the command "make help"

NOSETESTS2 ?= nosetests-2.7
NOSETESTS3 ?= nosetests3
PYPMC_MPI_NPROC ?= 2
TEST_INSTALL_DIR ?= /tmp/pypmc-test-install
PYTHON ?= python3

.DEFAULT_GOAL=build
.PHONY .SILENT : help
help :
	echo
	echo "    Implemented targets"
	echo "    ==================="
	echo
	echo "    build        build pypmc for python2 and python3"
	echo "    buildX       build pypmc for pythonX only where X is one of {2,3}"
	echo "    build-sdist  build pypmc from the dist directory (python 2 and 3)"
	echo "    build-sdistX build pypmc from the dist directory (pythonX, X in {2,3})"
	echo "    check        use nosetests to test pypmc with python 2.7 and 3"
	echo "    checkX       use nosetests to test pypmc with python 2.7 or 3,"
	echo "                 where X is one of {2,3}"
	echo "    check-fast   use nosetests to run only quick tests of pypmc"
	echo "                 using $(NOSETESTS2) and $(NOSETESTS3)"
	echo "    check-sdist  use $(NOSETESTS2) and $(NOSETESTS3) to test the distribution"
	echo "                 generated by 'make sdist'"
	echo "    check-sdistX use $(NOSETESTS2) or $(NOSETESTS3) to test the distribution"
	echo "                 generated by 'make sdist', where X is one of {2,3}"
	echo "    check-nompi  use nosetests to test pypmc with python 2.7 and 3"
	echo "                 but do not test mpi parallelization"
	echo "    clean        delete compiled and temporary files"
	echo "    coverage     produce and show a code coverage report"
	echo "                 Note: Cython modules cannot be analyzed"
	echo "    distcheck    runs 'check', check-sdist', 'run-examples' and"
	echo "                 opens a browser with the built documentation"
	echo "    doc          build the html documentation using sphinx"
	echo "    doc-pdf      build the pdf documentation using sphinx"
	echo "    help         show this message"
	echo "    install      install pypmc"
	echo "    installX     install pypmc for pythonX only where X is one of {2,3}"
	echo "    run-examples run all examples using python 2 and 3"
	echo "    sdist        make a source distribution"
	echo "    show-todos   show todo marks in the source code"
	echo
	echo "    Influential variables"
	echo "    ====================="
	echo
	echo "    NOSETESTS2   the executable to run nose tests for python 2"
	echo "    NOSETESTS3   the executable to run nose tests for python 3"

.PHONY : clean
clean:
	#remove build doc
	rm -rf ./doc/_build

	#remove .pyc files created by python 2.7
	rm -f ./*.pyc
	find -P . -name '*.pyc' -delete

	#remove .pyc files crated by python 3
	rm -rf ./__pycache__
	find -P . -name __pycache__ -delete

	#remove build folder in root directory
	rm -rf ./build

	#remove cythonized C source and object files
	find -P . -name '*.c' -delete

	#remove variational binaries only if command line argument specified
	find -P . -name '*.so' -delete

	#remove backup files
	find -P . -name '*~' -delete

	#remove files created by coverage
	rm -f .coverage
	rm -rf coverage

	# remove egg info
	rm -rf pypmc.egg-info

	# remove downloaded seutptools
	rm -f setuptools-3.3.zip

	# remove dist/
	rm -rf dist

	rm -rf $(TEST_INSTALL_DIR)

.PHONY : build
build : build2 build3

.PHONY : build2
build2 :
	python2 setup.py build_ext --inplace

.PHONY : build3
build3 :
	python3 setup.py build_ext --inplace

.PHONY : check
check : check2 check3 check2mpi check3mpi

.PHONY : check-nompi
check-nompi : check2 check3

.PHONY : check2
check2 : build2
	@ # run tests
	$(NOSETESTS2) --processes=-1 --process-timeout=60

.PHONY : check2mpi
check2mpi : build2
	@# run tests in parallel
	mpirun $(PYPMC_MPI_ARGS) -n $(PYPMC_MPI_NPROC) $(NOSETESTS2)

.PHONY : check3
check3 : build3
	@ # run tests
	$(NOSETESTS3) --processes=-1 --process-timeout=60

.PHONY : check3mpi
check3mpi : build3
	@# run tests in parallel
	mpirun $(PYPMC_MPI_ARGS) -n $(PYPMC_MPI_NPROC) $(NOSETESTS3)

.PHONY : check-fast
check-fast : build
	$(NOSETESTS2) -a '!slow' --processes=-1 --process-timeout=60
	$(NOSETESTS3)    -a '!slow' --processes=-1 --process-timeout=60

.PHONY : .build-system-default
.build-system-default :
	$(PYTHON) setup.py build_ext --inplace

.PHONY : doc
doc : .build-system-default
	cd doc && make html

.PHONY : doc-pdf
doc-pdf : .build-system-default
	cd doc; make latexpdf

.PHONY : run-examples
run-examples : build
	cd examples ; \
	for file in $$(ls) ; do \
			echo running $${file} with python2 && \
			python2 $${file} || exit 1 && \
			echo running $${file} with python3 && \
			python3 $${file} || exit 1 && \
			\
			# execute with mpirun if mpi4py appears in the file \
			if grep -Fq 'mpi4py' $${file} ; then \
		echo "$${file}" is mpi parallelized && \
		echo running $${file} in parallel with python2 && \
		mpirun $(PYPMC_MPI_ARGS) -n 2 python2 $${file} || exit 1 && \
		echo running $${file} in parallel with python3 && \
		mpirun $(PYPMC_MPI_ARGS) -n 2 python3 $${file} || exit 1  ; \
			fi \
	; \
	done

.PHONY : sdist
sdist :
	python3 setup.py sdist

.PHONY : build-sdist
build-sdist : build-sdist2 build-sdist3

./dist/pypmc*/NUL : sdist
	cd dist && tar xaf *.tar.gz && cd *

.PHONY : build-sdist2
build-sdist2 : ./dist/pypmc*/NUL
	cd dist/pypmc* && python2 setup.py build

.PHONY : build-sdist3
build-sdist3 : ./dist/pypmc*/NUL
	cd dist/pypmc* && python3 setup.py build

.PHONY: clean-test-install
clean-test-install:
	rm -rf $(TEST_INSTALL_DIR)

.PHONY: install-sdist3
install-sdist3: sdist clean-test-install
	pip3 install --target $(TEST_INSTALL_DIR) dist/pypmc-*.tar.gz

.PHONY : check-sdist
check-sdist : check-sdist2 check-sdist3

.PHONY : check-sdist2
check-sdist2 : build-sdist2
	cd dist/*/build/lib*2.7 && \
	$(NOSETESTS2) --processes=-1 --process-timeout=60 && \
	mpirun $(PYPMC_MPI_ARGS) -n $(PYPMC_MPI_NPROC) $(NOSETESTS2)

.PHONY : check-sdist3
check-sdist3 : build-sdist3 install-sdist3
	cd dist/*/build/lib*3.* && \
	$(NOSETESTS3) --processes=-1 --process-timeout=60 && \
	mpirun $(PYPMC_MPI_ARGS) -n $(PYPMC_MPI_NPROC) $(NOSETESTS3)

.PHONY : distcheck
distcheck : check check-sdist doc
	@ # execute "run-examples" after all other recipes makes are done
	make run-examples

.PHONY : show-todos
grep_cmd = ack-grep -i --no-html --no-cc --no-make [^"au""sphinx.ext."]todo
begin_red = "\033[0;31m"
end_red   = "\033[0m"
show-todos :
	@ # suppress errors here
	@ # note that no todo found is considered as error
	$(grep_cmd) . ; \
	echo ;	echo ; \
	echo $(begin_red)"********************************************************"$(end_red) ; \
	echo $(begin_red)"* The following file types are NOT searched for TODOs: *"$(end_red) ; \
	echo $(begin_red)"* o c source                                           *"$(end_red) ; \
	echo $(begin_red)"* o html source                                        *"$(end_red) ; \
	echo $(begin_red)"* o makefiles                                          *"$(end_red) ; \
	echo $(begin_red)"********************************************************"$(end_red) ; \
	echo

.PHONY : coverage
coverage : .build-system-default
	rm -rf coverage
	nosetests --with-coverage --cover-package=pypmc --cover-html --cover-html-dir=coverage
	xdg-open coverage/index.html

.PHONY : install
install : install2 install3

.PHONY : install2
install2 : build2
	python2 setup.py install --user

.PHONY : install3
install3 : build3
	python3 setup.py install --user
