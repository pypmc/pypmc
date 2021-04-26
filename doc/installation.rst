Installation
------------

pypmc is developed and tested for python 3.6 and higher.

pypmc depends only on a few standard modules:

* cython [only at build time]
* numpy
* scipy
* setuptools (>=3.3)

If you have pip, the easiest way to install is from the python package
index::

   pip install pypmc

and all dependencies are automatically downloaded if necessary.

If you don't have pip, just run::

   python setup.py install

The optional modules

* mpi4py (parallelization)
* matplotlib (plotting)
* nose (testing)

can be obtained for example by::

  pip install pypmc[plotting,parallelization]

Note there is no blank inside ``[...]``.

If you have nose installed, you can run pypmc's self tests::

  nosetests pypmc

after installation. Note that this is expected to fail if you run it from the
pypmc source directory due to import mismatches. Just change to any other directory
and it should work if the installation was successful.

Developer notes
````````````````````

If you want to build the latest version from `source
<https://github.com/pypmc/pypmc/>`_, you need install the dependencies
yourself.

To facilitate the handling, the ``Makefile`` in the top-level directory has useful targets for building, testing, doc generation and more::

  make build
  make check
  make doc
  make sdist

Debian or derivative
''''''''''''''''''''''''

On a debian-based system such as ubuntu >= 20.04, you can install all required
and optional dependencies from the package manager like this::

  sudo apt install cython3 python3-numpy python3-scipy
  sudo apt install python3-matplotlib python3-mpi4py python3-nose

To build the documentation::

  sudo apt install python3-sphinx-rtd-theme

Conda
'''''''

In a ``conda`` environment, you achieve the same with::

  conda install cython numpy scipy
  conda install matplotlib mpi4py nose

To build the documentation::

  conda install sphinx sphinx_rtd_theme
