Installation
------------

pypmc depends only on a few standard modules:

* numpy
* scipy
* cython
* setuptools (>=3.3)

If you have pip, the easiest way to install is from the python package
index::

   pip install pypmc

If you don't have pip, just run::

   python setup.py install

and all dependencies are automatically downloaded if necessary.

The optional modules

* mpi4py (parallelization)
* matplotlib (plotting)
* nose (testing)

can be obtained for example by::

  pip install pypmc[plotting,parallelization]

Note there is no blank inside ``[...]``.

pypmc is developed and tested for both python 2.7 and 3.x. On a
debian-based system such as ubuntu >= 12.04, you can install all
dependencies from the package manager.
