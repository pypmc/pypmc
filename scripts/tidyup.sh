#remove build doc
rm -rf ../doc/_build

#remove .pyc files crated by python 2.7
find -P .. -name *.pyc -delete

#remove .pyc files crated by python 3
find -P .. -name __pycache__ -delete

#remove build folder in root directory
rm -rf ../build

#remove variational.c
rm -rf ../pypmc/mix_adapt/variational.c

#remove variational binaries only if command line argument specified
if [ "$1" = "bin" ]
then
  rm -rf ../pypmc/mix_adapt/*.so
else
  echo 'type "./tidyup bin" to delete the compiled cython objects'
fi

#remove backup files
find -P -name *~ -delete
find -P .. -name *~ -delete

#remove file created by coverage
rm -f .coverage

#remove plot created by tools.plot_test
find -P -name TestPlotMixture_python2.png -delete
find -P -name TestPlotMixture_python3.png -delete
find -P .. -name TestPlotMixture_python2.png -delete
find -P .. -name TestPlotMixture_python3.png -delete
