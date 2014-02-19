#remove build doc
rm -rf ../doc/_build

#remove .pyc files crated by python 2.7
find -P .. -name *.pyc -delete

#remove .pyc files crated by python 3
find -P .. -name __pycache__ -delete

#remove backup files
find -P -name *~ -delete
find -P .. -name *~ -delete

#remove file created by coverage
rm -f .coverage
