nosetests  pypmc -a '!slow' --processes=-1 --process-timeout=60 &
nosetests3 pypmc -a '!slow' --processes=-1 --process-timeout=60
wait
