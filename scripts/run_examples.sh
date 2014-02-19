cd ../examples
for file in $(ls)
do
    python  $file &
    python3 $file &
done
wait
