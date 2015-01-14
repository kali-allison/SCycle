path=/data/dunham/kallison/eqCycle/data
prefix=withH_



for filename in *;
do
  #~echo $filename;
  #~echo "$prefix${filename#2_20_muIn36_}"
  mv "$filename" "$prefix${filename#2_20_muIn36_}"
done;
#~
#~mv foo/bar/blee/{blaz,foobar}.txt
