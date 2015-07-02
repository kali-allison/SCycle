path=/Users/kallison/muIn_16
prefix=d_4_

for filename in $path/*.info;
do
  baseFileName=$(basename $filename);
  #~echo $baseFileName;
  #~echo "$prefix_$baseFileName";
  echo "${prefix}";
  #~mv $filename "$path/d_4_baseFileName";
done;
#~
#~mv foo/bar/blee/{blaz,foobar}.txt
