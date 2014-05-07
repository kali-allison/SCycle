#!/bin/bash

rm -rf eq_cycle_order*

printf "\nSubmitting job using: \n\n"
cat runcode.sh
printf "\n\n"

qsub runcode.sh
qstat | grep kallison | grep -v C
RETVAL=$?
while [ $RETVAL -eq 0 ]
do
  qstat | grep kallison | grep -v C
  RETVAL=$?
  sleep 30
done
