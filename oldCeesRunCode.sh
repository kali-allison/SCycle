#!/bin/bash
#PBS -N scaling
#PBS -l nodes=1
#PBS -q Q26b
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e data/err.txt
#PBS -o data/out.txt
#

EXEC_DIR=/data/dunham/kallison/eqCycle
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main
