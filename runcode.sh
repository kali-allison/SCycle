#!/bin/bash
#PBS -N eqCycleLU
#PBS -l nodes=1:ppn=16
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e dataLU/err.txt
#PBS -o dataLU/out.txt
#

EXEC_DIR=/data/dunham/kallison/eqCycle
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main
