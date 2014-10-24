#!/bin/bash
#PBS -N eqCycleLU
#PBS -l nodes=4:ppn=16
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e data/err.txt
#PBS -o data/out.txt
#

EXEC_DIR=/data/dunham/kallison/eqCycle
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main
