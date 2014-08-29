#!/bin/bash
#PBS -N eqCycle
#PBS -l nodes=2:ppn=16
#PBS -q default
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e data/eqCycle.err
#PBS -o data/eqCycle.out
#

EXEC_DIR=/data/dunham/kallison/eqCycle
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main
