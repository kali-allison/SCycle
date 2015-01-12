#!/bin/bash
#PBS -N 4_13_muIn36_LU
#PBS -l nodes=1:ppn=16
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e data/4_13_muIn36.err
#PBS -o data/4_13_muIn36.out
#

EXEC_DIR=/data/dunham/kallison/eqCycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main $INIT_DIR/basin.in
