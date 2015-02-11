#!/bin/bash
#PBS -N test
#PBS -l nodes=1:ppn=1
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e data/test.err
#PBS -o data/test.out
#

EXEC_DIR=/data/dunham/kallison/eqCycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main $INIT_DIR/test.in
