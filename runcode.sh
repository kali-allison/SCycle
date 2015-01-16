#!/bin/bash
#PBS -N o4_a2hNN_chol_withH_
#PBS -l nodes=1:ppn=1
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e data/o4_a2hNN_chol_withH.err
#PBS -o data/o4_a2hNN_chol_withH.out
#

EXEC_DIR=/data/dunham/kallison/eqCycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main $INIT_DIR/test.in
