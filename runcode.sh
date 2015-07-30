#!/bin/bash
#PBS -N linEl_2D
#PBS -l nodes=1:ppn=16
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e testMaxwell/linEl_2D.err
#PBS -o testMaxwell/linEl_2D.out
#

EXEC_DIR=/data/dunham/kallison/eqCycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main $INIT_DIR/maxwell2D.in
