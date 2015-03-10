#!/bin/bash
#PBS -N mu36_Dc8_b2_sConst
#PBS -l nodes=1:ppn=16
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e data/mu36_Dc8_b2_sConst.err
#PBS -o data/mu36_Dc8_b2_sConst.out
#

EXEC_DIR=/data/dunham/kallison/eqCycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main $INIT_DIR/basin.in
