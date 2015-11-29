#!/bin/bash
#PBS -N mvfc_l_Dc8e-3_N201
#PBS -l nodes=1:ppn=16
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/newEqCycle/data/mvfc_l_Dc8e-3_N201.err
#PBS -o /data/dunham/kallison/newEqCycle/data/mvfc_l_Dc8e-3_N201.out
#

EXEC_DIR=/data/dunham/kallison/newEqCycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main $INIT_DIR/test.in
