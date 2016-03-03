#!/bin/bash
#PBS -N p2d_uc
#PBS -l nodes=1:ppn=16
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/newEqCycle/data/p2d_uc.err
#PBS -o /data/dunham/kallison/newEqCycle/data/p2d_uc.out
#

EXEC_DIR=/data/dunham/kallison/newEqCycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main $INIT_DIR/powerLaw2D.in
