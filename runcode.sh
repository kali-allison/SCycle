#!/bin/bash
#PBS -N m2D_v5e12
#PBS -l nodes=1:ppn=16
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/newEqCycle/data/m2D_v5e12.err
#PBS -o /data/dunham/kallison/newEqCycle/data/m2D_v5e12.out
#

EXEC_DIR=/data/dunham/kallison/newEqCycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

#~mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/powerLaw2D.in
mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/powerLaw2D.in
