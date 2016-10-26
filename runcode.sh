#!/bin/bash
#PBS -N linEl_Nz601
#PBS -l nodes=1:ppn=16
#PBS -q default
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/newEqCycle/spinUpTests/testSpinUp2_Dc8e-3_Nz601.err
#PBS -o /data/dunham/kallison/newEqCycle/spinUpTests/testSpinUp2_Dc8e-3_Nz601.out
#

EXEC_DIR=/data/dunham/kallison/newEqCycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/linEl2D.in
mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/m2D.in
#~mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/powerLaw2D.in
