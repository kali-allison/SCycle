#!/bin/bash
#PBS -N l2D_nT
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/eqcycle/data/nehrp/l2D_nT.err
#PBS -o /data/dunham/kallison/eqcycle/data/nehrp/l2D_nTout
#

EXEC_DIR=/data/dunham/kallison/eqcycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/linEl2D.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/linEl2D.in
mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/m2D.in
