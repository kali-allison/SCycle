#!/bin/bash
#PBS -N l2D_IMEX_wTrans
#PBS -l nodes=1:ppn=16
#PBS -q default
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/eqcycle/data/l2D.err
#PBS -o /data/dunham/kallison/eqcycle/data/l2D.out
#

EXEC_DIR=/data/dunham/kallison/eqcycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/m2D.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/m2D.in
mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/m2D.in
