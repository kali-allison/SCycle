#!/bin/bash
#PBS -N l2D_Dc8e-3_Ly120
#PBS -l nodes=1:ppn=20
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/eqcycle/data/linEl/l2D_Dc8e-3_Ly120.err
#PBS -o /data/dunham/kallison/eqcycle/data/linEl/l2D_Dc8e-3_Ly120.out
#

EXEC_DIR=/data/dunham/kallison/eqcycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/m2D.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/m2D.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/m2D.in
