#!/bin/bash
#PBS -N m2D_nehrp
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/eqcycle/data/nehrp/nehrp.err
#PBS -o /data/dunham/kallison/eqcycle/data/nehrp/nehrp.out
#

EXEC_DIR=/data/dunham/kallison/eqcycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/nehrp.in
#~mpirun $EXEC_DIR/main $INIT_DIR/powerLaw2D.in
