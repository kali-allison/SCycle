#!/bin/bash
#PBS -N const1e13_mc
#PBS -l nodes=1:ppn=16
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/newEqCycle/data/const1e13_mc.err
#PBS -o /data/dunham/kallison/newEqCycle/data/const1e13_mc.out
#

EXEC_DIR=/data/dunham/kallison/newEqCycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/maxwell2D.in
