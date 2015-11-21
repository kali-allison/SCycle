#!/bin/bash
#PBS -N mvfc4
#PBS -l nodes=1:ppn=16
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/newEqCycle/data/mvfc4.err
#PBS -o /data/dunham/kallison/newEqCycle/data/mvfc4.out
#

EXEC_DIR=/data/dunham/kallison/newEqCycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main_linearElastic $INIT_DIR/test.in
