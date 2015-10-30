#!/bin/bash
#PBS -N b0.000_sNMin0.1_sNMax50
#PBS -l nodes=1:ppn=16
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/newEqCycle/shallowFricProps/b0.000_sNMin0.1_sNMax50.err
#PBS -o /data/dunham/kallison/newEqCycle/shallowFricProps/b0.000_sNMin0.1_sNMax50.out
#

EXEC_DIR=/data/dunham/kallison/newEqCycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main_linearElastic $INIT_DIR/realisticFriction.in
