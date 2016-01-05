#!/bin/bash
#PBS -N FDP
#PBS -l nodes=4:ppn=16
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e FDP16p4n.err
#PBS -o FDP16p4n.out
#

EXEC_DIR=/data/dunham/peterdo/MatMultTests/
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/FDP



