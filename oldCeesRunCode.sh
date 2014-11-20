#!/bin/bash
#PBS -N memCheck
#PBS -l nodes=4:ppn=8
#PBS -q Q26b
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e data/err.txt
#PBS -o data/out.txt
#

EXEC_DIR=/data/dunham/kallison/eqCycle
INIT_DIR=/data/dunham/kallison/eqCycle
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main $INIT_DIR/init_mu_9.in
