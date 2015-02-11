#!/bin/bash
#PBS -N muIn36_Ly24
#PBS -l nodes=1:ppn=1
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e domainSizeTests/muIn36_Ly24.err
#PBS -o domainSizeTests/muIn36_Ly24.out
#

EXEC_DIR=/data/dunham/kallison/eqCycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main $INIT_DIR/test.in
