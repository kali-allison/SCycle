#!/bin/bash
#PBS -N symm_constant
#PBS -l nodes=1:ppn=16
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e domainSizeTests/symm_constant.err
#PBS -o domainSizeTests/symm_constant.out
#

EXEC_DIR=/data/dunham/kallison/eqCycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main $INIT_DIR/test.in
