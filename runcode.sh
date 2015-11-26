#!/bin/bash
#PBS -N mvfc_v1e20_rate_bc
#PBS -l nodes=1:ppn=16
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/newEqCycle/data/mvfc_v1e20_rate_bc.err
#PBS -o /data/dunham/kallison/newEqCycle/data/mvfc_v1e20__rate_bc.out
#

EXEC_DIR=/data/dunham/kallison/newEqCycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main $INIT_DIR/test.in
