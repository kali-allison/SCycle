#!/bin/bash
#PBS -N v7
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/eqcycle/outFiles/v7.err
#PBS -o /data/dunham/kallison/eqcycle/outFiles/v7.out

EXEC_DIR=/data/dunham/kallison/eqcycle
INIT_DIR=/data/dunham/kallison/eqcycle/in
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main $INIT_DIR/test.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/test_fh.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/ex3.in

