#!/bin/bash
#PBS -N test
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/scycle/outFiles/test.err
#PBS -o /data/dunham/kallison/scycle/outFiles/test.out

EXEC_DIR=/data/dunham/kallison/scycle/source
INIT_DIR=/data/dunham/kallison/scycle/in
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main $INIT_DIR/test.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/testSS.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/quasidynamic.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/singleDynamicRupture.in

