#!/bin/bash
#PBS -N SScs
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/scycle/outFiles/SScs.err
#PBS -o /data/dunham/kallison/scycle/outFiles/SScs.out

EXEC_DIR=/data/dunham/kallison/scycle/source
INIT_DIR=/data/dunham/kallison/scycle/tempIn
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main $INIT_DIR/SScs_Lz60_zlab60_l0.8_w10.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/testSS.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/quasidynamic.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/singleDynamicRupture.in

