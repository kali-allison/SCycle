#!/bin/bash
#PBS -N im
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/scycle/outFiles/im.err
#PBS -o /data/dunham/kallison/scycle/outFiles/im.out

EXEC_DIR=/data/dunham/kallison/scycle/source
#~ INIT_DIR=/data/dunham/kallison/scycle/in
INIT_DIR=/data/dunham/kallison/grainSizeEv/in
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/main $INIT_DIR/ex2.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/test_lm.in
mpirun $EXEC_DIR/main $INIT_DIR/test_olivine.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/ss_wdisl.in

