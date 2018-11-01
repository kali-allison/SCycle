#!/bin/bash
#PBS -N qd
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/scycle/outFiles/qd.err
#PBS -o /data/dunham/kallison/scycle/outFiles/qd.out

EXEC_DIR=/data/dunham/kallison/scycle/source
INIT_DIR=/data/dunham/kallison/scycle/in
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/main $INIT_DIR/test.in
mpirun $EXEC_DIR/main $INIT_DIR/ice.in

