#!/bin/bash
#PBS -N nhe_g25_l8
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/eqcycle/outFiles/test.err
#PBS -o /data/dunham/kallison/eqcycle/outFiles/test.out


EXEC_DIR=/data/dunham/kallison/eqcycle
INIT_DIR=/data/dunham/kallison/eqcycle/in
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main $INIT_DIR/spinUpTest.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/bruteForceViscShear.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/flashHeating.in

