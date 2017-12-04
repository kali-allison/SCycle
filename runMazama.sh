#!/bin/bash
#PBS -N bf_g30_MUMPS_v09_a6_al
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/eqcycle/outFiles/bf_g30_MUMPS_v09_a6_al.err
#PBS -o /data/dunham/kallison/eqcycle/outFiles/bf_g30_MUMPS_v09_a6_al.out


EXEC_DIR=/data/dunham/kallison/eqcycle
INIT_DIR=/data/dunham/kallison/eqcycle/in
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/main $INIT_DIR/spinUpTest.in
mpirun $EXEC_DIR/main $INIT_DIR/bruteForce.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/flashHeating.in

