#!/bin/bash
#PBS -N g30_b7_1e19_qd
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/eqcycle/outeqs/g30_b7_1e19_qd.err
#PBS -o /data/dunham/kallison/eqcycle/outeqs/g30_b7_1e19_qd.out

EXEC_DIR=/data/dunham/kallison/eqcycle
INIT_DIR=/data/dunham/kallison/eqcycle/in
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main $INIT_DIR/test.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/exploreLambda.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/aguEqs.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/spinUpTest.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/bruteForceViscShear.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/flashHeating.in

