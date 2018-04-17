#!/bin/bash
#PBS -N lab50_l0.37_w0.2
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/eqcycle/outEqs/lab50_l0.37_w0.2.err
#PBS -o /data/dunham/kallison/eqcycle/outEqs/lab50_l0.37_w0.2.out

EXEC_DIR=/data/dunham/kallison/eqcycle
INIT_DIR=/data/dunham/kallison/eqcycle/in
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/main $INIT_DIR/he_ssits.in
mpirun $EXEC_DIR/main $INIT_DIR/he_eqs.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/test.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/heCont.in

