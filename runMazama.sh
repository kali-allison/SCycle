#!/bin/bash
#PBS -N n1_Nz3329_AMG
#PBS -l nodes=1:ppn=24
#PBS -q default
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/eqcycle/outTime/n1_Nz3329_AMG.err
#PBS -o /data/dunham/kallison/eqcycle/outTime/n1_Nz3329_AMG.out

EXEC_DIR=/data/dunham/kallison/eqcycle
INIT_DIR=/data/dunham/kallison/eqcycle/in
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/main $INIT_DIR/he_ssits.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/he_eqs.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/linEl.in
mpirun $EXEC_DIR/main $INIT_DIR/n1_Nz3329_AMG.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/heCont.in

