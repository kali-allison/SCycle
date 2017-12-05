#!/bin/bash
#PBS -N ssg30_Tss_cs_v09_lambda7_Ny251_Nz301
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/eqcycle/outFiles/ssg30_Tss_cs_v09_lambda7_Ny251_Nz301.err
#PBS -o /data/dunham/kallison/eqcycle/outFiles/ssg30_Tss_cs_v09_lambda7_Ny251_Nz301.out


EXEC_DIR=/data/dunham/kallison/eqcycle
INIT_DIR=/data/dunham/kallison/eqcycle/in
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main $INIT_DIR/spinUpTest.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/bruteForceViscShear.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/flashHeating.in

