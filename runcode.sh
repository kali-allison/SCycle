#!/bin/bash
#PBS -N eq_cycle_order4
#PBS -l nodes=1:ppn=16
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e eq_cycle_order4.err
#PBS -o eq_cycle_order4.out
#

EXEC_DIR=/data/dunham/kallison/eqCycle
cd $PBS_O_WORKDIR

#~mpirun $EXEC_DIR/main -order 4 -Ny 301 -Nz 301
#~mpirun $EXEC_DIR/main -order 4 -Ny 601 -Nz 601
mpirun $EXEC_DIR/main
#mpirun $EXEC_DIR/main -ksp_type gmres -ORDER 4 -Ny 151 -Nz 151
