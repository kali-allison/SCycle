#!/bin/bash
#PBS -N eq_cycle_order2
#PBS -l nodes=1:ppn=1
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e eq_cycle_order2.err
#PBS -o eq_cycle_order2.out
#

EXEC_DIR=/data/dunham/kallison/localEqCycle
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main -order 2 -Ny 151 -Nz 151
#mpirun $EXEC_DIR/main -order 2 -Ny 41 -Nz 41
#mpirun $EXEC_DIR/main -ksp_type gmres -ORDER 4 -Ny 151 -Nz 151
