#!/bin/bash
#PBS -N eq_cycle_order4_np1
#PBS -l nodes=1:ppn=5
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e test.err
#PBS -o test.out
#

EXEC_DIR=/data/dunham/kallison/eqCycle
cd $PBS_O_WORKDIR

#~mpirun $EXEC_DIR/main -order 2 -Ny 5 -Nz 7

mpirun $EXEC_DIR/main -order 2 -Ny 76 -Nz 76
mpirun $EXEC_DIR/main -order 2 -Ny 151 -Nz 151
mpirun $EXEC_DIR/main -order 2 -Ny 301 -Nz 301
mpirun $EXEC_DIR/main -order 2 -Ny 601 -Nz 601
mpirun $EXEC_DIR/main -order 2 -Ny 1201 -Nz 1201
#~
#~mpirun $EXEC_DIR/main -order 4 -Ny 76 -Nz 76
#~mpirun $EXEC_DIR/main -order 4 -Ny 151 -Nz 151
#~mpirun $EXEC_DIR/main -order 4 -Ny 301 -Nz 301
#~mpirun $EXEC_DIR/main -order 4 -Ny 601 -Nz 601
#~mpirun $EXEC_DIR/main -order 4 -Ny 1201 -Nz 1201
