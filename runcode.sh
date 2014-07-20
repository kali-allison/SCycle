#!/bin/bash
#PBS -N homCycle
#PBS -l nodes=1:ppn=1
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e homnp1_NyNz201.err
#PBS -o homnp1_NyNz201.out
#

EXEC_DIR=/data/dunham/kallison/eqCycle
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main -order 2 -Ny 21 -Nz 21

#mpirun $EXEC_DIR/main -order 2 -Ny 76 -Nz 76
#mpirun $EXEC_DIR/main -order 2 -Ny 151 -Nz 151
#mpirun $EXEC_DIR/main -order 2 -Ny 301 -Nz 301
#mpirun $EXEC_DIR/main -order 2 -Ny 601 -Nz 601
#mpirun $EXEC_DIR/main -order 2 -Ny 1201 -Nz 1201

#mpirun $EXEC_DIR/main -order 4 -Ny 76 -Nz 76
#mpirun $EXEC_DIR/main -order 4 -Ny 151 -Nz 151
#mpirun $EXEC_DIR/main -order 4 -Ny 301 -Nz 301
#mpirun $EXEC_DIR/main -order 4 -Ny 601 -Nz 601
#mpirun $EXEC_DIR/main -order 4 -Ny 1201 -Nz 1201
