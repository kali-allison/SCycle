#!/bin/bash
#PBS -N order4
#PBS -l nodes=2:ppn=16
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e hetCycle.err
#PBS -o hetCycle.out
#

EXEC_DIR=/data/dunham/kallison/eqCycle
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main -order 2 -Ny 401 -Nz 401
