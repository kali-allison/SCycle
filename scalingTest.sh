#!/bin/bash
#PBS -N nodes1_Ny401_Nz1201_KA
#PBS -l nodes=1:ppn=16
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e eqCycle_nodes1.err
#PBS -o eqCycle_nodes1.out
#
#EXEC_DIR=/data/dunham/kallison/eqCycle
cd $PBS_O_WORKDIR
#
mpirun EXEC_DIR/main -order 4 -Ny 401 -Nz 1201
