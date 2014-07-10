#!/bin/bash
#PBS -N eq_cycle_order4_np1
#PBS -l nodes=1:ppn=1
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e eqCycle_order4_nodes1_np1.err
#PBS -o eqCycle_order4_nodes1_np1.out
#

EXEC_DIR=/data/dunham/kallison/eqCycle
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main -order 4 -Ny 1201 -Nz 401
