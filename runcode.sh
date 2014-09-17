#!/bin/bash
#PBS -N
#PBS -l nodes=4:ppn=16
#PBS -q default
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e err.txt
#PBS -o out.txt
#

EXEC_DIR=/data/dunham/kallison/
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main
