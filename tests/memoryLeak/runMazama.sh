#!/bin/bash
#PBS -N m4
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/yyang/memoryLeak/outFiles/m4.err
#PBS -o /data/dunham/yyang/memoryLeak/outFiles/m4.out

EXEC_DIR=/data/dunham/yyang/memoryLeak
cd $PBS_O_WORKDIR

valgrind --leak-check=full mpirun $EXEC_DIR/main
