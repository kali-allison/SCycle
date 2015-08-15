#!/bin/bash
#PBS -N lowV1e9D15_Lz24
#PBS -l nodes=1:ppn=16
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e ../maxwellVisc/layered/lowV1e9D15_Lz24.err
#PBS -o ../maxwellVisc/layered/lowV1e9D15_Lz24.out
#

EXEC_DIR=/data/dunham/kallison/eqCycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main $INIT_DIR/maxwell2D.in
