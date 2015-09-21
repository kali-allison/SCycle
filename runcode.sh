#!/bin/bash
#PBS -N Ly60_gradient/lcD20_Ly60_bcR0
#PBS -l nodes=2:ppn=16
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/maxwellVisc/gradient/lcD20_Ly60_bcR0_Ly60.err
#PBS -o /data/dunham/kallison/maxwellVisc/gradient/lcD20_Ly60_bcR0_Ly60.out
#

EXEC_DIR=/data/dunham/kallison/eqCycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main $INIT_DIR/maxwell2D.in
