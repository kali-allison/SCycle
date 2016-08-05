#!/bin/bash
#PBS -N max/rice1993svw_gradsN_s1e-14
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/eqcycle/data/max/rice1993svw_gradsN_s1e-14.err
#PBS -o /data/dunham/kallison/eqcycle/data/max/rice1993svw_gradsN_s1e-14.out
#

EXEC_DIR=/data/dunham/kallison/eqcycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/m2D.in
mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/m2D.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/m2D.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/pl.in
