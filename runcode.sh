#!/bin/bash
<<<<<<< HEAD
#PBS -N el2D_o4_v1e9
=======
#PBS -N el4_SAT-5i_fc
>>>>>>> 983554d6b9e7b48e8a95b9b1ca0fbe1e1d380896
#PBS -l nodes=1:ppn=16
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
<<<<<<< HEAD
#PBS -e /data/dunham/kallison/newEqCycle/data/el2D_o4_v1e9.err
#PBS -o /data/dunham/kallison/newEqCycle/data/el2D_o4_v1e9.out
=======
#PBS -e /data/dunham/kallison/newEqCycle/data/el4_SAT-5i_fc.err
#PBS -o /data/dunham/kallison/newEqCycle/data/el4_SAT-5i_fc.out
>>>>>>> 983554d6b9e7b48e8a95b9b1ca0fbe1e1d380896
#

EXEC_DIR=/data/dunham/kallison/newEqCycle
INIT_DIR=$EXEC_DIR
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/main_linearElastic $INIT_DIR/test.in
