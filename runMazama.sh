#!/bin/bash
#PBS -N wTherm_test1
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/eqcycle/data/wTherm/wTherm_test1.err
#PBS -o /data/dunham/kallison/eqcycle/data/wTherm/wTherm_test1.out


EXEC_DIR=/data/dunham/kallison/eqcycle
INIT_DIR=$EXEC_DIR
#~ EXEC_DIR=/scratch/kallison
#~ INIT_DIR=/data/dunham/kallison/eqcycle
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/m2D.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/m2D.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/m2D.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/pl.in

#mv /scratch/kallison/l2D_rice1993svw_Dc16e-3_gradsNCap50* /data/dunham/kallison/eqcycle/data/linEl/
mv /scratch/kallison/wTherm_test1* /data/dunham/kallison/eqcycle/data/wTherm/
