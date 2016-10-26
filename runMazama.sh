#!/bin/bash
#PBS -N blanpied_Nz701
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/eqcycle/spinUpTests/testSpinUp2_blanpied_Dc8e-3_Nz701.err
#PBS -o /data/dunham/kallison/eqcycle/spinUpTests/testSpinUp2_blanpied_Dc8e-3_Nz701.out


EXEC_DIR=/data/dunham/kallison/eqcycle
INIT_DIR=/data/dunham/kallison/eqcycle
#~ EXEC_DIR=/scratch/kallison
#~ INIT_DIR=/scratch/kallison
cd $PBS_O_WORKDIR

mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/m2D.in
#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/h2D.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/m2D.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/m2D_spinUp.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/mms.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/h2D.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/m2D.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/pl.in

#~ mv /scratch/kallison/l2D_wTherm_test1* /data/dunham/kallison/eqcycle/data/wTherm/
