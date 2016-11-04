#!/bin/bash
#PBS -N cremeBrulee_blansVW_spinUp3_23_Ny201_Nz451_Ly120_Lz50
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/eqcycle/stdoutFiles/cremeBrulee_blansVW_spinUp3_23_Ny201_Nz451_Ly120_Lz50.err
#PBS -o /data/dunham/kallison/eqcycle/stdoutFiles/cremeBrulee_blansVW_spinUp3_23_Ny201_Nz451_Ly120_Lz50.out


EXEC_DIR=/data/dunham/kallison/eqcycle
INIT_DIR=/data/dunham/kallison/eqcycle
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/m2D.in
#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/h2D.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_cremeBrulee_spinUp1.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_jellySandwich_spinUp1.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_jellySandwich_cycle.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/max_cremeBrulee_cycle.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/max_jellySandwich_cycle.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/max_cremeBrulee_spinUp2.in
mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_cremeBrulee_spinUp2.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_spinUp.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/max.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/max_spinUp.in

#~ mv /scratch/kallison/l2D_wTherm_test1* /data/dunham/kallison/eqcycle/data/wTherm/
