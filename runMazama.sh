#!/bin/bash
#PBS -N pljellySandwich_blansVW_dc32e-3_cycle_Ny201_Nz151
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/eqcycle/stdoutFiles/pljellySandwich_blansVW_dc32e-3_cycle_Ny201_Nz151.err
#PBS -o /data/dunham/kallison/eqcycle/stdoutFiles/pljellySandwich_blansVW_dc32e-3_cycle_Ny201_Nz151.out


EXEC_DIR=/data/dunham/kallison/eqcycle
INIT_DIR=/data/dunham/kallison/eqcycle
#~ EXEC_DIR=/scratch/kallison
#~ INIT_DIR=/scratch/kallison
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/m2D.in
#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/h2D.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_cremeBrulee_spinUp1.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_jellySandwich_spinUp1.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_jellySandwich_spinUp2.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_jellySandwich_cycle.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/max_cremeBrulee_cycle.in
mpirun $EXEC_DIR/main $INIT_DIR/max_jellySandwich_cycle.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_spinUp.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/max.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/max_spinUp.in

#~ mv /scratch/kallison/l2D_wTherm_test1* /data/dunham/kallison/eqcycle/data/wTherm/
