#!/bin/bash
#PBS -N t01.5e14_pljs_blansVW_cycle_dc16e-3_Ny201_Nz451
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/eqcycle/stdoutFiles/pljs_blansVW_cycle_dc16e-3_t01.5e14_Ny201_Nz451.err
#PBS -o /data/dunham/kallison/eqcycle/stdoutFiles/pljs_blansVW_cycle_dc16e-3_t01.5e14_Ny201_Nz451.out


EXEC_DIR=/data/dunham/kallison/eqcycle
INIT_DIR=/data/dunham/kallison/eqcycle
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/m2D.in
#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/h2D.in

mpirun $EXEC_DIR/main $INIT_DIR/max_jellySandwich_cycle.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_jellySandwich_spinUp1.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_jellySandwich_spinUp2.in

#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_cremeBrulee_spinUp1.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_jellySandwich_spinUp2.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/max_cremeBrulee_cycle.in



#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_cremeBrulee_spinUp2.in
