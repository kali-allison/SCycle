#!/bin/bash
#PBS -N smallTests/pljs_step1
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/eqcycle/outFiles/pljs_step1.err
#PBS -o /data/dunham/kallison/eqcycle/outFiles/pljs_step1.out


EXEC_DIR=/data/dunham/kallison/eqcycle
INIT_DIR=/data/dunham/kallison/eqcycle
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/m2D.in
#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/h2D.in

#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_jellySandwich_cycle.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_jellySandwich_spinUp1.in


#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_cremeBrulee_cycle.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_cremeBrulee_spinUp1.in

#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/max.in
mpirun $EXEC_DIR/main $INIT_DIR/max.in

