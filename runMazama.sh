#!/bin/bash
#PBS -N mms
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/eqcycle/stdoutFiles/mms.err
#PBS -o /data/dunham/kallison/eqcycle/stdoutFiles/mms.out


EXEC_DIR=/data/dunham/kallison/eqcycle
INIT_DIR=/data/dunham/kallison/eqcycle
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/m2D.in
#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/h2D.in

#~ mpirun $EXEC_DIR/main $INIT_DIR/max_jellySandwich_cycle.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_jellySandwich_spinUp1.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_jellySandwich_spinUp2.in

mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/mms.in

#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_cremeBrulee_spinUp1.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_jellySandwich_spinUp2.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/max_cremeBrulee_cycle.in



#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_cremeBrulee_spinUp2.in
