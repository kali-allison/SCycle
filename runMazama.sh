#!/bin/bash
#PBS -N test
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/eqcycle/outFiles/test.err
#PBS -o /data/dunham/kallison/eqcycle/outFiles/test.out


EXEC_DIR=/data/dunham/kallison/eqcycle
INIT_DIR=/data/dunham/kallison/eqcycle
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/max_cremeBrulee_cycle.in
#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/m2D.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/m2D.in
#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/max.in


#~ mpirun $EXEC_DIR/main $INIT_DIR/max_cremeBrulee_cycle.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/max_cremeBrulee_spinUp.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_cremeBrulee_cycle.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_cremeBrulee_spinUp.in


#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/he.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/he.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/he.in

#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/rs.in
#~ rm -f /data/dunham/kallison/eqcycle/heOutFiles/*.err
#~ rm -f /data/dunham/kallison/eqcycle/heOutFiles/*.out
#~ rm -f /scratch/kallison/heSims/*
mpirun $EXEC_DIR/main $INIT_DIR/he2D.in
#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/he2D.in

