#!/bin/bash
#PBS -N ex1
#PBS -N SS_zlab50_l0.37
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/yyang/scycle/outFiles/ex1.err
#PBS -o /data/dunham/yyang/scycle/outFiles/ex1.out
#PBS -e /data/dunham/kallison/scycle/outFiles/SS_zlab50_l0.37.err
#PBS -o /data/dunham/kallison/scycle/outFiles/SS_zlab50_l0.37.out

EXEC_DIR=/data/dunham/yyang/scycle/source
INIT_DIR=/data/dunham/yyang/scycle/examples
cd $PBS_O_WORKDIR

# mpirun $EXEC_DIR/main $INIT_DIR/base_dynamic.in
# mpirun $EXEC_DIR/main $INIT_DIR/test_pressureEq.in
mpirun $EXEC_DIR/main $INIT_DIR/ex1.in
# mpirun $EXEC_DIR/main $INIT_DIR/ex2.in
# mpirun $EXEC_DIR/main $INIT_DIR/test_lm.in
mpirun $EXEC_DIR/main $INIT_DIR/test_grainSize.in
