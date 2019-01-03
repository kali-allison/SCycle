#!/bin/bash
#PBS -N ex2
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/scycle/outFiles/ex2.err
#PBS -o /data/dunham/kallison/scycle/outFiles/ex2.out

EXEC_DIR=/data/dunham/kallison/scycle/source
INIT_DIR=/data/dunham/kallison/scycle/in
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/main $INIT_DIR/base_dynamic.in
mpirun $EXEC_DIR/main $INIT_DIR/ex2.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/test.in
#mpirun $EXEC_DIR/main $INIT_DIR/ice.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/ice_pl2.in

