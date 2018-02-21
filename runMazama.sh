#!/bin/bash
<<<<<<< HEAD
<<<<<<< HEAD
#PBS -N test
=======
#PBS -N v31
>>>>>>> e5c095440c0fdf90cd990451bdb069d9c2ec21b0
=======
#PBS -N ice
>>>>>>> lockFault
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
<<<<<<< HEAD
<<<<<<< HEAD
#PBS -e /data/dunham/kallison/eqcycle/outFiles/test.err
#PBS -o /data/dunham/kallison/eqcycle/outFiles/test.out
=======
#PBS -e /data/dunham/kallison/eqcycle/outFiles/v31.err
#PBS -o /data/dunham/kallison/eqcycle/outFiles/v31.out
>>>>>>> e5c095440c0fdf90cd990451bdb069d9c2ec21b0
=======
#PBS -e /data/dunham/kallison/eqcycle/outFiles/ice.err
#PBS -o /data/dunham/kallison/eqcycle/outFiles/ice.out
>>>>>>> lockFault

EXEC_DIR=/data/dunham/kallison/eqcycle
INIT_DIR=/data/dunham/kallison/eqcycle/in
cd $PBS_O_WORKDIR

<<<<<<< HEAD
mpirun $EXEC_DIR/main $INIT_DIR/test.in
<<<<<<< HEAD
#~ mpirun $EXEC_DIR/main $INIT_DIR/spinUpTest.in
=======
#~ mpirun $EXEC_DIR/main $INIT_DIR/ex2.in
>>>>>>> e5c095440c0fdf90cd990451bdb069d9c2ec21b0
=======
#~ mpirun $EXEC_DIR/main $INIT_DIR/test.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/test_fh.in
mpirun $EXEC_DIR/main $INIT_DIR/ex3.in
>>>>>>> lockFault

