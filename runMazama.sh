#!/bin/bash
#PBS -N qd_le
#PBS -l nodes=2:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/scycle/outFiles/qd_le.err
#PBS -o /data/dunham/kallison/scycle/outFiles/qd_le.out

EXEC_DIR=/data/dunham/kallison/scycle
INIT_DIR=/data/dunham/kallison/scycle/in
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/main $INIT_DIR/he_ssits.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/he_eqs.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/qdfdcycles.in
mpirun $EXEC_DIR/main $INIT_DIR/testQD.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/qdSS.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/quasidynamic.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/singleDynamicRupture.in

