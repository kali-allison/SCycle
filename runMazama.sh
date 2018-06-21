#!/bin/bash
#PBS -N qdfd_al_nct_qd2fd5e-3_fd2qd1e-3
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/scycle/outFiles/qdfd_al_nct_qd2fd5e-3_fd2qd1e-3.err
#PBS -o /data/dunham/kallison/scycle/outFiles/qdfd_al_nct_qd2fd5e-3_fd2qd1e-3.out

EXEC_DIR=/data/dunham/kallison/scycle
INIT_DIR=/data/dunham/kallison/scycle/in
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/main $INIT_DIR/he_ssits.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/he_eqs.in
mpirun $EXEC_DIR/main $INIT_DIR/dynamic.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/quasidynamic.in

