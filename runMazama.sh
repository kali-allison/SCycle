#!/bin/bash
#PBS -N qdfd_pl_qd2fd1e-3_fd2qd1e-4_zlab50
#PBS -l nodes=2:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/scycle/outFiles/qdfd_pl_qd2fd1e-3_fd2qd1e-4_zlab50.err
#PBS -o /data/dunham/kallison/scycle/outFiles/qdfd_pl_qd2fd1e-3_fd2qd1e-4_zlab50.out

EXEC_DIR=/data/dunham/kallison/scycle
INIT_DIR=/data/dunham/kallison/scycle/in
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/main $INIT_DIR/he_ssits.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/he_eqs.in
mpirun $EXEC_DIR/main $INIT_DIR/qdfdcycles.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/qdSS.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/quasidynamic.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/singleDynamicRupture.in

