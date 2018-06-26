#!/bin/bash
#PBS -N qd_sl_nct_Ny501_Nz501
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/scycle/outFiles/qd_sl_nct_Ny501_Nz501.err
#PBS -o /data/dunham/kallison/scycle/outFiles/qd_sl_nct_Ny501_Nz501.out

EXEC_DIR=/data/dunham/kallison/scycle
INIT_DIR=/data/dunham/kallison/scycle/in
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/main $INIT_DIR/he_ssits.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/he_eqs.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/qdfdcycles.in
mpirun $EXEC_DIR/main $INIT_DIR/quasidynamic.in

