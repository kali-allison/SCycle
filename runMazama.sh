#!/bin/bash
#PBS -N wvsh_zlab60_l0.37_w10
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/scycle/outFiles/wvsh_zlab60_l0.37_w10.err
#PBS -o /data/dunham/kallison/scycle/outFiles/wvsh_zlab60_l0.37_w10.out

EXEC_DIR=/data/dunham/kallison/scycle
INIT_DIR=/data/dunham/kallison/scycle/tempIn
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/main $INIT_DIR/he_ssits.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/he_eqs.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/linEl.in
mpirun $EXEC_DIR/main $INIT_DIR/SS_wvsh_zlab60_l0.37_w10.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/heCont.in

