#!/bin/bash
#PBS -N ice_test
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/scycle/outFiles/ice_test.err
#PBS -o /data/dunham/kallison/scycle/outFiles/ice_test.out

EXEC_DIR=/data/dunham/kallison/scycle/source
#~ INIT_DIR=/data/dunham/kallison/scycle/in
#~ INIT_DIR=/data/dunham/kallison/grainSizeEv/in
INIT_DIR=/data/dunham/kallison/lmData/in
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/main $INIT_DIR/test_lm.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/grainSize_eqs.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/cs_wdisl_ndiff.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/cs_wdisl_wdiff_constGrainSize.in


#~ mpirun $EXEC_DIR/main $INIT_DIR/lm_VW_1e25_1e19.in
mpirun $EXEC_DIR/main $INIT_DIR/Ice-VW-Toy-test-1b.in

