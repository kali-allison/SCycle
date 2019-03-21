#!/bin/bash
<<<<<<< HEAD
#PBS -N ice_test
=======
#PBS -N ex1
#PBS -N SS_zlab50_l0.37
>>>>>>> master
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
<<<<<<< HEAD
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

=======
#PBS -e /data/dunham/yyang/scycle/outFiles/ex1.err
#PBS -o /data/dunham/yyang/scycle/outFiles/ex1.out
#PBS -e /data/dunham/kallison/scycle/outFiles/SS_zlab50_l0.37.err
#PBS -o /data/dunham/kallison/scycle/outFiles/SS_zlab50_l0.37.out

EXEC_DIR=/data/dunham/yyang/scycle/source
INIT_DIR=/data/dunham/yyang/scycle/examples
cd $PBS_O_WORKDIR

module load compilers/intel16
module load mpich/intelmpi
# the above 2 lines need to be run before compiling the code
module load CLAB-5/netcdf-4.4.1

mpdboot
#~ mpirun $EXEC_DIR/main $INIT_DIR/base_dynamic.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/test_pressureEq.in
mpirun $EXEC_DIR/main $INIT_DIR/ex1.in
mpdallexit
>>>>>>> master
