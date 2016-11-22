#!/bin/bash
#PBS -N step3_pljs_nTB_psi_sN6_dc10_0_Ny201_Nz585_t1e16_step3
#PBS -l nodes=1:ppn=24
#PBS -q tgp
#PBS -V
#PBS -m n
#PBS -k oe
#PBS -e /data/dunham/kallison/eqcycle/nov21_outFiles/pljs_nTB_psi_sN6_dc10_0_Ny201_Nz585_t1e16_step3.err
#PBS -o /data/dunham/kallison/eqcycle/nov21_outFiles/pljs_nTB_psi_sN6_dc10_0_Ny201_Nz585_t1e16_step3.out


EXEC_DIR=/data/dunham/kallison/eqcycle
INIT_DIR=/data/dunham/kallison/eqcycle
cd $PBS_O_WORKDIR

#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/m2D.in
#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/max.in
#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/h2D.in

mpirun $EXEC_DIR/main $INIT_DIR/max_jellySandwich_cycle.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/max_jellySandwich_spinUp1.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_jellySandwich_cycle.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_jellySandwich_spinUp1.in


#~ mpirun $EXEC_DIR/main $INIT_DIR/max_cremeBrulee_cycle.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/max_cremeBrulee_spinUp1.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_cremeBrulee_cycle.in
#~ mpirun $EXEC_DIR/mainMaxwell $INIT_DIR/max_cremeBrulee_spinUp1.in

#~ mpirun $EXEC_DIR/mainLinearElastic $INIT_DIR/max.in
#~ mpirun $EXEC_DIR/main $INIT_DIR/max.in

