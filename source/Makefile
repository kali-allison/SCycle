all: main

DEBUG_MODULES   = -DVERBOSE=1
CFLAGS          = $(DEBUG_MODULES)
CXXFLAGS        = $(CFLAGS) -std=c++14 -Wall -Werror -g
FFLAGS	        = -I${PETSC_DIR}/include/finclude
CLINKER		= openmpicc

OBJECTS := domain.o fault.o genFuncs.o\
 odeSolver.o rootFinder.o \
 linearElastic.o \
 dislocationCreep.o diffusionCreep.o dissolutionPrecipitationCreep.o pseudoplasticity.o powerLaw.o \
 heatEquation.o grainSizeEvolution.o \
 spmat.o sbpOps_m_constGrid.o sbpOps_m_varGrid.o \
 odeSolverImex.o odeSolver_WaveEq.o odeSolver_WaveImex.o pressureEq.o \
 strikeSlip_linearElastic_qd.o strikeSlip_powerLaw_qd.o \
 strikeSlip_linearElastic_fd.o strikeSlip_linearElastic_qd_fd.o strikeSlip_powerLaw_qd_fd.o


include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


main:  main.o $(OBJECTS)
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm main.o

FDP: FDP.o
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}

#.PHONY : clean
clean::
	-rm -f *.o main

depend:
	-g++ -MM *.c*

include ${PETSC_DIR}/lib/petsc/conf/test

#=========================================================
# Dependencies
#=========================================================
domain.o: domain.cpp domain.hpp genFuncs.hpp
dislocationCreep.o: dislocationCreep.cpp dislocationCreep.hpp genFuncs.hpp domain.hpp \
 heatEquation.hpp sbpOps.hpp spmat.hpp sbpOps_m_constGrid.hpp \
 sbpOps_m_varGrid.hpp
diffusionCreep.o: diffusionCreep.cpp diffusionCreep.hpp genFuncs.hpp domain.hpp \
 heatEquation.hpp sbpOps.hpp spmat.hpp sbpOps_m_constGrid.hpp \
 sbpOps_m_varGrid.hpp
dissolutionPrecipitationCreep.o: dissolutionPrecipitationCreep.cpp dissolutionPrecipitationCreep.hpp genFuncs.hpp domain.hpp \
 heatEquation.hpp sbpOps.hpp spmat.hpp sbpOps_m_constGrid.hpp \
 sbpOps_m_varGrid.hpp
pseudoplasticity.o: pseudoplasticity.cpp pseudoplasticity.hpp genFuncs.hpp domain.hpp \
 sbpOps.hpp spmat.hpp sbpOps_m_constGrid.hpp \
 sbpOps_m_varGrid.hpp
fault.o: fault.cpp fault.hpp genFuncs.hpp domain.hpp \
 rootFinderContext.hpp rootFinder.hpp
genFuncs.o: genFuncs.cpp genFuncs.hpp
grainSizeEvolution.o: grainSizeEvolution.cpp grainSizeEvolution.hpp \
 genFuncs.hpp domain.hpp heatEquation.hpp
heatEquation.o: heatEquation.cpp heatEquation.hpp genFuncs.hpp domain.hpp \
 sbpOps.hpp spmat.hpp sbpOps_m_constGrid.hpp sbpOps_m_varGrid.hpp \
 integratorContextEx.hpp odeSolver.hpp integratorContextImex.hpp \
 odeSolverImex.hpp
linearElastic.o: linearElastic.cpp linearElastic.hpp genFuncs.hpp \
 domain.hpp sbpOps.hpp spmat.hpp sbpOps_m_constGrid.hpp \
 sbpOps_m_varGrid.hpp
main.o: main.cpp genFuncs.hpp spmat.hpp domain.hpp sbpOps.hpp fault.hpp \
 rootFinderContext.hpp rootFinder.hpp linearElastic.hpp \
 sbpOps_m_constGrid.hpp sbpOps_m_varGrid.hpp heatEquation.hpp \
 dislocationCreep.hpp diffusionCreep.hpp dissolutionPrecipitationCreep.hpp pseudoplasticity.hpp powerLaw.hpp \
 problemContext.hpp integratorContextEx.hpp odeSolver.hpp integratorContextImex.hpp \
 odeSolverImex.hpp pressureEq.hpp \
 strikeSlip_linearElastic_qd.hpp strikeSlip_linearElastic_fd.hpp \
 integratorContext_WaveEq.hpp odeSolver_WaveEq.hpp \
 strikeSlip_linearElastic_qd_fd.hpp integratorContext_WaveEq_Imex.hpp \
 odeSolver_WaveImex.hpp strikeSlip_powerLaw_qd.hpp
mainLinearElastic.o: mainLinearElastic.cpp genFuncs.hpp spmat.hpp \
 domain.hpp sbpOps.hpp sbpOps_m_constGrid.hpp sbpOps_sc.hpp \
 sbpOps_m_varGrid.hpp fault.hpp rootFinderContext.hpp rootFinder.hpp \
 linearElastic.hpp
odeSolver.o: odeSolver.cpp odeSolver.hpp integratorContextEx.hpp \
 genFuncs.hpp
odeSolverImex.o: odeSolverImex.cpp odeSolverImex.hpp \
 integratorContextImex.hpp genFuncs.hpp odeSolver.hpp \
 integratorContextEx.hpp
odeSolver_WaveEq.o: odeSolver_WaveEq.cpp odeSolver_WaveEq.hpp \
 integratorContext_WaveEq.hpp genFuncs.hpp odeSolver.hpp \
 integratorContextEx.hpp
odeSolver_WaveImex.o: odeSolver_WaveImex.cpp odeSolver_WaveImex.hpp \
 integratorContext_WaveEq_Imex.hpp genFuncs.hpp odeSolver.hpp \
 integratorContextEx.hpp
powerLaw.o: powerLaw.cpp powerLaw.hpp genFuncs.hpp domain.hpp \
 heatEquation.hpp sbpOps.hpp spmat.hpp sbpOps_m_constGrid.hpp \
 sbpOps_m_varGrid.hpp integratorContextEx.hpp odeSolver.hpp \
 integratorContextImex.hpp odeSolverImex.hpp \
 dislocationCreep.hpp diffusionCreep.hpp dissolutionPrecipitationCreep.hpp pseudoplasticity.hpp
pressureEq.o: pressureEq.cpp pressureEq.hpp genFuncs.hpp domain.hpp \
 fault.hpp rootFinderContext.hpp rootFinder.hpp sbpOps.hpp \
 spmat.hpp sbpOps_m_constGrid.hpp sbpOps_m_varGrid.hpp integratorContextEx.hpp \
 odeSolver.hpp integratorContextImex.hpp
rootFinder.o: rootFinder.cpp rootFinder.hpp rootFinderContext.hpp
sbpOps_m_varGrid.o: sbpOps_m_varGrid.cpp sbpOps_m_varGrid.hpp \
 domain.hpp genFuncs.hpp spmat.hpp sbpOps.hpp
sbpOps_m_constGrid.o: sbpOps_m_constGrid.cpp sbpOps_m_constGrid.hpp domain.hpp genFuncs.hpp \
 spmat.hpp sbpOps.hpp
spmat.o: spmat.cpp spmat.hpp
strikeSlip_linearElastic_fd.o: strikeSlip_linearElastic_fd.cpp \
 strikeSlip_linearElastic_fd.hpp integratorContext_WaveEq.hpp \
 genFuncs.hpp odeSolver.hpp integratorContextEx.hpp odeSolver_WaveEq.hpp \
 domain.hpp sbpOps.hpp spmat.hpp sbpOps_m_constGrid.hpp \
 sbpOps_m_varGrid.hpp fault.hpp rootFinderContext.hpp rootFinder.hpp \
 pressureEq.hpp problemContext.hpp integratorContextImex.hpp heatEquation.hpp \
 odeSolverImex.hpp linearElastic.hpp
strikeSlip_linearElastic_qd.o: strikeSlip_linearElastic_qd.cpp \
 strikeSlip_linearElastic_qd.hpp integratorContextEx.hpp problemContext.hpp genFuncs.hpp \
 odeSolver.hpp integratorContextImex.hpp odeSolverImex.hpp domain.hpp \
 sbpOps.hpp spmat.hpp sbpOps_m_constGrid.hpp sbpOps_m_varGrid.hpp \
 fault.hpp rootFinderContext.hpp rootFinder.hpp pressureEq.hpp \
 heatEquation.hpp linearElastic.hpp
strikeSlip_linearElastic_qd_fd.o: strikeSlip_linearElastic_qd_fd.cpp \
 strikeSlip_linearElastic_qd_fd.hpp integratorContextEx.hpp problemContext.hpp genFuncs.hpp \
 odeSolver.hpp integratorContextImex.hpp integratorContext_WaveEq.hpp \
 integratorContext_WaveEq_Imex.hpp odeSolverImex.hpp odeSolver_WaveEq.hpp \
 odeSolver_WaveImex.hpp domain.hpp sbpOps.hpp spmat.hpp \
 sbpOps_m_constGrid.hpp sbpOps_m_varGrid.hpp fault.hpp rootFinderContext.hpp \
 rootFinder.hpp pressureEq.hpp heatEquation.hpp linearElastic.hpp
strikeSlip_powerLaw_qd.o: strikeSlip_powerLaw_qd.cpp \
 strikeSlip_powerLaw_qd.hpp integratorContextEx.hpp problemContext.hpp genFuncs.hpp \
 odeSolver.hpp integratorContextImex.hpp odeSolverImex.hpp domain.hpp \
 sbpOps.hpp spmat.hpp sbpOps_m_constGrid.hpp sbpOps_m_varGrid.hpp \
 fault.hpp rootFinderContext.hpp rootFinder.hpp pressureEq.hpp \
 heatEquation.hpp powerLaw.hpp grainSizeEvolution.hpp \
 dislocationCreep.hpp diffusionCreep.hpp dissolutionPrecipitationCreep.hpp pseudoplasticity.hpp
strikeSlip_powerLaw_qd_fd.o: strikeSlip_powerLaw_qd_fd.cpp \
 strikeSlip_powerLaw_qd_fd.hpp integratorContextEx.hpp problemContext.hpp genFuncs.hpp \
 odeSolver.hpp integratorContextImex.hpp odeSolverImex.hpp domain.hpp \
 sbpOps.hpp spmat.hpp sbpOps_m_constGrid.hpp sbpOps_m_varGrid.hpp \
 fault.hpp rootFinderContext.hpp rootFinder.hpp pressureEq.hpp \
 heatEquation.hpp powerLaw.hpp grainSizeEvolution.hpp \
 dislocationCreep.hpp diffusionCreep.hpp dissolutionPrecipitationCreep.hpp pseudoplasticity.hpp
