all: main

DEBUG_MODULES   = -DVERBOSE=1 -DODEPRINT=0
CFLAGS          = $(DEBUG_MODULES)
CPPFLAGS        = $(CFLAGS)
FFLAGS	        = -I${PETSC_DIR}/include/finclude
CLINKER		      = openmpicc

OBJECTS := domain.o mediator.o fault.o genFuncs.o\
 odeSolver.o rootFinder.o sbpOps_c.o \
 spmat.o powerLaw.o heatEquation.o \
 sbpOps_fc.o sbpOps_sc.o  sbpOps_fc_coordTrans.o \
 odeSolverImex.o odeSolver_WaveEq.o pressureEq.o \
 strikeSlip_linearElastic_qd.o mat_linearElastic.o \
 strikeSlip_powerLaw_qd.o mat_powerLaw.o


include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


main:  main.o $(OBJECTS)
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm main.o

FDP: FDP.o
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}

helloWorld: helloWorld.o
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm helloWorld.o

test: test.o
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm test.o

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
fault.o: fault.cpp fault.hpp genFuncs.hpp domain.hpp heatEquation.hpp \
 sbpOps.hpp sbpOps_c.hpp spmat.hpp sbpOps_fc.hpp sbpOps_fc_coordTrans.hpp \
 integratorContextEx.hpp odeSolver.hpp integratorContextImex.hpp \
 odeSolverImex.hpp rootFinderContext.hpp rootFinder.hpp
genFuncs.o: genFuncs.cpp genFuncs.hpp
heatEquation.o: heatEquation.cpp heatEquation.hpp genFuncs.hpp domain.hpp \
 sbpOps.hpp sbpOps_c.hpp spmat.hpp sbpOps_fc.hpp sbpOps_fc_coordTrans.hpp \
 integratorContextEx.hpp odeSolver.hpp integratorContextImex.hpp \
 odeSolverImex.hpp
helloWorld.o: helloWorld.cpp
linearElastic.o: linearElastic.cpp linearElastic.hpp \
 integratorContextEx.hpp genFuncs.hpp odeSolver.hpp \
 integratorContextImex.hpp momBalContext.hpp odeSolverImex.hpp domain.hpp \
 sbpOps.hpp sbpOps_c.hpp spmat.hpp sbpOps_fc.hpp sbpOps_fc_coordTrans.hpp \
 heatEquation.hpp
main.o: main.cpp genFuncs.hpp spmat.hpp domain.hpp sbpOps.hpp fault.hpp \
 heatEquation.hpp sbpOps_c.hpp sbpOps_fc.hpp sbpOps_fc_coordTrans.hpp \
 integratorContextEx.hpp odeSolver.hpp integratorContextImex.hpp \
 odeSolverImex.hpp rootFinderContext.hpp rootFinder.hpp linearElastic.hpp \
 momBalContext.hpp powerLaw.hpp pressureEq.hpp mediator.hpp \
 integratorContextWave.hpp odeSolver_WaveEq.hpp mat_linearElastic.hpp \
 mat_powerLaw.hpp strikeSlip_linearElastic_qd.hpp strikeSlip_powerLaw_qd.hpp
mainLinearElastic.o: mainLinearElastic.cpp genFuncs.hpp spmat.hpp \
 domain.hpp sbpOps.hpp sbpOps_fc.hpp sbpOps_c.hpp sbpOps_sc.hpp \
 sbpOps_fc_coordTrans.hpp fault.hpp heatEquation.hpp \
 integratorContextEx.hpp odeSolver.hpp integratorContextImex.hpp \
 odeSolverImex.hpp rootFinderContext.hpp rootFinder.hpp linearElastic.hpp \
 momBalContext.hpp
mat_linearElastic.o: mat_linearElastic.cpp mat_linearElastic.hpp \
 genFuncs.hpp domain.hpp sbpOps.hpp sbpOps_c.hpp spmat.hpp sbpOps_fc.hpp \
 sbpOps_fc_coordTrans.hpp
mat_powerLaw.o: mat_powerLaw.cpp mat_powerLaw.hpp genFuncs.hpp domain.hpp \
 heatEquation.hpp sbpOps.hpp sbpOps_c.hpp spmat.hpp sbpOps_fc.hpp \
 sbpOps_fc_coordTrans.hpp integratorContextEx.hpp odeSolver.hpp \
 integratorContextImex.hpp odeSolverImex.hpp
odeSolver.o: odeSolver.cpp odeSolver.hpp integratorContextEx.hpp \
 genFuncs.hpp
odeSolverImex.o: odeSolverImex.cpp odeSolverImex.hpp \
 integratorContextImex.hpp genFuncs.hpp odeSolver.hpp \
 integratorContextEx.hpp
odeSolver_WaveEq.o: odeSolver_WaveEq.cpp odeSolver_WaveEq.hpp \
 integratorContextWave.hpp genFuncs.hpp domain.hpp odeSolver.hpp \
 integratorContextEx.hpp
powerLaw.o: powerLaw.cpp powerLaw.hpp integratorContextEx.hpp \
 genFuncs.hpp odeSolver.hpp momBalContext.hpp domain.hpp \
 linearElastic.hpp integratorContextImex.hpp odeSolverImex.hpp sbpOps.hpp \
 sbpOps_c.hpp spmat.hpp sbpOps_fc.hpp sbpOps_fc_coordTrans.hpp \
 heatEquation.hpp
pressureEq.o: pressureEq.cpp pressureEq.hpp genFuncs.hpp domain.hpp \
 fault.hpp heatEquation.hpp sbpOps.hpp sbpOps_c.hpp spmat.hpp \
 sbpOps_fc.hpp sbpOps_fc_coordTrans.hpp integratorContextEx.hpp \
 odeSolver.hpp integratorContextImex.hpp odeSolverImex.hpp \
 rootFinderContext.hpp rootFinder.hpp
rootFinder.o: rootFinder.cpp rootFinder.hpp rootFinderContext.hpp
sbpOps_arrays.o: sbpOps_arrays.cpp sbpOps.hpp domain.hpp genFuncs.hpp
sbpOps_c.o: sbpOps_c.cpp sbpOps_c.hpp domain.hpp genFuncs.hpp spmat.hpp \
 sbpOps.hpp
sbpOps_fc_coordTrans.o: sbpOps_fc_coordTrans.cpp sbpOps_fc_coordTrans.hpp \
 domain.hpp genFuncs.hpp spmat.hpp sbpOps.hpp
sbpOps_fc.o: sbpOps_fc.cpp sbpOps_fc.hpp domain.hpp genFuncs.hpp \
 spmat.hpp sbpOps.hpp
sbpOps_sc.o: sbpOps_sc.cpp sbpOps_sc.hpp domain.hpp genFuncs.hpp \
 spmat.hpp sbpOps.hpp
spmat.o: spmat.cpp spmat.hpp
strikeSlip_linearElastic_qd.o: strikeSlip_linearElastic_qd.cpp strikeSlip_linearElastic_qd.hpp \
 integratorContextEx.hpp genFuncs.hpp odeSolver.hpp \
 integratorContextImex.hpp odeSolverImex.hpp domain.hpp sbpOps.hpp \
 sbpOps_c.hpp spmat.hpp sbpOps_fc.hpp sbpOps_fc_coordTrans.hpp fault.hpp \
 heatEquation.hpp rootFinderContext.hpp rootFinder.hpp pressureEq.hpp \
 mat_linearElastic.hpp
strikeSlip_powerLaw_qd.o: strikeSlip_powerLaw_qd.cpp \
 strikeSlip_powerLaw_qd.hpp integratorContextEx.hpp genFuncs.hpp \
 odeSolver.hpp integratorContextImex.hpp odeSolverImex.hpp domain.hpp \
 sbpOps.hpp sbpOps_c.hpp spmat.hpp sbpOps_fc.hpp sbpOps_fc_coordTrans.hpp \
 fault.hpp heatEquation.hpp rootFinderContext.hpp rootFinder.hpp \
 pressureEq.hpp mat_powerLaw.hpp
