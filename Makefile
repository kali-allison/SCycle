all: main

DEBUG_MODULES   = -DVERBOSE=1 -DODEPRINT=0
CFLAGS          = $(DEBUG_MODULES)
CPPFLAGS        = $(CFLAGS) -std=c++11 -O3
FFLAGS	        = -I${PETSC_DIR}/include/finclude
CLINKER		      = openmpicc

OBJECTS := domain.o fault.o genFuncs.o\
 odeSolver.o rootFinder.o \
 linearElastic.o powerLaw.o heatEquation.o \
 spmat.o sbpOps_c.o sbpOps_fc.o sbpOps_sc.o  sbpOps_fc_coordTrans.o \
 odeSolverImex.o odeSolver_WaveEq.o odeSolver_WaveImex.o pressureEq.o \
 strikeSlip_linearElastic_qd.o strikeSlip_powerLaw_qd.o powerLaw.o iceStream_linearElastic_qd.o \
 strikeSlip_linearElastic_fd.o strikeSlip_linearElastic_qd_fd.o strikeSlip_powerLaw_qd_fd.o


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
fault.o: fault.cpp fault.hpp genFuncs.hpp domain.hpp \
 rootFinderContext.hpp rootFinder.hpp
genFuncs.o: genFuncs.cpp genFuncs.hpp
heatEquation.o: heatEquation.cpp heatEquation.hpp genFuncs.hpp domain.hpp \
 sbpOps.hpp sbpOps_c.hpp spmat.hpp sbpOps_fc.hpp sbpOps_fc_coordTrans.hpp \
 integratorContextEx.hpp odeSolver.hpp integratorContextImex.hpp \
 odeSolverImex.hpp
helloWorld.o: helloWorld.cpp
iceStream_linearElastic_qd.o: iceStream_linearElastic_qd.cpp \
 iceStream_linearElastic_qd.hpp integratorContextEx.hpp genFuncs.hpp \
 odeSolver.hpp integratorContextImex.hpp odeSolverImex.hpp domain.hpp \
 sbpOps.hpp sbpOps_c.hpp spmat.hpp sbpOps_fc.hpp sbpOps_fc_coordTrans.hpp \
 fault.hpp rootFinderContext.hpp rootFinder.hpp pressureEq.hpp \
 heatEquation.hpp linearElastic.hpp
linearElastic.o: linearElastic.cpp linearElastic.hpp genFuncs.hpp \
 domain.hpp sbpOps.hpp sbpOps_c.hpp spmat.hpp sbpOps_fc.hpp \
 sbpOps_fc_coordTrans.hpp
main.o: main.cpp genFuncs.hpp spmat.hpp domain.hpp sbpOps.hpp fault.hpp \
 rootFinderContext.hpp rootFinder.hpp linearElastic.hpp sbpOps_c.hpp \
 sbpOps_fc.hpp sbpOps_fc_coordTrans.hpp powerLaw.hpp heatEquation.hpp \
 integratorContextEx.hpp odeSolver.hpp integratorContextImex.hpp \
 odeSolverImex.hpp pressureEq.hpp iceStream_linearElastic_qd.hpp \
 strikeSlip_linearElastic_qd.hpp strikeSlip_linearElastic_fd.hpp \
 integratorContext_WaveEq.hpp odeSolver_WaveEq.hpp \
 strikeSlip_linearElastic_qd_fd.hpp integratorContext_WaveEq_Imex.hpp \
 odeSolver_WaveImex.hpp strikeSlip_powerLaw_qd.hpp
mainLinearElastic.o: mainLinearElastic.cpp genFuncs.hpp spmat.hpp \
 domain.hpp sbpOps.hpp sbpOps_fc.hpp sbpOps_c.hpp sbpOps_sc.hpp \
 sbpOps_fc_coordTrans.hpp fault.hpp rootFinderContext.hpp rootFinder.hpp \
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
 heatEquation.hpp sbpOps.hpp sbpOps_c.hpp spmat.hpp sbpOps_fc.hpp \
 sbpOps_fc_coordTrans.hpp integratorContextEx.hpp odeSolver.hpp \
 integratorContextImex.hpp odeSolverImex.hpp
pressureEq.o: pressureEq.cpp pressureEq.hpp genFuncs.hpp domain.hpp \
 fault.hpp rootFinderContext.hpp rootFinder.hpp sbpOps.hpp sbpOps_c.hpp \
 spmat.hpp sbpOps_fc.hpp sbpOps_fc_coordTrans.hpp integratorContextEx.hpp \
 odeSolver.hpp integratorContextImex.hpp
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
strikeSlip_linearElastic_fd.o: strikeSlip_linearElastic_fd.cpp \
 strikeSlip_linearElastic_fd.hpp integratorContext_WaveEq.hpp \
 genFuncs.hpp odeSolver.hpp integratorContextEx.hpp odeSolver_WaveEq.hpp \
 domain.hpp sbpOps.hpp sbpOps_c.hpp spmat.hpp sbpOps_fc.hpp \
 sbpOps_fc_coordTrans.hpp fault.hpp rootFinderContext.hpp rootFinder.hpp \
 pressureEq.hpp integratorContextImex.hpp heatEquation.hpp \
 odeSolverImex.hpp linearElastic.hpp
strikeSlip_linearElastic_qd.o: strikeSlip_linearElastic_qd.cpp \
 strikeSlip_linearElastic_qd.hpp integratorContextEx.hpp genFuncs.hpp \
 odeSolver.hpp integratorContextImex.hpp odeSolverImex.hpp domain.hpp \
 sbpOps.hpp sbpOps_c.hpp spmat.hpp sbpOps_fc.hpp sbpOps_fc_coordTrans.hpp \
 fault.hpp rootFinderContext.hpp rootFinder.hpp pressureEq.hpp \
 heatEquation.hpp linearElastic.hpp
strikeSlip_linearElastic_qd_fd.o: strikeSlip_linearElastic_qd_fd.cpp \
 strikeSlip_linearElastic_qd_fd.hpp integratorContextEx.hpp genFuncs.hpp \
 odeSolver.hpp integratorContextImex.hpp integratorContext_WaveEq.hpp \
 integratorContext_WaveEq_Imex.hpp odeSolverImex.hpp odeSolver_WaveEq.hpp \
 odeSolver_WaveImex.hpp domain.hpp sbpOps.hpp sbpOps_c.hpp spmat.hpp \
 sbpOps_fc.hpp sbpOps_fc_coordTrans.hpp fault.hpp rootFinderContext.hpp \
 rootFinder.hpp pressureEq.hpp heatEquation.hpp linearElastic.hpp
strikeSlip_powerLaw_qd.o: strikeSlip_powerLaw_qd.cpp \
 strikeSlip_powerLaw_qd.hpp integratorContextEx.hpp genFuncs.hpp \
 odeSolver.hpp integratorContextImex.hpp odeSolverImex.hpp domain.hpp \
 sbpOps.hpp sbpOps_c.hpp spmat.hpp sbpOps_fc.hpp sbpOps_fc_coordTrans.hpp \
 fault.hpp rootFinderContext.hpp rootFinder.hpp pressureEq.hpp \
 heatEquation.hpp powerLaw.hpp
strikeSlip_powerLaw_qd_fd.o: strikeSlip_powerLaw_qd_fd.cpp \
 strikeSlip_powerLaw_qd_fd.hpp integratorContextEx.hpp genFuncs.hpp \
 odeSolver.hpp integratorContextImex.hpp odeSolverImex.hpp domain.hpp \
 sbpOps.hpp sbpOps_c.hpp spmat.hpp sbpOps_fc.hpp sbpOps_fc_coordTrans.hpp \
 fault.hpp rootFinderContext.hpp rootFinder.hpp pressureEq.hpp \
 heatEquation.hpp powerLaw.hpp
