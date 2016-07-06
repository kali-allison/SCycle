all: mainMaxwell

DEBUG_MODULES   = -DVERBOSE=1 -DODEPRINT=0 -DCALCULATE_ENERGY=1 -DLOCK_FAULT=0
CFLAGS          = $(DEBUG_MODULES)
CPPFLAGS        = $(CFLAGS)
FFLAGS	        = -I${PETSC_DIR}/include/finclude
CLINKER		      = openmpicc

OBJECTS := domain.o debuggingFuncs.o fault.o genFuncs.o linearElastic.o\
 maxwellViscoelastic.o odeSolver.o rootFinder.o sbpOps_c.o sbpOps_fc.o\
 spmat.o powerLaw.o sbpOps_sc.o heatEquation.o sbpOps_fc_coordTrans.o \
 odeSolverImex.o



include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


main:  main.o $(OBJECTS)
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm main.o

mainMaxwell:  mainMaxwell.o $(OBJECTS)
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm mainMaxwell.o


mainLinearElastic:  mainLinearElastic.o $(OBJECTS)
	-${CLINKER} $^ $(CFLAGS) -o $@ ${PETSC_SYS_LIB} $(CFLAGS)
#~	-rm mainLinearElastic.o

main_iceSheet:  main_iceSheet.o $(OBJECTS)
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm main_iceSheet.o

FDP: FDP.o
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}

mainEx: mainEx.o $(OBJECTS)
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm mainEx.o

testMain: testMain.o $(OBJECTS)
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm testMain.o

helloWorld: helloWorld.o
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm helloWorld.o

test: test.o
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm test.o

#.PHONY : clean
clean::
	-rm -f *.o main helloWorld mainLinearElastic mainMaxwell

depend:
	-g++ -MM *.c*

include ${PETSC_DIR}/lib/petsc/conf/test

#=========================================================
# Dependencies
#=========================================================
debuggingFuncs.o: debuggingFuncs.cpp debuggingFuncs.hpp genFuncs.hpp
domain.o: domain.cpp domain.hpp genFuncs.hpp
fault.o: fault.cpp fault.hpp genFuncs.hpp domain.hpp \
 rootFinderContext.hpp rootFinder.hpp
genFuncs.o: genFuncs.cpp genFuncs.hpp
heatEquation.o: heatEquation.cpp heatEquation.hpp genFuncs.hpp domain.hpp \
 sbpOps.hpp sbpOps_c.hpp debuggingFuncs.hpp spmat.hpp sbpOps_fc.hpp
helloWorld.o: helloWorld.cpp
iceSheet.o: iceSheet.cpp iceSheet.hpp integratorContextEx.hpp odeSolver.hpp \
 genFuncs.hpp domain.hpp maxwellViscoelastic.hpp linearElastic.hpp \
 sbpOps.hpp sbpOps_c.hpp debuggingFuncs.hpp spmat.hpp sbpOps_fc.hpp \
 fault.hpp rootFinderContext.hpp rootFinder.hpp
linearElastic.o: linearElastic.cpp linearElastic.hpp \
 integratorContextEx.hpp odeSolver.hpp genFuncs.hpp domain.hpp sbpOps.hpp \
 sbpOps_c.hpp debuggingFuncs.hpp spmat.hpp sbpOps_fc.hpp fault.hpp \
 rootFinderContext.hpp rootFinder.hpp heatEquation.hpp
main.o: main.cpp genFuncs.hpp spmat.hpp domain.hpp sbpOps.hpp fault.hpp \
 rootFinderContext.hpp rootFinder.hpp linearElastic.hpp \
 integratorContextEx.hpp odeSolver.hpp sbpOps_c.hpp debuggingFuncs.hpp \
 sbpOps_fc.hpp maxwellViscoelastic.hpp powerLaw.hpp
mainEx.o: mainEx.cpp
mainLinearElastic.o: mainLinearElastic.cpp genFuncs.hpp spmat.hpp \
 domain.hpp sbpOps.hpp sbpOps_fc.hpp debuggingFuncs.hpp sbpOps_c.hpp \
 sbpOps_sc.hpp fault.hpp rootFinderContext.hpp rootFinder.hpp \
 linearElastic.hpp integratorContextEx.hpp integratorContextImex.hpp \
 odeSolver.hpp odeSolverImex.hpp sbpOps_fc_coordTrans.hpp
mainMaxwell.o: mainMaxwell.cpp genFuncs.hpp spmat.hpp domain.hpp \
 sbpOps.hpp fault.hpp rootFinderContext.hpp rootFinder.hpp \
 linearElastic.hpp integratorContextEx.hpp odeSolver.hpp sbpOps_c.hpp \
 debuggingFuncs.hpp sbpOps_fc.hpp maxwellViscoelastic.hpp powerLaw.hpp
main_iceSheet.o: main_iceSheet.cpp genFuncs.hpp spmat.hpp domain.hpp \
 sbpOps.hpp fault.hpp rootFinderContext.hpp rootFinder.hpp \
 linearElastic.hpp integratorContextEx.hpp odeSolver.hpp sbpOps_c.hpp \
 debuggingFuncs.hpp sbpOps_fc.hpp maxwellViscoelastic.hpp iceSheet.hpp
maxwellViscoelastic.o: maxwellViscoelastic.cpp maxwellViscoelastic.hpp \
 integratorContextEx.hpp odeSolver.hpp genFuncs.hpp domain.hpp \
 linearElastic.hpp sbpOps.hpp sbpOps_c.hpp debuggingFuncs.hpp spmat.hpp \
 sbpOps_fc.hpp fault.hpp rootFinderContext.hpp rootFinder.hpp heatEquation.hpp
odeSolver.o: odeSolver.cpp odeSolver.hpp integratorContextEx.hpp \
 genFuncs.hpp
odeSolverImex.o: odeSolverImex.cpp odeSolverImex.hpp integratorContextImex.hpp \
 genFuncs.hpp
powerLaw.o: powerLaw.cpp powerLaw.hpp integratorContextEx.hpp odeSolver.hpp \
 genFuncs.hpp domain.hpp linearElastic.hpp sbpOps.hpp sbpOps_c.hpp \
 debuggingFuncs.hpp spmat.hpp sbpOps_fc.hpp fault.hpp \
 rootFinderContext.hpp rootFinder.hpp heatEquation.hpp
rootFinder.o: rootFinder.cpp rootFinder.hpp rootFinderContext.hpp
sbpOps_arrays.o: sbpOps_arrays.cpp sbpOps.hpp domain.hpp genFuncs.hpp
sbpOps_c.o: sbpOps_c.cpp sbpOps_c.hpp domain.hpp genFuncs.hpp \
 debuggingFuncs.hpp spmat.hpp sbpOps.hpp
sbpOps_fc.o: sbpOps_fc.cpp sbpOps_fc.hpp domain.hpp genFuncs.hpp \
 debuggingFuncs.hpp spmat.hpp sbpOps.hpp
sbpOps_fc_coordTrans.o: sbpOps_fc_coordTrans.cpp sbpOps_fc_coordTrans.hpp \
 domain.hpp genFuncs.hpp debuggingFuncs.hpp spmat.hpp sbpOps.hpp
sbpOps_sc.o: sbpOps_sc.cpp sbpOps_sc.hpp domain.hpp genFuncs.hpp \
 debuggingFuncs.hpp spmat.hpp sbpOps.hpp
sbpOps_temp.o: sbpOps_temp.cpp sbpOps.hpp domain.hpp genFuncs.hpp
spmat.o: spmat.cpp spmat.hpp
test.o: test.cpp sbpOps.hpp domain.hpp genFuncs.hpp sbpOps_c.hpp \
 debuggingFuncs.hpp spmat.hpp
testMain.o: testMain.cpp genFuncs.hpp domain.hpp spmat.hpp sbpOps.hpp \
 sbpOps_c.hpp debuggingFuncs.hpp sbpOps_fc.hpp testOdeSolver.hpp \
 integratorContextEx.hpp odeSolver.hpp
test.o: test.cpp

