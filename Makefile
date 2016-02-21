all: mainLinearElastic main mainMaxwell

DEBUG_MODULES   = -DVERBOSE=0 -DODEPRINT=0 -DDEBUG=0 -DVERSION=${PETSC_VERSION_NUM}
CFLAGS          = $(DEBUG_MODULES)
FFLAGS	        = -I${PETSC_DIR}/include/finclude
CLINKER		      = openmpicc

OBJECTS := domain.o debuggingFuncs.o fault.o genFuncs.o linearElastic.o\
 maxwellViscoelastic.o odeSolver.o rootFinder.o sbpOps_c.o sbpOps_fc.o\
 spmat.o testOdeSolver.o powerLaw.o sbpOps_sc.o



include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules


trial:
	@echo ${PETSC_DIR}/conf/variables

main:  main.o $(OBJECTS)
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	echo ${PETSC_DIR}
	-rm main.o

mainMaxwell:  mainMaxwell.o $(OBJECTS)
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm mainMaxwell.o

mainLinearElastic:  mainLinearElastic.o $(OBJECTS)
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm mainLinearElastic.o

test:  test.o domain.o sbpOps_c.o spmat.o genFuncs.o
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm test.o

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

#.PHONY : clean
clean::
	-rm -f *.o main helloWorld mainLinearElastic mainMaxwell

depend:
	-g++ -MM *.c*

include ${PETSC_DIR}/conf/test

#=========================================================
# Dependencies
#=========================================================
debuggingFuncs.o: debuggingFuncs.cpp debuggingFuncs.hpp genFuncs.hpp
domain.o: domain.cpp domain.hpp genFuncs.hpp
fault.o: fault.cpp fault.hpp genFuncs.hpp domain.hpp \
 rootFinderContext.hpp rootFinder.hpp
genFuncs.o: genFuncs.cpp genFuncs.hpp
helloWorld.o: helloWorld.cpp
iceSheet.o: iceSheet.cpp iceSheet.hpp integratorContext.hpp odeSolver.hpp \
 genFuncs.hpp domain.hpp maxwellViscoelastic.hpp linearElastic.hpp \
 sbpOps.hpp sbpOps_c.hpp debuggingFuncs.hpp spmat.hpp sbpOps_fc.hpp \
 fault.hpp rootFinderContext.hpp rootFinder.hpp
linearElastic.o: linearElastic.cpp linearElastic.hpp \
 integratorContext.hpp odeSolver.hpp genFuncs.hpp domain.hpp sbpOps.hpp \
 sbpOps_c.hpp debuggingFuncs.hpp spmat.hpp sbpOps_fc.hpp fault.hpp \
 rootFinderContext.hpp rootFinder.hpp
main.o: main.cpp genFuncs.hpp spmat.hpp domain.hpp sbpOps.hpp fault.hpp \
 rootFinderContext.hpp rootFinder.hpp linearElastic.hpp \
 integratorContext.hpp odeSolver.hpp sbpOps_c.hpp debuggingFuncs.hpp \
 sbpOps_fc.hpp maxwellViscoelastic.hpp powerLaw.hpp
mainEx.o: mainEx.cpp
mainLinearElastic.o: mainLinearElastic.cpp genFuncs.hpp spmat.hpp \
 domain.hpp sbpOps.hpp sbpOps_fc.hpp debuggingFuncs.hpp sbpOps_c.hpp \
 sbpOps_sc.hpp fault.hpp rootFinderContext.hpp rootFinder.hpp \
 linearElastic.hpp integratorContext.hpp odeSolver.hpp
mainMaxwell.o: mainMaxwell.cpp genFuncs.hpp spmat.hpp domain.hpp \
 sbpOps.hpp fault.hpp rootFinderContext.hpp rootFinder.hpp \
 linearElastic.hpp integratorContext.hpp odeSolver.hpp sbpOps_c.hpp \
 debuggingFuncs.hpp sbpOps_fc.hpp maxwellViscoelastic.hpp powerLaw.hpp
main_iceSheet.o: main_iceSheet.cpp genFuncs.hpp spmat.hpp domain.hpp \
 sbpOps.hpp fault.hpp rootFinderContext.hpp rootFinder.hpp \
 linearElastic.hpp integratorContext.hpp odeSolver.hpp sbpOps_c.hpp \
 debuggingFuncs.hpp sbpOps_fc.hpp maxwellViscoelastic.hpp iceSheet.hpp
maxwellViscoelastic.o: maxwellViscoelastic.cpp maxwellViscoelastic.hpp \
 integratorContext.hpp odeSolver.hpp genFuncs.hpp domain.hpp \
 linearElastic.hpp sbpOps.hpp sbpOps_c.hpp debuggingFuncs.hpp spmat.hpp \
 sbpOps_fc.hpp fault.hpp rootFinderContext.hpp rootFinder.hpp
odeSolver.o: odeSolver.cpp odeSolver.hpp integratorContext.hpp \
 genFuncs.hpp
powerLaw.o: powerLaw.cpp powerLaw.hpp integratorContext.hpp odeSolver.hpp \
 genFuncs.hpp domain.hpp linearElastic.hpp sbpOps.hpp sbpOps_c.hpp \
 debuggingFuncs.hpp spmat.hpp sbpOps_fc.hpp fault.hpp \
 rootFinderContext.hpp rootFinder.hpp
rootFinder.o: rootFinder.cpp rootFinder.hpp rootFinderContext.hpp
sbpOps_arrays.o: sbpOps_arrays.cpp sbpOps.hpp domain.hpp genFuncs.hpp
sbpOps_c.o: sbpOps_c.cpp sbpOps_c.hpp domain.hpp genFuncs.hpp \
 debuggingFuncs.hpp spmat.hpp sbpOps.hpp
sbpOps_fc.o: sbpOps_fc.cpp sbpOps_fc.hpp domain.hpp genFuncs.hpp \
 debuggingFuncs.hpp spmat.hpp sbpOps.hpp
sbpOps_sc.o: sbpOps_sc.cpp sbpOps_sc.hpp domain.hpp genFuncs.hpp \
 debuggingFuncs.hpp spmat.hpp sbpOps.hpp
sbpOps_temp.o: sbpOps_temp.cpp sbpOps.hpp domain.hpp genFuncs.hpp
spmat.o: spmat.cpp spmat.hpp
test.o: test.cpp sbpOps.hpp domain.hpp genFuncs.hpp sbpOps_c.hpp \
 debuggingFuncs.hpp spmat.hpp
testMain.o: testMain.cpp genFuncs.hpp domain.hpp spmat.hpp sbpOps.hpp \
 sbpOps_c.hpp debuggingFuncs.hpp sbpOps_fc.hpp testOdeSolver.hpp \
 integratorContext.hpp odeSolver.hpp
testOdeSolver.o: testOdeSolver.cpp testOdeSolver.hpp \
 integratorContext.hpp odeSolver.hpp genFuncs.hpp

