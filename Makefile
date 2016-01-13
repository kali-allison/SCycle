all: main

DEBUG_MODULES   = -DVERBOSE=1 -DODEPRINT=0 -DDEBUG=0 -DVERSION=${PETSC_VERSION_NUM}
CFLAGS          = $(DEBUG_MODULES)
FFLAGS	        = -I${PETSC_DIR}/include/finclude
CLINKER		      = openmpicc

OBJECTS := domain.o debuggingFuncs.o fault.o genFuncs.o linearElastic.o\
 maxwellViscoelastic.o odeSolver.o rootFinder.o sbpOps_c.o sbpOps_fc.o sbpOps.o\
 spmat.o testOdeSolver.o

ifeq (${PETSC_VERSION_NUM},4)
	include ${PETSC_DIR}/conf/variables
	include ${PETSC_DIR}/conf/rules
else
	include ${PETSC_DIR}/real/lib/petsc/conf/variables
	include ${PETSC_DIR}/real/lib/petsc/conf/rules
endif


trial:
	@echo ${PETSC_VERSION_NUM}

main:  main.o $(OBJECTS)
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm main.o

main_linearElastic:  main_linearElastic.o $(OBJECTS)
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm main_linearElastic.o

main_iceSheet:  main_iceSheet.o $(OBJECTS)
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm main_iceSheet.o

FDP: FDP.o
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}

testMain: testMain.o $(OBJECTS)
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm testMain.o

helloWorld: helloWorld.o
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm helloWorld.o

#.PHONY : clean
clean::
	-rm -f *.o main helloWorld main_linearElastic main_iceSheet

depend:
	-g++ -MM *.c*

ifeq (${PETSC_VERSION_NUM},4)
	include ${PETSC_DIR}/conf/test
else
	include ${PETSC_DIR}/real/lib/petsc/conf/test
endif
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
 sbpOps.hpp debuggingFuncs.hpp spmat.hpp sbpOps_c.hpp sbpOps_fc.hpp \
 fault.hpp rootFinderContext.hpp rootFinder.hpp
linearElastic.o: linearElastic.cpp linearElastic.hpp \
 integratorContext.hpp odeSolver.hpp genFuncs.hpp domain.hpp sbpOps.hpp \
 debuggingFuncs.hpp spmat.hpp sbpOps_c.hpp sbpOps_fc.hpp fault.hpp \
 rootFinderContext.hpp rootFinder.hpp
main.o: main.cpp genFuncs.hpp spmat.hpp domain.hpp sbpOps.hpp \
 debuggingFuncs.hpp sbpOps_c.hpp sbpOps_fc.hpp fault.hpp \
 rootFinderContext.hpp rootFinder.hpp linearElastic.hpp \
 integratorContext.hpp odeSolver.hpp maxwellViscoelastic.hpp
main_iceSheet.o: main_iceSheet.cpp genFuncs.hpp spmat.hpp domain.hpp \
 sbpOps.hpp debuggingFuncs.hpp sbpOps_c.hpp sbpOps_fc.hpp fault.hpp \
 rootFinderContext.hpp rootFinder.hpp linearElastic.hpp \
 integratorContext.hpp odeSolver.hpp maxwellViscoelastic.hpp iceSheet.hpp
main_linearElastic.o: main_linearElastic.cpp genFuncs.hpp spmat.hpp \
 domain.hpp sbpOps_fc.hpp debuggingFuncs.hpp sbpOps_c.hpp fault.hpp \
 rootFinderContext.hpp rootFinder.hpp linearElastic.hpp \
 integratorContext.hpp odeSolver.hpp sbpOps.hpp
maxwellViscoelastic.o: maxwellViscoelastic.cpp maxwellViscoelastic.hpp \
 integratorContext.hpp odeSolver.hpp genFuncs.hpp domain.hpp \
 linearElastic.hpp sbpOps.hpp debuggingFuncs.hpp spmat.hpp sbpOps_c.hpp \
 sbpOps_fc.hpp fault.hpp rootFinderContext.hpp rootFinder.hpp
odeSolver.o: odeSolver.cpp odeSolver.hpp integratorContext.hpp \
 genFuncs.hpp
powerLaw.o: powerLaw.cpp powerLaw.hpp integratorContext.hpp odeSolver.hpp \
 genFuncs.hpp domain.hpp linearElastic.hpp sbpOps.hpp debuggingFuncs.hpp \
 spmat.hpp sbpOps_c.hpp sbpOps_fc.hpp fault.hpp rootFinderContext.hpp \
 rootFinder.hpp
rootFinder.o: rootFinder.cpp rootFinder.hpp rootFinderContext.hpp
sbpOps.o: sbpOps.cpp sbpOps.hpp domain.hpp genFuncs.hpp \
 debuggingFuncs.hpp spmat.hpp sbpOps_c.hpp sbpOps_fc.hpp
sbpOps_arrays.o: sbpOps_arrays.cpp sbpOps.hpp domain.hpp genFuncs.hpp \
 debuggingFuncs.hpp spmat.hpp sbpOps_c.hpp sbpOps_fc.hpp
sbpOps_c.o: sbpOps_c.cpp sbpOps_c.hpp domain.hpp genFuncs.hpp \
 debuggingFuncs.hpp spmat.hpp
sbpOps_fc.o: sbpOps_fc.cpp sbpOps_fc.hpp domain.hpp genFuncs.hpp \
 debuggingFuncs.hpp spmat.hpp
spmat.o: spmat.cpp spmat.hpp
testMain.o: testMain.cpp genFuncs.hpp domain.hpp sbpOps.hpp \
 debuggingFuncs.hpp spmat.hpp sbpOps_c.hpp sbpOps_fc.hpp \
 testOdeSolver.hpp integratorContext.hpp odeSolver.hpp
testOdeSolver.o: testOdeSolver.cpp testOdeSolver.hpp \
 integratorContext.hpp odeSolver.hpp genFuncs.hpp






