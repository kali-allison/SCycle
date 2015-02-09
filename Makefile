all: main

DEBUG_MODULES   = -DVERBOSE=1 -DDEBUG=0
CFLAGS          = $(DEBUG_MODULES)
FFLAGS	        = -I${PETSC_DIR}/include/finclude
CLINKER		      = openmpicc

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

main:  main.o debuggingFuncs.o odeSolver.o sbpOps.o lithosphere.o fault.o\
 domain.o rootFinder.o debuggingFuncs.o spmat.o asthenosphere.o
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm main.o

helloWorld: helloWorld.o
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm helloWorld.o


#.PHONY : clean
clean::
	-rm -f *.o main helloWorld

depend:
	-g++ -MM *.c*

include ${PETSC_DIR}/conf/test

#=========================================================
# Dependencies
#=========================================================


domain.o: domain.cpp domain.hpp
sbpOps.o: sbpOps.cpp sbpOps.hpp domain.hpp debuggingFuncs.hpp spmat.hpp
fault.o: fault.cpp fault.hpp domain.hpp rootFinderContext.hpp rootFinder.hpp
lithosphere.o: lithosphere.cpp lithosphere.hpp domain.hpp sbpOps.hpp \
 debuggingFuncs.hpp fault.hpp integratorContext.hpp
asthenosphere.o: asthenosphere.cpp asthenosphere.hpp domain.hpp lithosphere.hpp
main.o: main.cpp lithosphere.hpp domain.hpp spmat.hpp sbpOps.hpp
debuggingFuncs.o: debuggingFuncs.cpp debuggingFuncs.hpp
odeSolver.o: odeSolver.cpp odeSolver.hpp integratorContext.hpp
rootFinder.o: rootFinder.cpp rootFinder.hpp rootFinderContext.hpp
spmat.o: spmat.cpp spmat.hpp

