all: main

DEBUG_MODULES   = -DVERBOSE=1 -DODEPRINT=1 -DDEBUG=0
CFLAGS          = $(DEBUG_MODULES)
FFLAGS	        = -I${PETSC_DIR}/include/finclude
CLINKER		      = openmpicc

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

main:  main.o genFuncs.o debuggingFuncs.o odeSolver.o sbpOps.o lithosphere.o fault.o\
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

genFuncs.o: genFuncs.cpp genFuncs.hpp
domain.o: domain.cpp domain.hpp
sbpOps.o: sbpOps.cpp sbpOps.hpp genFuncs.hpp domain.hpp debuggingFuncs.hpp spmat.hpp
fault.o: fault.cpp fault.hpp genFuncs.hpp domain.hpp rootFinderContext.hpp rootFinder.hpp
lithosphere.o: lithosphere.cpp lithosphere.hpp genFuncs.hpp domain.hpp sbpOps.hpp \
 debuggingFuncs.hpp fault.hpp integratorContext.hpp
asthenosphere.o: asthenosphere.cpp asthenosphere.hpp genFuncs.hpp domain.hpp lithosphere.hpp
main.o: main.cpp lithosphere.hpp domain.hpp spmat.hpp sbpOps.hpp
debuggingFuncs.o: debuggingFuncs.cpp debuggingFuncs.hpp genFuncs.hpp
odeSolver.o: odeSolver.cpp odeSolver.hpp genFuncs.hpp integratorContext.hpp
rootFinder.o: rootFinder.cpp rootFinder.hpp rootFinderContext.hpp
spmat.o: spmat.cpp spmat.hpp

