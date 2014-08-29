all: main

DEBUG_MODULES   = -DVERBOSE=2 -DDEBUG=0
CFLAGS          = $(DEBUG_MODULES)
FFLAGS	        = -I${PETSC_DIR}/include/finclude
CLINKER		      = openmpicc

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

main:  main.o debuggingFuncs.o odeSolver.o sbpOps.o lithosphere.o fault.o domain.o rootFinder.o debuggingFuncs.o userContext.o
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}

ex12: ex12.o
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}

#.PHONY : clean
clean::
	-rm -f *.o main trial

depend:
	-g++ -MM *.c*

include ${PETSC_DIR}/conf/test

#=========================================================
# Dependencies
#=========================================================

rootFinder.o: rootFinder.cpp rootFinder.hpp fault.hpp
fault.o: fault.cpp fault.hpp domain.hpp rootFinder.hpp

debuggingFuncs.o: debuggingFuncs.cpp debuggingFuncs.hpp userContext.h
lithosphere.o: lithosphere.cpp lithosphere.hpp domain.hpp sbpOps.hpp \
 debuggingFuncs.hpp fault.hpp
main.o: main.cpp lithosphere.hpp domain.hpp
odeSolver.o: odeSolver.cpp odeSolver.hpp
rootFindingScalar.o: rootFindingScalar.cpp rootFindingScalar.h
sbpOps.o: sbpOps.cpp sbpOps.hpp domain.hpp debuggingFuncs.hpp
userContext.o: userContext.cpp userContext.h
domain.o: domain.cpp domain.hpp

