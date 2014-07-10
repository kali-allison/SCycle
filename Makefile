all: main

DEBUG_MODULES   = -DVERBOSE=1 -DDEBUG=1 -DPERFORM_MMS=0
CFLAGS          = $(DEBUG_MODULES)
FFLAGS	        = -I${PETSC_DIR}/include/finclude
CLINKER		      = openmpicc

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

main:  main.o userContext.o init.o debuggingFuncs.o rateAndState.o rootFindingScalar.o linearSysFuncs.o rootFindingScalar.o timeStepping.o odeSolver.o
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}

ex12: ex12.o
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}

#.PHONY : clean
clean::
	-rm -f *.o main
	-rm -f data/*

depend:
	-g++ -MM *.c*

include ${PETSC_DIR}/conf/test

#=========================================================
# Dependencies
#=========================================================
main.o: main.cpp userContext.h init.hpp debuggingFuncs.hpp rateAndState.h rootFindingScalar.h linearSysFuncs.h odeSolver.h
userContext.o: userContext.cpp userContext.h
rateAndState.o: rateAndState.cpp userContext.h rateAndState.h
linearSysFuncs.o: linearSysFuncs.cpp userContext.h linearSysFuncs.h
rootFindingScalar.o: rootFindingScalar.cpp rootFindingScalar.h
timeStepping.o: timeStepping.cpp userContext.h linearSysFuncs.h rootFindingScalar.h timeStepping.h
init.o: init.cpp userContext.h init.hpp
debuggingFuncs.o: debuggingFuncs.cpp userContext.h debuggingFuncs.hpp
odeSolver.o: odeSolver.cpp odeSolver.h userContext.h
