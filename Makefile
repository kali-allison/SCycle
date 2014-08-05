all: main

DEBUG_MODULES   = -DVERBOSE=2 -DDEBUG=0 -DPERFORM_MMS=0
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

depend:
	-g++ -MM *.c*

include ${PETSC_DIR}/conf/test

#=========================================================
# Dependencies
#=========================================================
main.o: main.cpp odeSolver.h  userContext.h init.hpp debuggingFuncs.hpp rateAndState.h rootFindingScalar.h linearSysFuncs.h
userContext.o: userContext.cpp odeSolver.h userContext.h
rateAndState.o: rateAndState.cpp odeSolver.h userContext.h rateAndState.h
linearSysFuncs.o: linearSysFuncs.cpp odeSolver.h userContext.h linearSysFuncs.h
rootFindingScalar.o: rootFindingScalar.cpp rootFindingScalar.h
timeStepping.o: timeStepping.cpp odeSolver.h  userContext.h linearSysFuncs.h rootFindingScalar.h timeStepping.h
init.o: init.cpp odeSolver.h  userContext.h init.hpp
debuggingFuncs.o: debuggingFuncs.cpp odeSolver.h  userContext.h debuggingFuncs.hpp
odeSolver.o: odeSolver.cpp odeSolver.h
