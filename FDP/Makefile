all: FDP

DEBUG_MODULES   = -DDEBUG=1
FFLAGS	        = -I${PETSC_DIR}/include/finclude
MANSEC          = Sys
CLINKER		    = openmpicc

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

FDP: FDP.o
	${CLINKER} $(DEBUG_MODULES) $^ -o $@ ${PETSC_SYS_LIB}

FDNP: FDNP.o
	${CLINKER} $(DEBUG_MODULES) $^ -o $@ ${PETSC_SYS_LIB}

.PHONY : clean
clean::
	-rm -f *.o $(objects) FDP FDNP
depend:
	g++ -MM *.c*

include ${PETSC_DIR}/conf/test

