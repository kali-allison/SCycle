all: output

DEBUG_MODULES = -DVERBOSE=1
CFLAGS        = $(DEBUG_MODULES)
CPPFLAGS      = $(DEBUG_MODULES) -std=c++11 -g -Wall -Werror
FFLAGS        = -I${PETSC_DIR}/include/finclude
CLINKER       = openmpicc

OBJECTS := genFuncs.o

PETSC_DIR = /home/yyy910805/petsc
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

output: testCheckpointLoadVal.o $(OBJECTS)
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm testCheckpointLoadVal.o

depend:
	-g++ -MM *.c*

clean::
	-rm -f *.o output

# Dependencies
testCheckpointLoadVal.o: testCheckpointLoadVal.cpp genFuncs.hpp
genFuncs.o: genFuncs.hpp genFuncs.cpp
