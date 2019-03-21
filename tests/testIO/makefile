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

output: test_io_initiateWriteAppend.o $(OBJECTS)
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm  test_io_initiateWriteAppend.o

depend:
	-g++ -MM *.c*

clean::
	-rm -f *.o output

# Dependencies
test_io_initiateWriteAppend.o: test_io_initiateWriteAppend.cpp genFuncs.hpp
genFuncs.o: genFuncs.hpp genFuncs.cpp
