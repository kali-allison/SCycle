all: output

DEBUG_MODULES = -DVERBOSE=2
CFLAGS        = $(DEBUG_MODULES)
CPPFLAGS      = $(DEBUG_MODULES) -std=c++11 -g -Wall -O0
FFLAGS        = -I${PETSC_DIR}/include/finclude
CLINKER       = openmpicc

OBJECTS := domain.o

PETSC_DIR = /home/yyy910805/petsc
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

output: test_domain.o $(OBJECTS)
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm test_domain.o

depend:
	-g++ -MM *.c*

clean::
	-rm -f *.o output

# Dependencies
test_domain.o: test_domain.cpp domain.hpp
domain.o: domain.cpp domain.hpp
