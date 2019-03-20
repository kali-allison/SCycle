all: main

DEBUG_MODULES   = -DVERBOSE=1
CFLAGS          = $(DEBUG_MODULES)
CPPFLAGS        = $(CFLAGS) -std=c++11 -g -O0
FFLAGS	        = -I${PETSC_DIR}/include/finclude
CLINKER		      = openmpicc

OBJECTS := genFuncs.o spmat.o sbpOps_m_constGrid.o sbpOps_m_varGrid.o

PETSC_DIR=/home/yyy910805/petsc

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

main:  main.o $(OBJECTS)
	-${CLINKER} $^ -o $@ ${PETSC_SYS_LIB}
	-rm main.o

#.PHONY : clean
clean::
	-rm -f *.o main

depend:
	-g++ -MM *.c*

include ${PETSC_DIR}/lib/petsc/conf/test

#=========================================================
# Dependencies
#=========================================================
genFuncs.o: genFuncs.cpp genFuncs.hpp
main.o: main.cpp genFuncs.hpp sbpOps.hpp sbpOps_m_constGrid.hpp sbpOps_m_varGrid.hpp \
 spmat.hpp
sbpOps_m_constGrid.o: sbpOps_m_constGrid.cpp sbpOps_m_constGrid.hpp spmat.hpp sbpOps.hpp genFuncs.hpp
sbpOps_m_varGrid.o: sbpOps_m_varGrid.cpp sbpOps_m_varGrid.hpp \
 spmat.hpp sbpOps.hpp genFuncs.hpp
spmat.o: spmat.cpp spmat.hpp
