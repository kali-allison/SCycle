#ifndef DEBUGGINGFUNCS_H_INCLUDED
#define DEBUGGINGFUNCS_H_INCLUDED

#include <petscts.h>
#include <cmath>
#include <string>
//~#include "userContext.h"

using namespace std;

//~PetscErrorCode checkMatrix(Mat *mat,std::string fileLoc,std::string name, UserContext *D);
PetscErrorCode checkMatrix(Mat *mat,std::string fileLoc,std::string name);

/*
 * Same as MatEqual, except returns true if one matrix has 0 where the other
 * has no entry.
 */
//~PetscErrorCode MatEqualVals(Mat *A, Mat *B, PetscBool *flg, UserContext *D);
PetscErrorCode MatEqualVals(Mat *A, Mat *B, PetscBool *flg);


PetscErrorCode printMyArray(PetscScalar *myArray, PetscInt N);

// test function
PetscErrorCode testDebugFuncs();

#endif
