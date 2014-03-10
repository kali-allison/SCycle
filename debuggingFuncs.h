#ifndef DEBUGGINGFUNCS_H_INCLUDED
#define DEBUGGINGFUNCS_H_INCLUDED

PetscErrorCode checkMatrix(Mat * mat, const char fileLoc[],const char name[], UserContext * D);

PetscErrorCode checkVector(Vec * vec, const char fileLoc[],const char name[], UserContext * D);

/*
 * Same as MatEqual, except returns true if one matrix has 0 where the other
 * has no entry.
 */
PetscErrorCode MatEqualVals(Mat * A, Mat * B, PetscBool * bool,UserContext * D);

PetscErrorCode printMyArray(PetscScalar *myArray, PetscInt N);



#endif
