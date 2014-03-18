#ifndef LINEARSYSFUNCS_H_INCLUDED
#define LINEARSYSFUNCS_H_INCLUDED

PetscErrorCode setLinearSystem(UserContext &D, const PetscBool loadMat);

PetscErrorCode ComputeRHS(UserContext &D);

PetscErrorCode loadOperators(UserContext &D);

PetscErrorCode createOperators(UserContext &D);

PetscErrorCode SBPoperators(PetscInt ORDER, PetscInt N, Mat *PinvMat, Mat *D, Mat *D2, Mat *S);

#endif
