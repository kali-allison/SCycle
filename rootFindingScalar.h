#ifndef ROOTFINDINGSCALAR_H_INCLUDED
#define ROOTFINDINGSCALAR_H_INCLUDED

PetscErrorCode func(PetscScalar in, PetscScalar *out);

PetscErrorCode funcPrime(PetscScalar in, PetscScalar *out);

PetscErrorCode secantMethod();

PetscErrorCode bisect(PetscErrorCode (*pt2Func)(const PetscInt,const PetscScalar,PetscScalar *,void*),
    const PetscInt ind, PetscScalar left,PetscScalar right,PetscScalar *out,PetscScalar atol,
    PetscInt itMax,void *ctx);


#endif
