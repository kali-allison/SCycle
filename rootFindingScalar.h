#ifndef ROOTFINDINGSCALAR_H_INCLUDED
#define ROOTFINDINGSCALAR_H_INCLUDED

#include <petscts.h>
#include <assert.h>


PetscErrorCode exFunc(const PetscInt ind, PetscScalar in,PetscScalar *out, void * ctx);

PetscErrorCode funcPrime(PetscScalar in, PetscScalar *out);

PetscErrorCode secantMethod(PetscErrorCode func(const PetscInt,const PetscScalar,PetscScalar *,void*),
    const PetscInt ind, PetscScalar left,PetscScalar right,PetscScalar *out,PetscInt *its,PetscScalar atol,
    PetscInt itMax,void *ctx);

PetscErrorCode bisect(PetscErrorCode (*func)(const PetscInt,const PetscScalar,PetscScalar *,void*),
    const PetscInt ind, PetscScalar left,PetscScalar right,PetscScalar *out,PetscInt *its,PetscScalar atol,
    PetscInt itMax,void *ctx);

/*
 * Safe-guarded secant method: uses a combination of secant and bisection
 * root finding algorithms.
 */
PetscErrorCode safeSecant(PetscErrorCode (*func)(const PetscInt,const PetscScalar,PetscScalar *,void*),
    const PetscInt ind, PetscScalar left,PetscScalar right,PetscScalar *out,PetscInt *its,PetscScalar atol,
    PetscInt itMax,void *ctx);


#endif
