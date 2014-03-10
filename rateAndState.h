#ifndef RATEANDSTATE_H_INCLUDED
#define RATEANDSTATE_H_INCLUDED

PetscErrorCode rateAndStateFrictionScalar(PetscInt ind,PetscScalar vel, PetscScalar *out, void * ctx);

PetscErrorCode agingLaw(const PetscInt ind, PetscScalar *dPsi, void *ctx);

PetscErrorCode setRateAndState(UserContext *D);

PetscErrorCode writeRateAndState(UserContext *D);

#endif
