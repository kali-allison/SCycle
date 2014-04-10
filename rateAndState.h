#ifndef RATEANDSTATE_H_INCLUDED
#define RATEANDSTATE_H_INCLUDED

PetscErrorCode stressMstrength(const PetscInt ind,const PetscScalar vel,PetscScalar *out, void * ctx);

PetscErrorCode agingLaw(const PetscInt ind,const PetscScalar psi, PetscScalar *dPsi,void *ctx);

PetscErrorCode setRateAndState(UserContext &D);

PetscErrorCode writeRateAndState(UserContext &D);

#endif
