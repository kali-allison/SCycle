#ifndef TIMESTEPPING_H_INCLUDED
#define TIMESTEPPING_H_INCLUDED

PetscErrorCode setInitialTimeStep(UserContext& D);

PetscErrorCode resumeCurrentTimeStep(UserContext& D);

PetscErrorCode computeTau(UserContext& D);

PetscErrorCode initSlipVel(UserContext& D);

PetscErrorCode computeSlipVel(UserContext& D);

PetscErrorCode timeStepRHSFuncPETSC(TS ts, PetscReal t, Vec in, Vec out, void* ctx);

PetscErrorCode timeMonitorPETSC(TS ts,PetscInt step,PetscReal time,Vec w,void* ctx);

PetscErrorCode rhsFunc(const PetscReal time,const int lenVar,Vec* var,Vec* dvar,void* userContext);

PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
                               const Vec* var,const int lenVar,void* userContext);

//~PetscErrorCode writeData(const char outFileRoot[],const char name[],PetscInt step,Vec w);

#endif
