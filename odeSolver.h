#ifndef TIMESOLVER_H_INCLUDED
#define TIMESOLVER_H_INCLUDED

#define STRINGIFY(name) #name

class TimeSolver
{

  private:

    PetscReal     _initT,_finalT,_currT,_deltaT,_minDeltaT,_maxDeltaT;
    PetscReal     _atol,_reltol;
    PetscInt      _maxNumSteps,_stepCount;
    std::string   _solverType;
    Vec           *_var,*_dvar;
    int           _lenVar;
    void          *_userContext;
    PetscErrorCode (*_rhsFunc)(const PetscReal,const int,Vec*,Vec*,void*); // time, lenVar, var, dvar, ctx
    PetscErrorCode (*_timeMonitor)(const PetscReal,const PetscInt,const Vec*,const int,void*);

    PetscErrorCode odeFEULER();
    PetscErrorCode odeRK32();


  public:

    TimeSolver(PetscInt maxNumSteps,std::string solverType);
    TimeSolver(PetscScalar finalT,PetscInt maxNumSteps,std::string solverType);
    ~TimeSolver();

    PetscErrorCode setTimeRange(const PetscReal initT,const PetscReal finalT);
    PetscErrorCode setStepSize(const PetscReal deltaT);
    PetscErrorCode setTolerance(const PetscReal tol);
    PetscErrorCode setRhsFunc(PetscErrorCode (*rhsFunc)(const PetscReal,const int,Vec*,Vec*,void*));
    PetscErrorCode setTimeMonitor(PetscErrorCode (*timeMonitor)(const PetscReal,const PetscInt,const Vec*,const int,void*));
    PetscErrorCode setUserContext(void * userContext);
    PetscErrorCode setInitialConds(Vec *var, const int lenVar);
    PetscErrorCode setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT);

    PetscErrorCode viewSolver();
    PetscErrorCode debugMyCode(const PetscReal time,const PetscInt steps,const Vec *var,const char *str);

    PetscErrorCode runTimeSolver();

};

PetscErrorCode tempRhsFunc(const PetscReal time,const int lenVar,Vec *var,Vec *dvar,void *userContext);
PetscErrorCode tempTimeMonitor(const PetscReal time, const PetscInt stepCount,
                               const Vec *var, const int lenVar, void*userContext);

#endif
