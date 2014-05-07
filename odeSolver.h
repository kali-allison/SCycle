#ifndef ODESOLVER_H_INCLUDED
#define ODESOLVER_H_INCLUDED

#define STRINGIFY(name) #name

/*
 * Provides a set of algorithms to solve a system of ODEs of
 * the form y' = f(t,y).
 *
 * SOLVER TYPE      ALGORITHM
 *  "FEULER"      forward Euler
 *  "MANUAL"      time determined by user-specified file
 *  "RK32"        explicit Runge-Kutta (2,3)
 *
 * To obtain solutions at user-specified times, use FEULER and call setStepSize
 * in the routine f(t,y).
 *
 * y is represented as an array of one or more Vecs (PetSc data type).
 *
 * At minimum, the user must specify:
 *     QUANTITY               FUNCTION
 *  max number of steps      constructor
 *  solver type              constructor
 *  initial conditions       setInitialConds     Note: this array will be modified during integration
 *  f(t,y)                   setRhsFunc
 *  final time               constructor or setTimeRange
 *
 * Optional fields that can also be specified:
 *     QUANTITY               FUNCTION
 *  tolerance                setTolerance
 *  maximum step size        setStepSize
 *  minimum step size        setTimeStepBounds
 *  initial step size        setTimeStepBounds
 *
 * Once the odeSolver context is set, call runTimeSolver() to perform
 * the integration.
 *
 * y(t=final time) is stored in the initial conditions array.  Summary output
 * information is provided by viewSolver.  Users can obtain information at
 * each time step by specifying a user-defined monitor function with setTimeMonitor.
 *
 */


class OdeSolver
{

  private:

    PetscReal     _initT,_finalT,_currT,_deltaT,_minDeltaT,_maxDeltaT;
    PetscReal     _atol,_reltol;
    PetscInt      _maxNumSteps,_stepCount,_numRejectedSteps,_numMinSteps,_numMaxSteps;
    std::string   _solverType,_sourceFile;
    Vec           *_var,*_dvar;
    int           _lenVar;
    void          *_userContext;
    PetscErrorCode (*_rhsFunc)(const PetscReal,const int,Vec*,Vec*,void*); // time, lenVar, var, dvar, ctx
    PetscErrorCode (*_timeMonitor)(const PetscReal,const PetscInt,const Vec*,const int,void*);

    PetscErrorCode odeFEULER();
    PetscErrorCode odeMANUAL();
    PetscErrorCode odeRK32();


  public:

    OdeSolver(PetscInt maxNumSteps,std::string solverType);
    OdeSolver(PetscScalar finalT,PetscInt maxNumSteps,std::string solverType);
    ~OdeSolver();

    PetscErrorCode setTimeRange(const PetscReal initT,const PetscReal finalT);
    PetscErrorCode setStepSize(const PetscReal deltaT);
    PetscErrorCode setTolerance(const PetscReal tol);
    PetscErrorCode setRhsFunc(PetscErrorCode (*rhsFunc)(const PetscReal,const int,Vec*,Vec*,void*));
    PetscErrorCode setTimeMonitor(PetscErrorCode (*timeMonitor)(const PetscReal,const PetscInt,const Vec*,const int,void*));
    PetscErrorCode setUserContext(void * userContext);
    PetscErrorCode setInitialConds(Vec *var, const int lenVar);
    PetscErrorCode setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT);
    PetscErrorCode setSourceFile(const std::string sourceFile);

    PetscErrorCode viewSolver();
    PetscErrorCode debug(const PetscReal time,const PetscInt steps,const Vec *var,const Vec *dvar,const char *str);

    PetscErrorCode runOdeSolver();

};

PetscErrorCode tempRhsFunc(const PetscReal time,const int lenVar,Vec *var,Vec *dvar,void *userContext);
PetscErrorCode tempTimeMonitor(const PetscReal time, const PetscInt stepCount,
                               const Vec *var, const int lenVar, void*userContext);

#endif
