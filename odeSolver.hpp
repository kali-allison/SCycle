#ifndef ODESOLVER_HPP_INCLUDED
#define ODESOLVER_HPP_INCLUDED

#include <petscts.h>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <assert.h>
#include "integratorContext.hpp"

using namespace std;

/*
 * Provides a set of algorithms to solve a system of ODEs of
 * the form y' = f(t,y).
 *
 * SOLVER TYPE      ALGORITHM
 *  FEuler        forward Euler
 *  RK32          explicit Runge-Kutta (2,3)
 *
 * To obtain solutions at user-specified times, use FEuler and call setStepSize
 * in the routine f(t,y).
 *
 * y is represented as an array of one or more Vecs (PetSc data type).
 *
 * At minimum, the user must specify:
 *     QUANTITY               FUNCTION
 *  max number of steps      constructor
 *  solver type              constructor
 *  initial conditions       setInitialConds     Note: this array will be modified during integration
 *  initial step size        constructor
 *  step size alg            constructor (this is only used by the adaptive time-stepping algorithm)
 *  f(t,y)                   object passed to integrate must have member function d_dt(PetscScalar, PetscScalar*,PetscScalar*)
 *  final time               constructor and setTimeRange
 *  timeMonitor              object passed to integrate must have member function timeMonitor
 *
 *
 * Optional fields that can also be specified:
 *     QUANTITY               FUNCTION
 *  tolerance                setTolerance
 *  maximum step size        setStepSize
 *  minimum step size        setTimeStepBounds
 *  initial step size        setTimeStepBounds
 *
 * Once the odeSolver context is set, call integrate() to perform
 * the integration.
 *
 * y(t=final time) is stored in the initial conditions array.  Summary output
 * information is provided by viewSolver.  Users can obtain information at
 * each time step within a user-defined monitor function.
 *
 */

class OdeSolver
{
  protected:

    PetscReal           _initT,_finalT,_currT,_deltaT;
    PetscInt            _maxNumSteps,_stepCount;
    std::vector<Vec>    _var,_dvar;
    int                 _lenVar;
    double              _runTime;
    string              _controlType;

  public:

    // iterators for _var and _dvar
    typedef typename vector<Vec>::iterator it_vec;
    typedef typename vector<Vec>::const_iterator const_it_vec;

    OdeSolver(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType);
    ~OdeSolver();

    PetscErrorCode setTimeRange(const PetscReal initT,const PetscReal finalT);
    PetscErrorCode setStepSize(const PetscReal deltaT);
    //~PetscErrorCode setRhsFunc(PetscErrorCode (*rhsFunc)(const PetscReal,const int,Vec*,Vec*,void*));
    //~PetscErrorCode setTimeMonitor(PetscErrorCode (*timeMonitor)(const PetscReal,const PetscInt,const Vec*,const int,void*));

    virtual PetscErrorCode setTolerance(const PetscReal atol) = 0;
    virtual PetscErrorCode setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT) = 0;
    virtual PetscErrorCode setInitialConds(std::vector<Vec>& var) = 0;
    virtual PetscErrorCode view() = 0;
    virtual PetscErrorCode integrate(IntegratorContext *obj) = 0;

    // from Effective STL
    struct DeleteVecObject // used in destructor
    {
      void operator()(Vec& ptr) const
      {
        VecDestroy(&ptr);
      }
    };


};

PetscErrorCode newtempRhsFunc(const PetscReal time,const int lenVar,Vec *var,Vec *dvar,void *userContext);
PetscErrorCode newtempTimeMonitor(const PetscReal time, const PetscInt stepCount,
                               const Vec *var, const int lenVar, void*userContext);


class FEuler : public OdeSolver
{
  public:
    FEuler(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType);
    PetscErrorCode view();

    PetscErrorCode setTolerance(const PetscReal atol){return 0;};
    PetscErrorCode setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT){ return 0;};
    PetscErrorCode setInitialConds(vector<Vec>& var);
    PetscErrorCode integrate(IntegratorContext *obj);
};

class RK32 : public OdeSolver
{
  private:

    PetscReal   _minDeltaT,_maxDeltaT;
    PetscReal   _atol; // absolute and relative tolerances
    PetscReal   _kappa,_ord; // safety factor in step size determinance
    PetscReal   _absErr[3]; // safety factor in step size determinance
    PetscInt    _numRejectedSteps,_numMinSteps,_numMaxSteps;

    //~Vec *_varHalfdT,*_dvarHalfdT,*_vardT,*_dvardT,*_var2nd,*_dvar2nd,*_var3rd;
    //~Vec *_errVec;
    std::vector<Vec> _varHalfdT,_dvarHalfdT,_vardT,_dvardT,_var2nd,_dvar2nd,_var3rd;
    std::vector<Vec> _errVec;

    PetscReal computeStepSize(const PetscReal totErr);
    PetscReal computeError();

  public:

    RK32(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType);
    ~RK32();

    PetscErrorCode setTolerance(const PetscReal atol);
    PetscErrorCode setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT);
    PetscErrorCode setInitialConds(vector<Vec>& var);
    PetscErrorCode view();

    PetscErrorCode integrate(IntegratorContext *obj);

};

#endif

