#ifndef ODESOLVERIMEX_HPP_INCLUDED
#define ODESOLVERIMEX_HPP_INCLUDED

#include <petscts.h>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <assert.h>
#include "integratorContextImex.hpp"
#include "genFuncs.hpp"

/*
 * Provides a set of algorithms to solve a system of ODEs of
 * the form y' = f(t,y) using IMEX time stepping.
 *
 * SOLVER TYPE      ALGORITHM
 *  FEuler        forward Euler
 *  RK32          explicit Runge-Kutta (2,3)
 *
 * To obtain solutions at user-specified times, use FEuler and call setStepSize
 * in the routine f(t,y).
 *
 * y is represented as an array of one or more Vecs (PETSc data type).
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

class OdeSolverImex
{
  protected:

    PetscReal           _initT,_finalT,_currT,_deltaT;
    PetscInt            _maxNumSteps,_stepCount;
    std::vector<Vec>    _var,_dvar; // explicit integration variable and rate
    std::vector<Vec>    _varIm; // implicit integration variable
    std::vector<int>    _errInds; // which inds of _var to use for error control
    int                 _lenVar;
    double              _runTime;
    string              _controlType;

  public:

    PetscReal   _minDeltaT,_maxDeltaT;
    PetscReal   _atol; // absolute and relative tolerances
    PetscReal   _kappa,_ord; // safety factor in step size determinance
    PetscReal   _absErr[3]; // safety factor in step size determinance
    PetscInt    _numRejectedSteps,_numMinSteps,_numMaxSteps;

    // intermediate values for time stepping for the explicit variable
    std::vector<Vec> _varHalfdT,_dvarHalfdT,_vardT,_dvardT,_var2nd,_dvar2nd,_var3rd;
    std::vector<Vec> _varHalfdTIm,_vardTIm;


    OdeSolverImex(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType);
    ~OdeSolverImex();

    PetscErrorCode setTimeRange(const PetscReal initT,const PetscReal finalT);
    PetscErrorCode setStepSize(const PetscReal deltaT);

    PetscErrorCode setTolerance(const PetscReal atol);
    PetscErrorCode setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT);
    PetscErrorCode setInitialConds(std::vector<Vec>& varEx,std::vector<Vec>& varIm);
    PetscErrorCode setErrInds(std::vector<int>& errInds);
    PetscErrorCode view();
    PetscErrorCode integrate(IntegratorContextImex *obj);


    PetscReal computeStepSize(const PetscReal totErr);
    PetscReal computeError();
};

#endif

