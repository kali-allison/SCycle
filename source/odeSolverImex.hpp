#ifndef ODESOLVERIMEX_HPP_INCLUDED
#define ODESOLVERIMEX_HPP_INCLUDED

#include <petscts.h>
#include <string>
#include <cmath>
#include <vector>
#include <map>
#include <algorithm>
#include <assert.h>
#include "integratorContextImex.hpp"
#include "genFuncs.hpp"

using namespace std;

/*
 * Provides a set of algorithms to solve a system of ODEs of
 * the form y' = f(t,y) using IMEX time stepping,
 * where y is represented as an array of one or more Vecs
 * (PETSc data type).
 *
 * Containers for integration:
 *   var          map<string,Vec> of explicitly integrated variables
 *   varIm    map<string,Vec> of implicitly integrated variables
 *
 * SOLVER TYPE        ALGORITHM
 *  RK32_WBE        explicit part Runge-Kutta (2,3), implicit controlled by user
 *  RK43_WBE        explicit part Runge-Kutta (3,4), implicit controlled by user
 *
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
public:

  PetscReal               _initT,_finalT,_currT,_deltaT;
  PetscReal               _newDeltaT; // stores future deltaT for access by outside classes, primarily for checkpointing
  PetscInt                _maxNumSteps,_stepCount;
  map<string,Vec>         _varEx,_dvar; // explicit integration variable and rate
  map<string,Vec>         _varIm; // implicit integration variable, once per time step
  vector<string>          _errInds; // which inds of _var to use for error control
  vector<double>          _scale; // scale factor for entries in _errInds
  double                  _runTime;
  string                  _controlType;
  string                  _normType;

  PetscReal   _minDeltaT,_maxDeltaT;
  PetscReal   _totTol; // total tolerance, might be atol, or rtol, or a combination of both
  PetscInt    _numRejectedSteps,_numMinSteps,_numMaxSteps;

  // for PID error control
  double _errA[2];

  // constructor and destructor
  OdeSolverImex(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType);
  virtual ~OdeSolverImex() {};

  // member functions
  PetscErrorCode setInitialStepCount(const PetscReal stepCount);
  PetscErrorCode setToleranceType(const string normType); // type of norm used for error control

  // virtual member functions are declared in base class and redefined in derived class
  virtual PetscErrorCode setTimeRange(const PetscReal initT,const PetscReal finalT) = 0;
  virtual PetscErrorCode setStepSize(const PetscReal deltaT) = 0;
  virtual PetscErrorCode setTolerance(const PetscReal tol) = 0;
  virtual PetscErrorCode setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT) = 0;
  virtual PetscErrorCode setInitialConds(map<string,Vec>& varEx, map<string,Vec>& varIm) = 0;
  virtual PetscErrorCode setErrInds(vector<string>& errInds) = 0;
  virtual PetscErrorCode setErrInds(vector<string>& errInds, vector<double> scale) = 0;

  virtual PetscErrorCode view() = 0;
  virtual PetscErrorCode integrate(IntegratorContextImex *obj) = 0;
  virtual PetscReal computeStepSize(const PetscReal totErr) = 0;
  virtual PetscReal computeError() = 0;
  virtual PetscErrorCode writeCheckpoint(PetscViewer &viewer) = 0;
  virtual PetscErrorCode loadCheckpoint(const std::string inputDir) = 0;
};


// Explicit RK32 scheme from Hairer et al., with added Backward Euler implicit scheme once per time step
// derived class from OdeSolverImex
class RK32_WBE : public OdeSolverImex
{
public:

  // for P or PID error control
  PetscReal   _kappa,_ord; // safety factor in step size determinance, order of accuracy of method
  PetscReal   _totErr; // error between 3rd order solution and embedded 2nd order solution

  // intermediate values for time stepping for the explicit variable
  map<string,Vec> _k1,_f1,_k2,_f2,_y2,_y3;
  map<string,Vec> _vardTIm;

  // constructor and destructor
  RK32_WBE(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType);
  ~RK32_WBE();

  // member functions
  PetscErrorCode setTimeRange(const PetscReal initT,const PetscReal finalT);
  PetscErrorCode setStepSize(const PetscReal deltaT);
  PetscErrorCode setTolerance(const PetscReal tol);
  PetscErrorCode setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT);
  PetscErrorCode setInitialConds(map<string,Vec>& varEx, map<string,Vec>& varIm);
  PetscErrorCode setErrInds(vector<string>& errInds);
  PetscErrorCode setErrInds(vector<string>& errInds, vector<double> scale);
  PetscErrorCode view();
  PetscErrorCode integrate(IntegratorContextImex *obj);
  PetscReal computeStepSize(const PetscReal totErr);
  PetscReal computeError();
  PetscErrorCode writeCheckpoint(PetscViewer &viewer);
  PetscErrorCode loadCheckpoint(const std::string inputDir);
};


// Runge-Kutta 4(3) scheme for explicit time integration, with added
// Backward Euler implicit scheme once per time step.
// Based on "ARK4(3)6L[2]SA-ERK" algorithm from Kennedy and Carpenter (2003):
// "Additive Runge-Kutta schemes for convection-diffusion-reaction equations"
// Note: Has matching IMEX equivalent
// derived class from OdeSolverImex
class RK43_WBE : public OdeSolverImex
{
public:

  // for P or PID error control
  // safety factor in step size determinance, order of accuracy of method
  PetscReal   _kappa,_ord;
  // error between 3rd order solution and embedded 2nd order solution
  PetscReal   _totErr;

  // intermediate values for time stepping for the explicit variable
  map<string,Vec> _k1,_k2,_k3,_k4,_k5,_k6,_y4,_y3;
  map<string,Vec> _f1,_f2,_f3,_f4,_f5,_f6;

  // intermediate value for implict variable
  map<string,Vec> _vardTIm;

  // constructor and destructor
  RK43_WBE(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType);
  ~RK43_WBE();

  // member functions
  PetscErrorCode setTimeRange(const PetscReal initT,const PetscReal finalT);
  PetscErrorCode setStepSize(const PetscReal deltaT);
  PetscErrorCode setTolerance(const PetscReal tol);
  PetscErrorCode setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT);
  PetscErrorCode setInitialConds(map<string,Vec>& varEx, map<string,Vec>& varIm);
  PetscErrorCode setErrInds(vector<string>& errInds);
  PetscErrorCode setErrInds(vector<string>& errInds, vector<double> scale);
  PetscErrorCode view();
  PetscErrorCode integrate(IntegratorContextImex *obj);
  PetscReal computeStepSize(const PetscReal totErr);
  PetscReal computeError();
  PetscErrorCode writeCheckpoint(PetscViewer &viewer);
  PetscErrorCode loadCheckpoint(const std::string inputDir);
};

#endif

