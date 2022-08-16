#ifndef ODESOLVER_HPP_INCLUDED
#define ODESOLVER_HPP_INCLUDED

#include <petscts.h>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <assert.h>
#include "integratorContextEx.hpp"
#include "genFuncs.hpp"

using namespace std;
/*
 * Provides a set of algorithms to solve a system of ODEs of
 * the form y' = f(t,y) using EXPLICIT time stepping.
 *
 * SOLVER TYPE      ALGORITHM
 *  FEuler        forward Euler
 *  RK32          explicit Runge-Kutta (2,3)
 *  RK43          explicit Runge-Kutta (3,4)
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
 * y(t = final_time) is stored in the initial conditions array. Summary output
 * information is provided by viewSolver. Users can obtain information at
 * each time step within a user-defined monitor function.
 *
 */

class OdeSolver
{
public:

  PetscReal          _initT,_finalT,_currT,_deltaT;
  PetscReal          _newDeltaT; // stores future deltaT primarily for checkpointing
  PetscInt           _maxNumSteps,_stepCount;
  map<string,Vec>    _var,_dvar; // integration variable and rate
  vector<string>     _errInds; // which keys of _var to use for error control
  vector<double>     _scale; // scale factor for entries in _errInds
  double             _runTime;
  string             _controlType;
  string             _normType;

  // for PID error control
  double _errA[2];
  map<string,Vec> _y2,_y3,_y4;

  OdeSolver(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType);
  virtual ~OdeSolver() {};

  PetscErrorCode setTimeRange(const PetscReal initT,const PetscReal finalT);
  PetscErrorCode setInitialStepCount(const PetscReal stepCount);
  PetscErrorCode setStepSize(const PetscReal deltaT);
  PetscErrorCode setToleranceType(const string normType); // type of norm used for error control

  virtual PetscErrorCode setTolerance(const PetscReal tol) = 0;
  virtual PetscErrorCode setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT) = 0;
  virtual PetscErrorCode setInitialConds(map<string,Vec>& var){return 1;};
  virtual PetscErrorCode setErrInds(vector<string>& errInds) = 0;
  virtual PetscErrorCode setErrInds(vector<string>& errInds, vector<double> scale) = 0;
  virtual PetscErrorCode view() = 0;
  virtual PetscErrorCode integrate(IntegratorContextEx *obj) = 0;
  virtual PetscErrorCode writeCheckpoint(PetscViewer &viewer) = 0;
  virtual PetscErrorCode loadCheckpoint(const std::string inputDir) = 0;
};


// FEuler is a derived class from OdeSolver
class FEuler : public OdeSolver
{
public:
  FEuler(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType);
  ~FEuler();
  PetscErrorCode view();

  PetscErrorCode setTolerance(const PetscReal tol){return 0;};
  PetscErrorCode setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT){ return 0;};
  PetscErrorCode setInitialConds(map<string,Vec>& var);
  PetscErrorCode setErrInds(vector<string>& errInds) {return 0;};
  PetscErrorCode setErrInds(vector<string>& errInds, vector<double> scale) {return 0;};
  PetscErrorCode integrate(IntegratorContextEx *obj);
  PetscErrorCode writeCheckpoint(PetscViewer &viewer);
  PetscErrorCode loadCheckpoint(const std::string inputDir);
};


// Runge-kutta time-stepping, 3rd-order
// Based on algorithm from Hairer et al.
class RK32 : public OdeSolver
{
public:

  PetscReal   _minDeltaT,_maxDeltaT;
  PetscReal   _totTol; // total tolerance, might be atol, or rtol, or a combination of both
  PetscReal   _kappa,_ord; // safety factor in step size determinance, order of accuracy of method
  PetscInt    _numRejectedSteps,_numMinSteps,_numMaxSteps;

  PetscReal   _totErr;

  map<string,Vec> _k1,_f1,_k2,_f2,_y2,_y3;

  PetscReal computeStepSize(const PetscReal totErr);
  PetscReal computeError();

  // constructor and destructor
  RK32(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType);
  ~RK32();

  // member functions of this class
  PetscErrorCode setTolerance(const PetscReal tol);
  PetscErrorCode setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT);
  PetscErrorCode setInitialConds(map<string,Vec>& var);
  PetscErrorCode setErrInds(vector<string>& errInds);
  PetscErrorCode setErrInds(vector<string>& errInds, vector<double> scale);
  PetscErrorCode view();
  PetscErrorCode integrate(IntegratorContextEx *obj);
  PetscErrorCode writeCheckpoint(PetscViewer &viewer);
  PetscErrorCode loadCheckpoint(const std::string inputDir);
};


// Based on "ARK4(3)6L[2]SA-ERK" algorithm from Kennedy and Carpenter (2003):
// "Additive Runge-Kutta schemes for convection-diffusion-reaction equations"
// Runge-Kutta time-stepping, 4th order
// Note: Has matching IMEX equivalent
class RK43 : public OdeSolver
{
public:

  PetscReal   _minDeltaT,_maxDeltaT;
  PetscReal   _atol,_rtol;
  PetscReal   _totTol;
  PetscReal   _kappa,_ord;
  PetscInt    _numRejectedSteps,_numMinSteps,_numMaxSteps;
  PetscReal   _totErr;

  map<string,Vec> _k1,_k2,_k3,_k4,_k5,_k6,_y3,_y4;
  map<string,Vec> _f1,_f2,_f3,_f4,_f5,_f6;

  PetscReal computeStepSize(const PetscReal totErr);
  PetscReal computeError();

  // constructor and destructor
  RK43(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType);
  ~RK43();

  // various member functions
  PetscErrorCode setTolerance(const PetscReal tol);
  PetscErrorCode setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT);
  PetscErrorCode setInitialConds(map<string,Vec>& var);
  PetscErrorCode setErrInds(vector<string>& errInds);
  PetscErrorCode setErrInds(vector<string>& errInds, vector<double> scale);
  PetscErrorCode view();
  PetscErrorCode integrate(IntegratorContextEx *obj);
  PetscErrorCode writeCheckpoint(PetscViewer &viewer);
  PetscErrorCode loadCheckpoint(const std::string inputDir);
};

#endif
