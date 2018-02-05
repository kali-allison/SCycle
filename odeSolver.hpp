#ifndef ODESOLVER_HPP_INCLUDED
#define ODESOLVER_HPP_INCLUDED

#include <petscts.h>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <boost/circular_buffer.hpp>
#include "integratorContextEx.hpp"
#include "genFuncs.hpp"

/*
 * Provides a set of algorithms to solve a system of ODEs of
 * the form y' = f(t,y) using EXPLICIT time stepping.
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

class OdeSolver
{
  public:

    PetscReal               _initT,_finalT,_currT,_deltaT;
    PetscInt                _maxNumSteps,_stepCount;
    std::map<string,Vec>    _var,_dvar; // integration variable and rate
    std::vector<string>     _errInds; // which keys of _var to use for error control
    int                     _lenVar;
    double                  _runTime;
    string                  _controlType;

  //~ public:

    OdeSolver(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType);
    virtual ~OdeSolver() {};

    PetscErrorCode setTimeRange(const PetscReal initT,const PetscReal finalT);
    PetscErrorCode setStepSize(const PetscReal deltaT);

    virtual PetscErrorCode setTolerance(const PetscReal atol) = 0;
    virtual PetscErrorCode setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT) = 0;
    virtual PetscErrorCode setInitialConds(std::map<string,Vec>& var){return 1;};
    virtual PetscErrorCode setInitialCondsIm(std::map<string,Vec>& varIm) = 0;
    virtual PetscErrorCode setErrInds(std::vector<string>& errInds) = 0;
    virtual PetscErrorCode view() = 0;
    virtual PetscErrorCode integrate(IntegratorContextEx *obj){return 1;};
};

PetscErrorCode newtempRhsFunc(const PetscReal time,const int lenVar,Vec *var,Vec *dvar,void *userContext);
PetscErrorCode newtempTimeMonitor(const PetscReal time, const PetscInt stepCount,
                               const Vec *var, const int lenVar, void*userContext);


class FEuler : public OdeSolver
{
  public:
    FEuler(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType);
    ~FEuler() {};
    PetscErrorCode view();

    PetscErrorCode setTolerance(const PetscReal atol){return 0;};
    PetscErrorCode setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT){ return 0;};
    PetscErrorCode setInitialConds(std::map<string,Vec>& var);
    PetscErrorCode setInitialCondsIm(std::map<string,Vec>& varIm) {return 0;};
    PetscErrorCode setErrInds(std::vector<string>& errInds) {return 0;};
    PetscErrorCode integrate(IntegratorContextEx *obj);
};


// Based on algorithm from Hairer et al.
class RK32 : public OdeSolver
{
  public:

    PetscReal   _minDeltaT,_maxDeltaT;
    PetscReal   _atol; // absolute and relative tolerances
    PetscReal   _kappa,_ord; // safety factor in step size determinance, order of accuracy of method
    PetscReal   _absErr[3];
    PetscInt    _numRejectedSteps,_numMinSteps,_numMaxSteps;

    // for PID error control
    //~ typedef boost::circular_buffer<double> circular_buffer;
    //~ circular_buffer _errorArray;
    //~ boost::circular_buffer<double> _errorArray;



    std::map<string,Vec> _varHalfdT,_dvarHalfdT,_vardT,_dvardT,_y2,_dy2,_y3;

    PetscReal computeStepSize(const PetscReal totErr);
    PetscReal computeError();

  //~ public:

    RK32(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType);
    ~RK32();

    PetscErrorCode setTolerance(const PetscReal atol);
    PetscErrorCode setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT);
    PetscErrorCode setInitialConds(std::map<string,Vec>& var);
    PetscErrorCode setInitialCondsIm(std::map<string,Vec>& varIm) {return 0;};
    PetscErrorCode setErrInds(std::vector<string>& errInds);
    PetscErrorCode view();

    PetscErrorCode integrate(IntegratorContextEx *obj);
};


// Based on "ARK4(3)6L[2]SA-ERK" algorithm from Kennedy and Carpenter (2003):
// "Additive Runge-Kutta schemes for convection-diffusion-reaction equations"
// Note: Has matching IMEX equivalent
class RK43 : public OdeSolver
{
  public:

    PetscReal   _minDeltaT,_maxDeltaT;
    PetscReal   _atol; // absolute and relative tolerances
    PetscReal   _kappa,_ord; // safety factor in step size determinance
    PetscReal   _absErr[3]; // safety factor in step size determinance
    PetscInt    _numRejectedSteps,_numMinSteps,_numMaxSteps;

    std::map<string,Vec> _k1,_k2,_k3,_k4,_k5,_k6,_y4,_y3;
    std::map<string,Vec> _f1,_f2,_f3,_f4,_f5,_f6;
    std::map<string,Vec> _var, _dvar; // accepted stages

    PetscReal computeStepSize(const PetscReal totErr);
    PetscReal computeError();

  //~ public:

    RK43(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType);
    ~RK43();

    PetscErrorCode setTolerance(const PetscReal atol);
    PetscErrorCode setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT);
    PetscErrorCode setInitialConds(std::map<string,Vec>& var);
    PetscErrorCode setInitialCondsIm(std::map<string,Vec>& varIm) {return 0;};
    PetscErrorCode setErrInds(std::vector<string>& errInds);
    PetscErrorCode view();

    PetscErrorCode integrate(IntegratorContextEx *obj);
};

#endif

