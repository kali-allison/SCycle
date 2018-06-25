#ifndef ODESOLVER_WAVEEQ_HPP_INCLUDED
#define ODESOLVER_WAVEEQ_HPP_INCLUDED

#include <petscts.h>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <assert.h>
#include "integratorContext_WaveEq.hpp"
#include "genFuncs.hpp"


class OdeSolver_WaveEq
{
  public:

    PetscScalar             _initT,_finalT,_currT,_deltaT;
    PetscInt                _maxNumSteps,_stepCount;
    std::map<string,Vec>    _varNext,_var,_varPrev; // variable at time step: n+1, n, n-1
    int                     _lenVar;
    double                  _runTime;

  public:

    OdeSolver_WaveEq(PetscInt maxNumSteps,PetscScalar initT,PetscScalar finalT,PetscScalar deltaT);
    ~OdeSolver_WaveEq();

    PetscErrorCode setStepSize(const PetscReal deltaT);
    PetscErrorCode setInitialStepCount(const PetscReal stepCount);
    PetscErrorCode setInitialConds(std::map<string,Vec>& var);
    PetscErrorCode setInitialConds(std::map<string,Vec>& var, std::map<string,Vec>& varPrev);

    PetscErrorCode view();
    PetscErrorCode integrate(IntegratorContext_WaveEq *obj);

    std::map<string,Vec>& getVar(){return _var;};
};

#endif

