#ifndef ODESOLVER_WAVEIMEX_HPP_INCLUDED
#define ODESOLVER_WAVEIMEX_HPP_INCLUDED

#include <petscts.h>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <assert.h>
#include "integratorContext_WaveEq_Imex.hpp"
#include "genFuncs.hpp"


class OdeSolver_WaveEq_Imex
{
  public:

    PetscScalar             _initT,_finalT,_currT,_deltaT;
    PetscInt                _maxNumSteps,_stepCount;
    std::map<string,Vec>    _varNext,_var,_varPrev; // variable at time step: n+1, n, n-1
    std::map<string,Vec>    _varIm, _varImPrev; // integration variable and rate
    int                     _lenVar;
    double                  _runTime;

  public:

    OdeSolver_WaveEq_Imex(PetscInt maxNumSteps,PetscScalar initT,PetscScalar finalT,PetscScalar deltaT);
    virtual ~OdeSolver_WaveEq_Imex() {};

    PetscErrorCode setStepSize(const PetscReal deltaT);
    PetscErrorCode setInitialStepCount(const PetscReal stepCount);
    PetscErrorCode setTimeRange(const PetscReal initT,const PetscReal finalT);
    PetscErrorCode setInitialConds(std::map<string,Vec>& varEx, std::map<string,Vec>& varIm);
    PetscErrorCode setInitialConds(std::map<string,Vec>& varEx,std::map<string,Vec>& varExPrev, std::map<string,Vec>& varIm);

    PetscErrorCode view();
    PetscErrorCode integrate(IntegratorContext_WaveEq_Imex *obj);

    std::map<string,Vec>& getVar(){return _var;};
};

#endif

