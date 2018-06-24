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
    std::map<string,Vec>    _varEx,_varPrev, _varImex, _varImexPrev; // integration variable and rate
    int                     _lenVar;
    double                  _runTime;

  public:

    OdeSolver_WaveEq_Imex(PetscInt maxNumSteps,PetscScalar initT,PetscScalar finalT,PetscScalar deltaT);
    virtual ~OdeSolver_WaveEq_Imex() {};

    PetscErrorCode setStepSize(const PetscReal deltaT);
    PetscErrorCode setInitialConds(std::map<string,Vec>& varEx, std::map<string,Vec>& varImex);

    PetscErrorCode view();
    PetscErrorCode integrate(IntegratorContext_WaveEq_Imex *obj);

    std::map<string,Vec>& getVar(){return _varEx;};
};

#endif

