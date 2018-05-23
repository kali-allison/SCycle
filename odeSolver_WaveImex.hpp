#ifndef ODESOLVER_WAVEIMEX_HPP_INCLUDED
#define ODESOLVER_WAVEIMEX_HPP_INCLUDED

#include <petscts.h>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <assert.h>
#include "integratorContextWave.hpp"
#include "genFuncs.hpp"


class OdeSolver_WaveImex
{
  public:

    PetscScalar               _initT,_finalT,_currT,_deltaT;
    PetscInt                _maxNumSteps,_stepCount;
    std::map<string,Vec>    _varEx,_varPrev, _varImex, _varImexPrev; // integration variable and rate
    int                     _lenVar;
    double                  _runTime;

  public:

    OdeSolver_WaveImex(PetscInt maxNumSteps,PetscScalar initT,PetscScalar finalT,PetscScalar deltaT);
    virtual ~OdeSolver_WaveImex() {};

    PetscErrorCode setStepSize(const PetscReal deltaT);
    PetscErrorCode setInitialConds(std::map<string,Vec>& varEx, std::map<string,Vec>& varImex);

    PetscErrorCode view();
    PetscErrorCode integrate(IntegratorContextWave *obj);
    PetscErrorCode getCurrT(PetscScalar& currT);
    std::map<string,Vec>& getVar(){return _varEx;};
};

#endif

