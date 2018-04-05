#ifndef ODESOLVER_WAVEEQ_HPP_INCLUDED
#define ODESOLVER_WAVEEQ_HPP_INCLUDED

#include <petscts.h>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <assert.h>
#include "integratorContextWave.hpp"
#include "genFuncs.hpp"


class OdeSolver_WaveEq
{
  protected:

    PetscScalar               _initT,_finalT,_currT,_deltaT;
    PetscInt                _maxNumSteps,_stepCount;
    std::map<string,Vec>    _var,_varPrev; // integration variable and rate
    int                     _lenVar;
    double                  _runTime;

  public:

    OdeSolver_WaveEq(PetscInt maxNumSteps,PetscScalar initT,PetscScalar finalT,PetscScalar deltaT);
    virtual ~OdeSolver_WaveEq() {};

    PetscErrorCode setStepSize(const PetscReal deltaT);
    PetscErrorCode setInitialConds(std::map<string,Vec>& var);

    PetscErrorCode view();
    PetscErrorCode integrate(IntegratorContextWave *obj);
    PetscErrorCode integrate_switch(IntegratorContextWave *obj);
    PetscErrorCode getCurrT(PetscScalar& currT);
};

#endif

