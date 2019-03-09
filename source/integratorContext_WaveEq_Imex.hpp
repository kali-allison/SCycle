#ifndef INTEGRATORCONTEXT_WAVEQ_IMEX_HPP_INCLUDED
#define INTEGRATORCONTEXT_WAVEQ_IMEX_HPP_INCLUDED

#include <petscksp.h>
#include <vector>
#include <map>
#include "genFuncs.hpp"

/*
 * This abstract class defines an interface for OdeSolver. Classes
 * that will use OdeSolver routines must implement these virtual functions.
 */

class OdeSolver;

class IntegratorContext_WaveEq_Imex
{
  public:

    virtual PetscErrorCode integrate() = 0;


    // for time step including implicitly integrated variables
  virtual PetscErrorCode d_dt(const PetscScalar time, const PetscScalar deltaT, map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev, map<string,Vec>& varIm, const map<string,Vec>& varImPrev) = 0;

    // for output and monitoring as time integration progresses
    // this function is not required
    virtual PetscErrorCode timeMonitor(const PetscReal time,const PetscScalar deltaT, const PetscInt stepCount,int& stopIntegration){return 1;};


};

#include "odeSolver.hpp"

#endif
