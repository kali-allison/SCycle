#ifndef INTEGRATORCONTEXT_WAVEQ_HPP_INCLUDED
#define INTEGRATORCONTEXT_WAVEQ_HPP_INCLUDED

#include <petscksp.h>
#include <vector>
#include <map>
#include "genFuncs.hpp"

/*
 * This abstract class defines an interface for OdeSolver. Classes
 * that will use OdeSolver routines must implement these virtual functions.
 */

class OdeSolver;

class IntegratorContext_WaveEq
{
  public:

    virtual PetscErrorCode integrate() = 0;


    // for intermediate time steps for explicitly integrated variables
    virtual PetscErrorCode d_dt(const PetscScalar time, map<string,Vec>& var,map<string,Vec>& dvar) = 0;

    // for time step including implicitly integrated variables
    //~ virtual PetscErrorCode d_dt(const PetscScalar time, map<string,Vec>& var,map<string,Vec>& dvar,
      //~ map<string,Vec>& varImex,map<string,Vec>& varImexPrev){return 1;};

    // for output and monitoring as time integration progresses
    // this function is not required
    virtual PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
      const map<string,Vec>& var,const map<string,Vec>& dvar,int& stopIntegration){return 1;};


};

#include "odeSolver.hpp"

#endif
