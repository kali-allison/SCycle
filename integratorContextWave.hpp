#ifndef IntegratorContextWave_HPP_INCLUDED
#define IntegratorContextWave_HPP_INCLUDED

#include <petscksp.h>
#include <vector>
#include <map>
#include "genFuncs.hpp"
#include "domain.hpp"

/*
 * This abstract class defines an interface for OdeSolver. Classes
 * that will use OdeSolver routines must implement these virtual functions.
 */

class OdeSolver;

class IntegratorContextWave
{
  public:

    virtual PetscErrorCode d_dt(const PetscScalar time, map<string,Vec>& var,map<string,Vec>& dvar){return 1;};
    virtual PetscErrorCode d_dt(const PetscScalar time, map<string,Vec>& var,map<string,Vec>& dvar,
                                map<string,Vec>& varImex,map<string,Vec>& varImexPrev){return 1;};
    virtual PetscErrorCode d_dt_dyn(const PetscScalar time, map<string,Vec>& var,map<string,Vec>& dvar){return 1;};
    virtual PetscErrorCode integrate() = 0;
    virtual PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
      const map<string,Vec>& var,const map<string,Vec>& dvar,int& stopIntegration){return 1;};
    virtual PetscErrorCode timeMonitor_dyn(const PetscReal time,const PetscInt stepCount,
      const map<string,Vec>& var,const map<string,Vec>& dvar,int& stopIntegration){return 1;};
    virtual PetscErrorCode debug(const PetscReal time,const PetscInt stepCount,
      const map<string,Vec>& var,const map<string,Vec>& dvar, const char *stage){return 1;};
};

#include "odeSolver.hpp"

#endif
