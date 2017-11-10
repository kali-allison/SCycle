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

    virtual PetscErrorCode d_dt_WaveEq(const PetscScalar time, map<string,Vec>& var,map<string,Vec>& dvar, PetscReal _deltaT) = 0;
    virtual PetscErrorCode integrate() = 0;
    virtual PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
      const map<string,Vec>& var,const map<string,Vec>& dvar) = 0;
    virtual PetscErrorCode debug(const PetscReal time,const PetscInt stepCount,
      const map<string,Vec>& var,const map<string,Vec>& dvar, const char *stage) = 0;
};

#include "odeSolver.hpp"

#endif
