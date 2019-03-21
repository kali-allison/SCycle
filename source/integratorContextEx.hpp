#ifndef IntegratorContextEx_HPP_INCLUDED
#define IntegratorContextEx_HPP_INCLUDED

#include <petscksp.h>
#include <vector>
#include <map>
#include "genFuncs.hpp"

/*
 * This abstract class defines an interface for OdeSolver. Classes
 * that will use OdeSolver routines must implement these virtual functions.
 */

class OdeSolver;

class IntegratorContextEx
{
  public:

  virtual PetscErrorCode integrate() = 0;

  virtual PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& var,map<string,Vec>& dvar) = 0;

  // for output and monitoring as time integration progresses
  // this function is not required
  virtual PetscErrorCode timeMonitor(const PetscReal time,const PetscScalar deltaT, const PetscInt stepCount,int& stopIntegration){return 1;};
};

#include "odeSolver.hpp"

#endif
