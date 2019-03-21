#ifndef INTEGRATORCONTEXTIMEX_HPP_INCLUDED
#define INTEGRATORCONTEXTIMEX_HPP_INCLUDED

#include <petscksp.h>
#include <vector>
#include "genFuncs.hpp"

/*
 * This abstract class defines an interface for OdeSolver for IMEX methods. Classes
 * that will use OdeSolverImex routines must implement these virtual functions.
 */

class OdeSolver;

class IntegratorContextImex
{
  public:

    virtual PetscErrorCode integrate() = 0;

    // for intermediate time steps for explicitly integrated variables
  virtual PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx) = 0;

    // for time step including implicitly integrated variables
  virtual PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx, map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt) = 0;

    // for output and monitoring as time integration progresses
    // this function is not required
    virtual PetscErrorCode timeMonitor(const PetscReal time,const PetscScalar deltaT, const PetscInt stepCount,int& stopIntegration){return 1;};

};

#include "odeSolver.hpp"

#endif
