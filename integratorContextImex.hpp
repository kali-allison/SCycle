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

    virtual PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx){return 1;};
    virtual PetscErrorCode d_dt_qd(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx) {return 1;};
    virtual PetscErrorCode d_dt_dyn(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx) {return 1;};

    // backward Euler
    virtual PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt){return 1;};
    virtual PetscErrorCode d_dt_qd(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt){return 1;};
    virtual PetscErrorCode d_dt_dyn(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt){return 1;};

    virtual PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,const map<string,Vec>& varImex,int& stopIntegration){return 1;};
    virtual PetscErrorCode timeMonitor_qd(const PetscReal time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,const map<string,Vec>& varImex,int& stopIntegration){return 1;};
    virtual PetscErrorCode timeMonitor_dyn(const PetscReal time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,const map<string,Vec>& varImex,int& stopIntegration){return 1;};
};

#include "odeSolver.hpp"

#endif
