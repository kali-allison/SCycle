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

    //~ typedef std::vector<Vec>::iterator it_vec;
    //~ typedef std::vector<Vec>::const_iterator const_it_vec;

    virtual PetscErrorCode integrate() = 0;

     //~ // explicit only
    //~ virtual PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin) = 0;

    //~ // backward Euler
    //~ virtual PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBeginEx,it_vec dvarBeginEx,
                             //~ it_vec varBeginIm,const_it_vec varBeginImo,const PetscScalar dt) = 0;

    //~ virtual PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
                             //~ const_it_vec varBeginEx,const_it_vec dvarBeginEx) = 0;
    //~ // virtual PetscErrorCode debug(const PetscReal time,const PetscInt stepCount,const_it_vec varBeginEx,
                             //~ const_it_vec dvarBeginEx,const_it_vec varBeginIm,const char *stage) = 0;


    virtual PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx) = 0;

    // backward Euler
    virtual PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt) = 0;

    virtual PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,const map<string,Vec>& varImex) = 0;
    //~ virtual PetscErrorCode debug(const PetscReal time,const PetscInt stepCount,const_it_vec varBeginEx,
                             //~ const_it_vec dvarBeginEx,const_it_vec varBeginIm,const char *stage) = 0;
};

#include "odeSolver.hpp"

#endif
