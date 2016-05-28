#ifndef IntegratorContextEx_HPP_INCLUDED
#define IntegratorContextEx_HPP_INCLUDED

#include <petscksp.h>
#include <vector>
#include "genFuncs.hpp"

/*
 * This abstract class defines an interface for OdeSolver. Classes
 * that will use OdeSolver routines must implement these virtual functions.
 */

class OdeSolver;

class IntegratorContextEx
{
  public:

    typedef std::vector<Vec>::iterator it_vec;
    typedef std::vector<Vec>::const_iterator const_it_vec;

    virtual PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin) = 0;
    virtual PetscErrorCode integrate() = 0;
    virtual PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec dvarBegin) = 0;
    virtual PetscErrorCode debug(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec dvarBegin, const char *stage) = 0;
};

#include "odeSolver.hpp"

#endif
