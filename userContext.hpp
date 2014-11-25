#ifndef USERCONTEXT_H_INCLUDED
#define USERCONTEXT_H_INCLUDED

#include <petscksp.h>
#include <vector>

/*
 * This abstract class defines an interface for OdeSolver. Classes
 * that will use OdeSolver routines must implement these virtual functions.
 */

class OdeSolver;


class UserContext
{
  public:

    typedef typename std::vector<Vec>::iterator it_vec;
    typedef typename std::vector<Vec>::const_iterator const_it_vec;

    //~virtual PetscErrorCode d_dt(const PetscScalar time,const std::vector<Vec>& var,std::vector<Vec>& dvar) = 0;
    virtual PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
              it_vec dvarBegin,it_vec dvarEnd) = 0;
    virtual PetscErrorCode integrate() = 0;
    virtual PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec varEnd,
                             const_it_vec dvarBegin,const_it_vec dvarEnd) = 0;
                             //~const std::vector<Vec>& var,const std::vector<Vec>& dvar) = 0;
};

#include "odeSolver.hpp"

#endif
