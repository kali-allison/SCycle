#ifndef ROOTFINDERCONTEXT_H_INCLUDED
#define ROOTFINDERCONTEXT_H_INCLUDED

#include <petscksp.h>
#include <vector>

/*
 * This abstract class defines an interface for RootFinder. Classes
 * that will use RootFinder routines must implement the virtual function.
 */

class RootFinder;

class RootFinderContext
{
  public:

    virtual PetscErrorCode getResid(const PetscInt ind,const PetscScalar vel,PetscScalar *out) = 0;
};

#include "rootFinder.hpp"

#endif
