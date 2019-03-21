#ifndef ROOTFINDERCONTEXT_HPP_INCLUDED
#define ROOTFINDERCONTEXT_HPP_INCLUDED

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

    // for bisection method. Final argument is output.
    virtual PetscErrorCode getResid(const PetscInt, const PetscScalar, PetscScalar*) = 0;

    // for bracketed Newton method. Final arguments are output and Jacobian, respectively.
    virtual PetscErrorCode getResid(const PetscInt, const PetscScalar, PetscScalar*,PetscScalar*) = 0;

};

#include "rootFinder.hpp"

#endif
