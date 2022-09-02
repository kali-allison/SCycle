#ifndef ProblemContext_HPP_INCLUDED
#define ProblemContext_HPP_INCLUDED

#include <petscksp.h>
#include <vector>
#include <map>
#include "genFuncs.hpp"

/*
 * This abstract class defines an interface for the mediator classes, such as
 * StrikeSlip_linearElastic_qd.
 * Classes which inherit from this must implement these virtual functions.
 */

class ProblemContext
{
  public:

  virtual ~ProblemContext(){};

  virtual PetscErrorCode initiateIntegrand() = 0;
  virtual PetscErrorCode integrate() = 0;
  virtual PetscErrorCode writeContext() = 0;
  virtual PetscErrorCode view() = 0;

  // for running steady-state algorithm. Not used by every problem.
  // this function is not required
  virtual PetscErrorCode integrateSS(){return 1;};

};

#endif
