#ifndef MAXWELLVISCOELASTIC_H_INCLUDED
#define MAXWELLVISCOELASTIC_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <vector>
#include "integratorContext.hpp"
#include "genFuncs.hpp"
#include "domain.hpp"
#include "linearElastic.hpp"

// models a 1D Maxwell slider assuming symmetric boundary condition
// on the fault.
class SymmMaxwellViscoelastic: public SymmLinearElastic
{
  protected:
    Vec         _visc;

    Vec         _epsVxyP,_depsVxyP; // viscoelastic strain, and strain rate
    Vec         _epsVxzP,_depsVxzP; // viscoelastic strain, and strain rate

    PetscViewer _epsVxyPV,_depsVxyPV;
    PetscViewer _epsVxzPV,_depsVxzPV;

    // additional body fields for visualization
    Vec         _epsTotxyP,_epsTotxzP; // total strain
    PetscViewer _epsTotxyPV,_epsTotxzPV;
    Vec         _stressxzP;
    PetscViewer _stressxyPV,_stressxzPV;


  public:
    SymmMaxwellViscoelastic(Domain&D);
    ~SymmMaxwellViscoelastic();

    PetscErrorCode resetInitialConds();

    PetscErrorCode integrate(); // don't need now that LinearElastic defines this
    PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                     it_vec dvarBegin,it_vec dvarEnd);
    PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec varEnd,
                             const_it_vec dvarBegin,const_it_vec dvarEnd);

    PetscErrorCode writeStep();
    PetscErrorCode view();
};

#endif
