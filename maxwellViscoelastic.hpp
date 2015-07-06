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
    PetscScalar _visc; // viscosity

    Vec         _strainVisc,_dstrainVisc; // viscoelastic strain, and strain rate
    Vec         _rhsCorrection; // mu*_strainDamper for computation of uhat
    PetscViewer _strainViscV, _dstrainViscV;

    // fields that _quadrature needs to know about
    vector<Vec>         _var;

  public:
    SymmMaxwellViscoelastic(Domain&D);
    ~SymmMaxwellViscoelastic();

    PetscErrorCode resetInitialConds();

    //~PetscErrorCode integrate(); // don't need now that LinearElastic defines this
    PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                     it_vec dvarBegin,it_vec dvarEnd);
    PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec varEnd,
                             const_it_vec dvarBegin,const_it_vec dvarEnd);

    PetscErrorCode writeStep();
    PetscErrorCode view();
};


// models a 1D Maxwell slider
class FullMaxwellViscoelastic: public FullLinearElastic
{
  protected:
    PetscScalar _visc; // viscosity

    Vec         _strainVisc,_dstrainVisc; // viscoelastic strain, and strain rate
    Vec         _rhsCorrection; // mu*_strainDamper for computation of uhat
    PetscViewer _strainViscV, _dstrainViscV;

    // fields that _quadrature needs to know about
    vector<Vec>         _var;

  public:
    SymmMaxwellViscoelastic(Domain&D);
    ~SymmMaxwellViscoelastic();

    PetscErrorCode resetInitialConds();

    //~PetscErrorCode integrate(); // don't need now that LinearElastic defines this
    PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                     it_vec dvarBegin,it_vec dvarEnd);
    PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec varEnd,
                             const_it_vec dvarBegin,const_it_vec dvarEnd);

    PetscErrorCode writeStep();
    PetscErrorCode view();
};


#endif
