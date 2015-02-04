#ifndef ASTHENOSPHERE_H_INCLUDED
#define ASTHENOSPHERE_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <vector>
#include "integratorContext.hpp"
#include "domain.hpp"
#include "lithosphere.hpp"

// for models consisting of coupled spring sliders, no damping
class OnlyAsthenosphere: public FullLithosphere
{
  protected:
    PetscScalar _visc; // viscosity

    Vec         _strainDamper,_strainDamperRate; // inelastic strain from damper
    Vec         _rhsCorrection; // mu*_strainDamper for computation of uhat
    PetscViewer _strainDamperViewer;
    PetscViewer _strainDamperRateViewer;

    // fields that _quadrature needs to know about
    vector<Vec>         _var;

  public:
    OnlyAsthenosphere(Domain&D);
    ~OnlyAsthenosphere();

    PetscErrorCode resetInitialConds();

    PetscErrorCode integrate();
    PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                     it_vec dvarBegin,it_vec dvarEnd);
    PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec varEnd,
                             const_it_vec dvarBegin,const_it_vec dvarEnd);

    PetscErrorCode writeStep();
    PetscErrorCode view();
};


#endif
