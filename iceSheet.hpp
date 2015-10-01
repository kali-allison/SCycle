#ifndef ICESHEET_H_INCLUDED
#define ICESHEET_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <vector>
#include "integratorContext.hpp"
#include "genFuncs.hpp"
#include "domain.hpp"
#include "maxwellViscoelastic.hpp"

// models a 1D Maxwell slider assuming symmetric boundary condition
// on the fault.
class IceSheet: public SymmMaxwellViscoelastic
{
  protected:
    // material properties for power law
    string     _inputDir; // location of binary files containing fields
    Vec         _A,_n,_temp;

    Vec         _epsVxyP,_depsVxyP; // viscoelastic strain, and strain rate
    Vec         _epsVxzP,_depsVxzP;

    PetscViewer _epsVxyPV,_depsVxyPV;
    PetscViewer _epsVxzPV,_depsVxzPV;

    // additional body fields for visualization
    Vec         _epsTotxyP,_epsTotxzP; // total strain
    PetscViewer _epsTotxyPV,_epsTotxzPV;
    Vec         _stressxzP;
    PetscViewer _stressxyPV,_stressxzPV;

   PetscErrorCode setViscStrainsAndRates(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                                         it_vec dvarBegin,it_vec dvarEnd);

  public:
    IceSheet(Domain&D);
    ~IceSheet();

    PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                     it_vec dvarBegin,it_vec dvarEnd);
    PetscErrorCode d_dt_mms(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                            it_vec dvarBegin,it_vec dvarEnd);
    PetscErrorCode d_dt_eqCycle(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                                it_vec dvarBegin,it_vec dvarEnd);
};

#endif
