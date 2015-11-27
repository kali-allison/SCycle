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
    const char       *_file;
    std::string       _delim; // format is: var delim value (without the white space)
    std::string  _inputDir; // directory to load viscosity from

    std::string  _viscDistribution; // options: mms, layered
    std::vector<double> _viscVals,_viscDepths;
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

   PetscErrorCode setViscStrainSourceTerms(Vec& source);
   PetscErrorCode setViscStrainRates(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                                         it_vec dvarBegin,it_vec dvarEnd);
   PetscErrorCode addMMSViscStrainsAndRates(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                                         it_vec dvarBegin,it_vec dvarEnd);


    double MMS_visc(const double y,const double z);
    double MMS_epsVxy(const double y,const double z,const double t);
    PetscErrorCode MMS_epsVxy(Vec& vec,const double time);
    double MMS_epsVxy_y(const double y,const double z,const double t);
    double MMS_epsVxy_t_source(const double y,const double z,const double t);
    double MMS_epsVxz(const double y,const double z,const double t);
    PetscErrorCode MMS_epsVxz(Vec& vec,const double time);
    double MMS_epsVxz_z(const double y,const double z,const double t);
    double MMS_epsVxz_t_source(const double y,const double z,const double t);
    PetscErrorCode setMMSInitialConditions();
    PetscErrorCode setMMMSviscStrainSourceTerms(Vec& Hxsource,const PetscScalar time);
    PetscErrorCode setViscousStrainRateSAT(Vec &u, Vec &gL, Vec &gR, Vec &out);



  public:
    SymmMaxwellViscoelastic(Domain&D);
    ~SymmMaxwellViscoelastic();

    PetscErrorCode resetInitialConds();

    PetscErrorCode integrate(); // don't need now that LinearElastic defines this
    PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                     it_vec dvarBegin,it_vec dvarEnd);
    PetscErrorCode d_dt_mms(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                            it_vec dvarBegin,it_vec dvarEnd);
    PetscErrorCode d_dt_eqCycle(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                                it_vec dvarBegin,it_vec dvarEnd);
    PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec varEnd,
                             const_it_vec dvarBegin,const_it_vec dvarEnd);

    PetscErrorCode writeStep();
    PetscErrorCode view();

    PetscErrorCode measureMMSError();

    // load settings from input file
    PetscErrorCode loadSettings(const char *file);
    PetscErrorCode setVisc();
    PetscErrorCode loadFieldsFromFiles();

    // check input from file
    PetscErrorCode checkInput();
};

#endif
