#ifndef POWERLAW_H_INCLUDED
#define POWERLAW_H_INCLUDED

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
class PowerLaw: public SymmLinearElastic
{
  protected:
    const char       *_file;
    std::string       _delim; // format is: var delim value (without the white space)
    std::string       _inputDir; // directory to load viscosity from

    // material properties
    std::string  _viscDistribution; // options: mms, fromVector,loadFromFile
    std::string  _AFile,_BFile,_nFile,_TFile; // names of each file within loadFromFile
    std::vector<double> _AVals,_ADepths,_nVals,_nDepths,_BVals,_BDepths,_TVals,_TDepths;
    Vec         _A,_n,_B;
    Vec         _effVisc;

    Vec         _stressxzP,_sigmadev; // sigma_xz (MPa), deviatoric stress (MPa)
    Vec         _epsVxyP,_depsVxyP; // viscoelastic strain, strain rate
    Vec         _epsVxzP,_depsVxzP; // viscoelastic strain, strain rate
    Vec         _epsTotxyP,_epsTotxzP; // total strain
    Vec         _T; // temperature (K)

    // viewers
    PetscViewer _stressxyPV,_stressxzPV,_sigmadevV;
    PetscViewer _epsTotxyPV,_epsTotxzPV;
    PetscViewer _epsVxyPV,_depsVxyPV;
    PetscViewer _epsVxzPV,_depsVxzPV;
    PetscViewer _TV;
    PetscViewer _effViscV;

    PetscErrorCode setViscStrainSourceTerms(Vec& source,const_it_vec varBegin,const_it_vec varEnd);
    PetscErrorCode setViscStrainRates(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                                          it_vec dvarBegin,it_vec dvarEnd);
     PetscErrorCode setStresses(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd);
    PetscErrorCode addMMSViscStrainsAndRates(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                                          it_vec dvarBegin,it_vec dvarEnd);
    PetscErrorCode debug(const PetscReal time,const PetscInt stepCount,
                           const_it_vec varBegin,const_it_vec varEnd,
                           const_it_vec dvarBegin,const_it_vec dvarEnd,const char *stage);


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
    PowerLaw(Domain&D);
    ~PowerLaw();

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

    PetscErrorCode writeContext(const string outputDir);
    PetscErrorCode writeStep1D();
    PetscErrorCode writeStep2D();
    PetscErrorCode view();

    PetscErrorCode measureMMSError();

    // load settings from input file
    PetscErrorCode loadSettings(const char *file);
    PetscErrorCode setFields();
    PetscErrorCode setVecFromVectors(Vec& vec, vector<double>& vals,vector<double>& depths);
    PetscErrorCode loadFieldsFromFiles();

    // check input from file
    PetscErrorCode checkInput();
};

#endif
