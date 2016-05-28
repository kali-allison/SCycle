#ifndef POWERLAW_H_INCLUDED
#define POWERLAW_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <vector>
#include "integratorContextEx.hpp"
#include "genFuncs.hpp"
#include "domain.hpp"
#include "linearElastic.hpp"
#include "heatEquation.hpp"

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
    std::vector<double> _AVals,_ADepths,_nVals,_nDepths,_BVals,_BDepths;
    Vec         _A,_n,_B;
    Vec         _effVisc;

    Vec         _stressxzP,_sigmadev; // sigma_xz (MPa), deviatoric stress (MPa)
    Vec         _gxyP,_dgxyP; // viscoelastic strain, strain rate
    Vec         _gxzP,_dgxzP; // viscoelastic strain, strain rate
    Vec         _gTxyP,_gTxzP; // total strain
    Vec         _T; // temperature (K)

    // viewers
    PetscViewer _stressxyPV,_stressxzPV,_sigmadevV;
    PetscViewer _gTxyPV,_gTxzPV;
    PetscViewer _gxyPV,_dgxyPV;
    PetscViewer _gxzPV,_dgxzPV;
    PetscViewer _TV;
    PetscViewer _effViscV;

    PetscErrorCode setViscStrainSourceTerms(Vec& source,const_it_vec varBegin);
    PetscErrorCode setViscStrainRates(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin);
    PetscErrorCode setViscousStrainRateSAT(Vec &u, Vec &gL, Vec &gR, Vec &out);
    PetscErrorCode setStresses(const PetscScalar time,const_it_vec varBegin);

    PetscErrorCode debug(const PetscReal time,const PetscInt stepCount,
                     const_it_vec varBegin,const_it_vec dvarBegin,const char *stage);

    PetscErrorCode setMMSInitialConditions();
    PetscErrorCode setMMSBoundaryConditions(const double time);

  public:
    PowerLaw(Domain&D);
    ~PowerLaw();

    PetscErrorCode resetInitialConds();

    PetscErrorCode integrate(); // don't need now that LinearElastic defines this

    // methods for explicit time stepping
    PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin);
    PetscErrorCode d_dt_mms(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin);
    PetscErrorCode d_dt_eqCycle(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin);
    PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
                     const_it_vec varBegin,const_it_vec dvarBegin);

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
