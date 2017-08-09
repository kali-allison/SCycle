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


    //~ Vec         _T; // temperature (K)

    // viewers
    PetscViewer _sxyPV,_sxzPV,_sigmadevV;
    PetscViewer _gTxyPV,_gTxzPV;
    PetscViewer _gxyPV,_dgxyPV;
    PetscViewer _gxzPV,_dgxzPV;
    PetscViewer _TV;
    PetscViewer _effViscV;

    // initialize and set data
    PetscErrorCode loadSettings(const char *file); // load settings from input file
    PetscErrorCode checkInput(); // check input from file
    PetscErrorCode allocateFields(); // allocate space for member fields
    PetscErrorCode setMaterialParameters();
    PetscErrorCode loadEffViscFromFiles();
    PetscErrorCode setSSInitialConds(Domain& D); // try to skip some spin up steps
    PetscErrorCode guessSteadyStateEffVisc(); // inititialize effective viscosity
    PetscErrorCode loadFieldsFromFiles(); // load non-effective-viscosity parameters

    // functions needed each time step
    PetscErrorCode computeMaxTimeStep(PetscScalar& maxTimeStep); // limited by Maxwell time
    //~ PetscErrorCode setViscStrainSourceTerms(Vec& source,const_it_vec varBegin);
    PetscErrorCode setViscStrainSourceTerms(Vec& source,Vec& gxy, Vec& gxz);
    //~ PetscErrorCode setViscStrainRates(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin);
    PetscErrorCode setViscStrainRates(const PetscScalar time,const Vec& gVxy, const Vec& gVxz,
      Vec& gVxy_t, Vec& gVxz_t);
    PetscErrorCode setViscousStrainRateSAT(Vec &u, Vec &gL, Vec &gR, Vec &out);
    PetscErrorCode setStresses(const PetscScalar time);
    PetscErrorCode computeViscosity();

    PetscErrorCode debug(const PetscReal time,const PetscInt stepCount,
                     const_it_vec varBegin,const_it_vec dvarBegin,const char *stage);

    PetscErrorCode setMMSInitialConditions();
    PetscErrorCode setMMSBoundaryConditions(const double time);

    PetscErrorCode setVecFromVectors(Vec& vec, vector<double>& vals,vector<double>& depths);

  public:

    Vec         _sxzP,_sigmadev; // sigma_xz (MPa), deviatoric stress (MPa)
    Vec         _gxyP,_dgxyP; // viscoelastic strain, strain rate
    Vec         _gxzP,_dgxzP; // viscoelastic strain, strain rate
    Vec         _gTxyP,_gTxzP; // total strain

    PowerLaw(Domain&D);
    ~PowerLaw();


    PetscErrorCode resetInitialConds();

    PetscErrorCode integrate(); // don't need now that LinearElastic defines this

    // methods for explicit time stepping
    PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin);
    PetscErrorCode d_dt_mms(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin);
    PetscErrorCode d_dt_eqCycle(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin);
    PetscErrorCode d_dt(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin,
      it_vec varBeginIm,const_it_vec varBeginImo,const PetscScalar dt);
    PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec dvarBegin);
    PetscErrorCode timeMonitor(const PetscReal time,const PetscInt stepCount);

    PetscErrorCode writeDomain();
    PetscErrorCode writeContext();
    PetscErrorCode writeStep1D();
    PetscErrorCode writeStep2D();
    PetscErrorCode view();

    PetscErrorCode measureMMSError();

    // currently investigating utility
    PetscErrorCode psuedoTS_main();
    PetscErrorCode psuedoTS_computeIJacobian(Mat& J,PetscReal time,Vec& g,Vec& g_t);
    PetscErrorCode psuedoTS_computeJacobian(Mat& J,PetscReal time,Vec& g);
    PetscErrorCode psuedoTS_evaluateIRHS(Vec&F,PetscReal time,Vec& g,Vec& g_t);
    PetscErrorCode psuedoTS_evaluateRHS(Vec&F,PetscReal time,Vec& g);

};

PetscErrorCode computeIJacobian(TS ts,PetscReal t,Vec g,Vec g_t,PetscReal a,Mat Amat,Mat Pmat,void *ctx);
PetscErrorCode computeJacobian(TS ts,PetscReal t,Vec g,Mat Amat,Mat Pmat,void *ctx);
PetscErrorCode evaluateIRHS(TS ts,PetscReal t,Vec g,Vec g_t,Vec F,void *ptr);
PetscErrorCode evaluateRHS(TS ts,PetscReal t,Vec g,Vec F,void *ptr);
PetscErrorCode monitor(TS ts,PetscInt stepCount,PetscReal time,Vec g,void *ptr);

#endif
