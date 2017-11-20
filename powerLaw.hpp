#ifndef POWERLAW_H_INCLUDED
#define POWERLAW_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <vector>
#include "integratorContextEx.hpp"
#include "momBalContext.hpp"
#include "genFuncs.hpp"
#include "domain.hpp"
#include "linearElastic.hpp"
#include "heatEquation.hpp"

// models a 1D Maxwell slider assuming symmetric boundary condition
// on the fault.
class PowerLaw : public MomBalContext
//~ class PowerLaw
{
  protected:
    const char          *_file;
    std::string          _delim; // format is: var delim value (without the white space)
    std::string          _inputDir; // directory to load fields from
    std::string          _outputDir;  // output data
    const PetscInt       _order,_Ny,_Nz;
    PetscScalar          _Ly,_Lz,_dy,_dz;
    Vec                 *_y,*_z; // to handle variable grid spacing
    const bool           _isMMS; // true if running mms test
    const bool           _loadICs; // true if starting from a previous simulation
    std::string          _momBalType; // "dynamic", "static"
    std::string          _thermalCoupling;
    bool                 _bcLTauQS; // true if left boundary is traction
    PetscScalar          _currTime;
    PetscInt             _stepCount;
    PetscScalar          _vL; // loading velocity

    // material properties
    Vec                  _muVec, _rhoVec, _cs, _ay;
    PetscScalar          _muVal,_rhoVal; // if constant
    Vec                  _bcRShift,_surfDisp;
    Vec                  _rhs;
    std::string  _viscDistribution; // options: mms, fromVector,loadFromFile
    std::string  _AFile,_BFile,_nFile,_TFile; // names of each file within loadFromFile
    std::vector<double> _AVals,_ADepths,_nVals,_nDepths,_BVals,_BDepths;
    Vec         _A,_n,_QR,_T;
    Vec         _effVisc;
    Vec         SATL;
    PetscScalar _effViscCap; // imposed upper limit on effective viscosity

    // linear system data
    std::string          _linSolver;
    KSP                  _ksp;
    PC                   _pc;
    PetscScalar          _kspTol;
    SbpOps              *_sbp;
    std::string          _sbpType;
    Mat                  _B,_C; // composite matrices to make momentum balance simpler
    PetscErrorCode   initializeMomBalMats(); // computes B and C

    // for steady-state computations
    SbpOps   *_sbp_eta;
    KSP       _ksp_eta;
    PC        _pc_eta;
    PetscScalar _effViscCapSS; // imposed upper limit on effective viscosity for steady state computation
    PetscErrorCode initializeSSMatrices(); // compute Bss and Css

    // viewers
    PetscViewer      _timeV1D,_timeV2D;
    std::map <string,PetscViewer>  _viewers;

    // runtime data
    double       _integrateTime,_writeTime,_linSolveTime,_factorTime,_startTime,_miscTime;
    PetscInt     _linSolveCount;

    // initialize and set data
    PetscErrorCode loadSettings(const char *file); // load settings from input file
    PetscErrorCode checkInput(); // check input from file
    PetscErrorCode allocateFields(); // allocate space for member fields
    PetscErrorCode setMaterialParameters();
    PetscErrorCode loadEffViscFromFiles();
    PetscErrorCode setSSInitialConds(Domain& D,Vec& tauRS); // try to skip some spin up steps
    PetscErrorCode guessSteadyStateEffVisc(const PetscScalar ess_t); // inititialize effective viscosity
    PetscErrorCode loadFieldsFromFiles(); // load non-effective-viscosity parameters
    PetscErrorCode setUpSBPContext(Domain& D);
    PetscErrorCode setupKSP(SbpOps* sbp,KSP& ksp,PC& pc);

    // functions needed each time step
    PetscErrorCode setViscStrainSourceTerms(Vec& source,Vec& gxy, Vec& gxz);
    PetscErrorCode setViscStrainRates(const PetscScalar time,const Vec& gVxy, const Vec& gVxz,
      Vec& gVxy_t, Vec& gVxz_t);
    PetscErrorCode setViscousStrainRateSAT(Vec &u, Vec &gL, Vec &gR, Vec &out);
    PetscErrorCode computeTotalStrains(const PetscScalar time);
    PetscErrorCode computeStresses(const PetscScalar time);
    PetscErrorCode computeSDev();
    PetscErrorCode computeViscosity(const PetscScalar viscCap);
    PetscErrorCode setSurfDisp();

    PetscErrorCode setInitialSlip(Vec& out);
    PetscErrorCode setMMSInitialConditions();
    PetscErrorCode setMMSBoundaryConditions(const double time);

    PetscErrorCode setVecFromVectors(Vec& vec, vector<double>& vals,vector<double>& depths);

  public:

    Vec          _u;
    Vec          _sxy,_sxz,_sdev; // sigma_xz (MPa), deviatoric stress (MPa)
    Vec          _gxy,_dgxy; // viscoelastic strain and strain rate
    Vec          _gxz,_dgxz; // viscoelastic strain and strain rate
    Vec          _gTxy,_gTxz; // total strain
    std::string  _bcTType,_bcRType,_bcBType,_bcLType; // options: displacement, traction
    Vec          _bcT,_bcR,_bcB,_bcL;

    PowerLaw(Domain& D,HeatEquation& he);
    ~PowerLaw();


    // for steady state computations
    PetscErrorCode getTauVisc(Vec& tauVisc, const PetscScalar ess_t); // compute initial tauVisc
    PetscErrorCode updateSSa(map<string,Vec>& varSS); // update v, viscous strain rates, viscosity
    PetscErrorCode updateSSb(map<string,Vec>& varSS); // update v, viscous strain rates, viscosity
    PetscErrorCode initiateVarSS(map<string,Vec>& varSS); // put viscous strains etc in varSS

    // methods for explicit time stepping
    PetscErrorCode initiateIntegrand_qs(const PetscScalar time,map<string,Vec>& varEx);
    PetscErrorCode initiateIntegrand_dyn(const PetscScalar time,map<string,Vec>& varEx) { return 0; };
    PetscErrorCode updateFields(const PetscScalar time,const map<string,Vec>& varEx);
    PetscErrorCode updateTemperature(const Vec& T);
    PetscErrorCode computeMaxTimeStep(PetscScalar& maxTimeStep); // limited by Maxwell time
    PetscErrorCode getStresses(Vec& sxy, Vec& sxz, Vec& sdev);

    // methods for explicit time stepping
    PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);
    PetscErrorCode d_dt_mms(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);
    PetscErrorCode d_dt_eqCycle(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);
    PetscErrorCode d_dt_WaveEq(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& dvarEx, PetscScalar _deltaT) { return 0; };

    // methods for implicit/explicit time stepping
    PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt);

    PetscErrorCode timeMonitor(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx);
    PetscErrorCode timeMonitor(const PetscScalar time,const PetscInt stepCount);
    PetscErrorCode debug(const PetscReal time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,const char *stage) {return 0;};

    PetscErrorCode measureMMSError(const PetscScalar time);

    PetscErrorCode writeDomain(const std::string outputDir);
    PetscErrorCode writeContext(const std::string outputDir);
    PetscErrorCode writeStep1D(const PetscInt stepCount, const PetscScalar time);
    PetscErrorCode writeStep2D(const PetscInt stepCount, const PetscScalar time);
    PetscErrorCode writeStep1D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir);
    PetscErrorCode writeStep2D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir);
    PetscErrorCode view(const double totRunTime);




    // MMS functions
    static double zzmms_f(const double y,const double z);
    static double zzmms_f_y(const double y,const double z);
    static double zzmms_f_yy(const double y,const double z);
    static double zzmms_f_z(const double y,const double z);
    static double zzmms_f_zz(const double y,const double z);
    static double zzmms_g(const double t);
    static double zzmms_g_t(const double t);

    static double zzmms_mu(const double y,const double z);
    static double zzmms_mu_y(const double y,const double z);
    static double zzmms_mu_z(const double y,const double z);

    static double zzmms_uA(const double y,const double z,const double t);
    static double zzmms_uA_y(const double y,const double z,const double t);
    static double zzmms_uA_yy(const double y,const double z,const double t);
    static double zzmms_uA_z(const double y,const double z,const double t);
    static double zzmms_uA_zz(const double y,const double z,const double t);
    static double zzmms_uA_t(const double y,const double z,const double t);
    static double zzmms_sigmaxy(const double y,const double z,const double t);
    static double zzmms_sigmaxz(const double y,const double z,const double t);
    static double zzmms_uSource(const double y,const double z,const double t);


    static double zzmms_visc(const double y,const double z);
    static double zzmms_invVisc(const double y,const double z);
    static double zzmms_invVisc_y(const double y,const double z);
    static double zzmms_invVisc_z(const double y,const double z);

    static double zzmms_gxy(const double y,const double z,const double t);
    static double zzmms_gxy_y(const double y,const double z,const double t);
    static double zzmms_gxy_t(const double y,const double z,const double t);

    static double zzmms_gxz(const double y,const double z,const double t);
    static double zzmms_gxz_z(const double y,const double z,const double t);
    static double zzmms_gxz_t(const double y,const double z,const double t);
    static double zzmms_max_gxy_t_source(const double y,const double z,const double t);
    static double zzmms_max_gxz_t_source(const double y,const double z,const double t);
    static double zzmms_gSource(const double y,const double z,const double t);

    static double zzmms_A(const double y,const double z);
    static double zzmms_B(const double y,const double z);
    static double zzmms_T(const double y,const double z);
    static double zzmms_n(const double y,const double z);
    static double zzmms_pl_sigmaxy(const double y,const double z,const double t);
    static double zzmms_pl_sigmaxz(const double y,const double z,const double t);
    static double zzmms_sdev(const double y,const double z,const double t);

    static double zzmms_pl_gSource(const double y,const double z,const double t);
    static double zzmms_pl_gxy_t_source(const double y,const double z,const double t);
    static double zzmms_pl_gxz_t_source(const double y,const double z,const double t);


    // 1D
    static double zzmms_f1D(const double y);// helper function for uA
    static double zzmms_f_y1D(const double y);
    static double zzmms_f_yy1D(const double y);

    static double zzmms_uA1D(const double y,const double t);
    static double zzmms_uA_y1D(const double y,const double t);
    static double zzmms_uA_yy1D(const double y,const double t);
    static double zzmms_uA_z1D(const double y,const double t);
    static double zzmms_uA_zz1D(const double y,const double t);
    static double zzmms_uA_t1D(const double y,const double t);

    static double zzmms_mu1D(const double y);
    static double zzmms_mu_y1D(const double y);
    static double zzmms_mu_z1D(const double z);

    static double zzmms_sigmaxy1D(const double y,const double t);
    static double zzmms_uSource1D(const double y,const double t);

    static double zzmms_visc1D(const double y);
    static double zzmms_invVisc1D(const double y);
    static double zzmms_invVisc_y1D(const double y);
    static double zzmms_invVisc_z1D(const double y);

    static double zzmms_gxy1D(const double y,const double t);
    static double zzmms_gxy_y1D(const double y,const double t);
    static double zzmms_gxy_t1D(const double y,const double t);
    static double zzmms_gSource1D(const double y,const double t);

    static double zzmms_A1D(const double y);
    static double zzmms_B1D(const double y);
    static double zzmms_T1D(const double y);
    static double zzmms_n1D(const double y);
    static double zzmms_pl_sigmaxy1D(const double y,const double t);
    static double zzmms_pl_sigmaxz1D(const double y,const double t);
    static double zzmms_sdev1D(const double y,const double t);

    static double zzmms_pl_gSource1D(const double y,const double t);
    static double zzmms_pl_gxy_t_source1D(const double y,const double t);
    static double zzmms_pl_gxz_t_source1D(const double y,const double t);

};

#endif
