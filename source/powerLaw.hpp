#ifndef POWERLAW_H_INCLUDED
#define POWERLAW_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <vector>
#include <petscviewerhdf5.h>
#include "genFuncs.hpp"
#include "domain.hpp"
#include "heatEquation.hpp"
#include "sbpOps.hpp"
#include "sbpOps_m_constGrid.hpp"
#include "sbpOps_m_varGrid.hpp"
#include "dislocationCreep.hpp"
#include "diffusionCreep.hpp"
#include "dissolutionPrecipitationCreep.hpp"
#include "pseudoplasticity.hpp"

using namespace std;


class PowerLaw
{
  private:
    // disable default copy constructor and assignment operator
    PowerLaw(const PowerLaw &that);
    PowerLaw& operator=(const PowerLaw &rhs);

  public:

    Domain              *_D;
    const char          *_file;
    std::string          _delim; // format is: var delim value (without the white space)
    std::string          _inputDir; // directory to load fields from
    std::string          _outputDir;  // directory to output data into

    const PetscInt       _order,_Ny,_Nz;
    PetscScalar          _Ly,_Lz;
    Vec                 *_y,*_z; // to handle variable grid spacing
    const bool           _isMMS; // true if running mms test
    std::string          _wPlasticity,_wDissPrecCreep,_wDislCreep,_wDiffCreep,_wLinearMaxwell;
    std::string          _wDislCreep2; // if supporting wet and dry dislocation creep


    // deformation mechanisms
    Pseudoplasticity                  *_plastic;
    DissolutionPrecipitationCreep     *_dp;
    DislocationCreep                  *_disl;
    DislocationCreep                  *_disl2; // if supporting wet and dry dislocation creep
    DiffusionCreep                    *_diff;

    // material properties
    std::vector<double>   _muVals,_muDepths,_rhoVals,_rhoDepths,_TVals,_TDepths,_grainSizeVals,_grainSizeDepths;
    Vec                   _mu, _rho, _cs,_effVisc;
    Vec                   _T,_grainSize;
    Vec                   _wetDist; // range 0-1, determines if enough water is present to activate DP or otherwise wet rheologies
    std::vector<double>   _effViscVals_lm,_effViscDepths_lm; // linear Maxwell effective viscosity values
    PetscScalar           _effViscCap; // imposed upper limit on effective viscosity

    // displacement, strains, and strain rates
    Vec                   _u,_surfDisp;
    Vec                   _sxy,_sxz,_sdev; // sigma_xz (MPa), deviatoric stress (MPa)
    Vec                   _gTxy,_gVxy,_dgVxy; // total strain, viscous strain, and viscous strain rate
    Vec                   _gTxz,_gVxz,_dgVxz; // total strain, viscous strain, and viscous strain rate
    Vec                   _dgVdev,_dgVdev_disl; // deviatoric strain rate

    // linear system data
    std::string           _linSolverSS,_linSolverTrans;
    std::string           _bcRType,_bcTType,_bcLType,_bcBType; // BC options: Neumann, Dirichlet
    Vec                   _rhs,_bcR,_bcT,_bcL,_bcB,_bcRShift,_bcTShift;
    KSP                   _ksp;
    PC                    _pc;
    PetscScalar           _kspTol;
    SbpOps               *_sbp;
    Mat                   _B,_C; // composite matrices to make momentum balance simpler
    PetscErrorCode        initializeMomBalMats(); // computes B and C

    // for steady-state computations
    SbpOps               *_sbp_eta;
    KSP                   _ksp_eta;
    PC                    _pc_eta;
    PetscErrorCode        initializeSSMatrices(); // compute Bss and Css

    // runtime data
    double       _integrateTime,_writeTime,_linSolveTime,_factorTime,_startTime,_miscTime;
    PetscInt     _linSolveCount;


    PowerLaw(Domain& D,std::string bcRType,std::string bcTType,std::string bcLType,std::string bcBType);
    ~PowerLaw();

    // initialize and set data
    PetscErrorCode loadSettings(const char *file); // load settings from input file
    PetscErrorCode checkInput(); // check input from file
    PetscErrorCode allocateFields(); // allocate space for member fields
    PetscErrorCode setMaterialParameters();
    PetscErrorCode loadFieldsFromFiles(); // load non-effective-viscosity parameters
    PetscErrorCode setUpSBPContext(Domain& D);
    PetscErrorCode setupKSP(KSP& ksp,PC& pc,Mat& A,std::string& linSolver);
    //~ PetscErrorCode setupKSP_SSIts(KSP& ksp,PC& pc,Mat& A);

    // IO
    PetscErrorCode writeDomain(const std::string outputDir);
    PetscErrorCode writeContext(const std::string outputDir, PetscViewer& viewer);
    PetscErrorCode writeStep1D(PetscViewer& viewer);
    PetscErrorCode writeStep2D(PetscViewer& viewer);
    PetscErrorCode writeCheckpoint(PetscViewer& viewer);
    PetscErrorCode loadCheckpoint();
    PetscErrorCode loadCheckpointSS();
    PetscErrorCode view(const double totRunTime);


    // for steady state computations
    PetscErrorCode updateSSa(map<string,Vec>& varSS);
    PetscErrorCode updateSSb(map<string,Vec>& varSS,const PetscScalar time);
    PetscErrorCode guessSteadyStateEffVisc(const PetscScalar ess_t); // inititialize effective viscosity
    PetscErrorCode setSSRHS(map<string,Vec>& varSS,std::string bcRType,std::string bcTType,std::string bcLType,std::string bcBType);
    PetscErrorCode initializeSSMatrices(std::string bcRType,std::string bcTType,std::string bcLType,std::string bcBType);


    // methods for explicit time stepping
    PetscErrorCode initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx);
    PetscErrorCode updateFields(const PetscScalar time,const map<string,Vec>& varEx);
    PetscErrorCode updateTemperature(const Vec& T);
    PetscErrorCode updateGrainSize(const Vec& grainSize);
    PetscErrorCode computeMaxTimeStep(PetscScalar& maxTimeStep); // limited by Maxwell time
    PetscErrorCode computeViscStrainSourceTerms(Vec& source);
    PetscErrorCode computeViscStrainRates(const PetscScalar time);
    PetscErrorCode computeViscousStrainRateSAT(Vec &u, Vec &gL, Vec &gR, Vec &out);
    PetscErrorCode computeTotalStrains();
    PetscErrorCode computeStresses();
    PetscErrorCode computeSDev();
    PetscErrorCode computeDevViscStrainRates(); // deviatoric strains and strain rates
    PetscErrorCode computeViscosity(const PetscScalar viscCap);
    PetscErrorCode computeU();
    PetscErrorCode setRHS();
    PetscErrorCode changeBCTypes(std::string bcRTtype,std::string bcTTtype,std::string bcLTtype,std::string bcBTtype);
    PetscErrorCode setSurfDisp();
    PetscErrorCode getStresses(Vec& sxy, Vec& sxz, Vec& sdev);





    // MMS functions
    PetscErrorCode setMMSInitialConditions(const double time);
    PetscErrorCode setMMSBoundaryConditions(const double time);
    PetscErrorCode measureMMSError(const PetscScalar time);
    PetscErrorCode forceMMSSolutions_u(const PetscScalar time);
    PetscErrorCode forceMMSSolutions_viscStrainRates(const PetscScalar time,Vec& gVxy_t,Vec& gVxz_t);
    PetscErrorCode addRHS_MMSSource(const PetscScalar time,Vec& rhs);
    PetscErrorCode addViscStrainRates_MMSSource(const PetscScalar time,Vec& gVxy_t,Vec& gVxz_t);

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
