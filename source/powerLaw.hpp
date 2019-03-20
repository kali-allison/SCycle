#ifndef POWERLAW_H_INCLUDED
#define POWERLAW_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <vector>
#include "genFuncs.hpp"
#include "domain.hpp"
#include "heatEquation.hpp"
#include "sbpOps.hpp"
#include "sbpOps_m_constGrid.hpp"
#include "sbpOps_m_varGrid.hpp"

using namespace std;

// computes effective viscosity for pseudoplasticity
// 1 / (effVisc) = (yield stress) / (inelastic strain rate)
class Pseudoplasticity
{
private:
  // disable default copy constructor and assignment operator
  Pseudoplasticity(const Pseudoplasticity &that);
  Pseudoplasticity& operator=(const Pseudoplasticity &rhs);

  // load settings and set material parameters
  vector<double>  _yieldStressVals,_yieldStressDepths; // define yield stress
  PetscErrorCode loadSettings(); // load settings from input file
  PetscErrorCode loadFieldsFromFiles();
  PetscErrorCode checkInput(); // check input from file
  PetscErrorCode setMaterialParameters();

public:
  const char     *_file;
  const string    _delim;
  string          _inputDir;
  const Vec      *_y,*_z;
  Vec             _yieldStress; // (MPa)
  Vec             _invEffVisc; // (GPa) eff. viscosity from plasticity

  Pseudoplasticity(const Vec& y, const Vec& z, const char *file, const string delim);
  ~Pseudoplasticity();
  PetscErrorCode guessInvEffVisc(const double dg);
  PetscErrorCode computeInvEffVisc(const Vec& dgdev);
  PetscErrorCode writeContext(const string outputDir);
};

// computes effective viscosity for dislocation creep
// 1 / (effVisc) = A exp(-B/T) sdev^n
class DislocationCreep
{
private:
  // disable default copy constructor and assignment operator
  DislocationCreep(const DislocationCreep &that);
  DislocationCreep& operator=(const DislocationCreep &rhs);

  // load settings and set material parameters
  vector<double>  _AVals,_ADepths,_nVals,_nDepths,_BVals,_BDepths;
  PetscErrorCode loadSettings(); // load settings from input file
  PetscErrorCode loadFieldsFromFiles();
  PetscErrorCode checkInput(); // check input from file
  PetscErrorCode setMaterialParameters();

public:
  const char     *_file;
  string          _delim;
  string          _inputDir;
  const Vec      *_y,*_z;
  Vec             _A,_n,_QR;
  Vec             _invEffVisc; // 1 / (effective viscosity)

  DislocationCreep(const Vec& y, const Vec& z, const char *file, const string delim);
  ~DislocationCreep();
  PetscErrorCode guessInvEffVisc(const Vec& Temp, const double dg);
  PetscErrorCode computeInvEffVisc(const Vec& Temp,const Vec& sdev);
  PetscErrorCode writeContext(const string outputDir);
};

// computes effective viscosity for diffusion creep
// 1 / (effVisc) = A exp(-B/T) sdev^n d^-m
class DiffusionCreep
{
private:
  // disable default copy constructor and assignment operator
  DiffusionCreep(const DiffusionCreep &that);
  DiffusionCreep& operator=(const DiffusionCreep &rhs);

  // load settings and set material parameters
  vector<double>  _AVals,_ADepths,_nVals,_nDepths,_BVals,_BDepths,_mVals,_mDepths;
  PetscErrorCode loadSettings(); // load settings from input file
  PetscErrorCode loadFieldsFromFiles();
  PetscErrorCode checkInput(); // check input from file
  PetscErrorCode setMaterialParameters();

public:
  const char     *_file;
  string          _delim;
  string          _inputDir;
  const Vec      *_y,*_z;
  Vec             _A,_n,_QR,_m;
  Vec             _invEffVisc; // 1 / (effective viscosity)

  DiffusionCreep(const Vec& y, const Vec& z, const char *file, const string delim);
  ~DiffusionCreep();
  PetscErrorCode guessInvEffVisc(const Vec& Temp,const double dg,const Vec& grainSize);
  PetscErrorCode computeInvEffVisc(const Vec& Temp,const Vec& sdev,const Vec& grainSize);
  PetscErrorCode writeContext(const string outputDir);
};


class PowerLaw
{
private:
  // disable default copy constructor and assignment operator
  PowerLaw(const PowerLaw &that);
  PowerLaw& operator=(const PowerLaw &rhs);

public:

  Domain         *_D;
  const char     *_file;
  string          _delim; // format is: var delim value (without the white space)
  string          _inputDir;
  string          _outputDir;  // directory to output data into

  const PetscInt  _order,_Ny,_Nz;
  PetscScalar     _Ly,_Lz;
  Vec            *_y,*_z; // to handle variable grid spacing
  const bool      _isMMS; // true if running mms test
  string          _wDiffCreep, _wDislCreep,_wPlasticity,_wLinearMaxwell;

  // deformation mechanisms
  Pseudoplasticity     *_plastic;
  DislocationCreep     *_disl;
  DiffusionCreep       *_diff;

  // material properties
  vector<double>   _muVals,_muDepths,_rhoVals,_rhoDepths,_TVals,_TDepths,_grainSizeVals,_grainSizeDepths;
  Vec              _mu, _rho, _cs,_effVisc;
  Vec              _T,_grainSize;
  vector<double>   _effViscVals_lm,_effViscDepths_lm; // linear Maxwell effective viscosity values
  PetscScalar      _effViscCap; // imposed upper limit on effective viscosity

  // displacement, strains, and strain rates
  Vec              _u,_surfDisp;
  Vec              _sxy,_sxz,_sdev; // sigma_xz (MPa), deviatoric stress (MPa)
  Vec              _gTxy,_gVxy,_dgVxy; // total strain, viscous strain, and viscoeus strain rate
  Vec              _gTxz,_gVxz,_dgVxz; // total strain, viscous strain, and viscoeus strain rate
  Vec              _dgVdev,_dgVdev_disl; // deviatoric strain and strain rate

  // linear system data
  string           _linSolver;
  string           _sbpType,_bcRType,_bcTType,_bcLType,_bcBType; // BC options: Neumann, Dirichlet
  Vec              _rhs,_bcT,_bcR,_bcB,_bcL,_bcRShift;
  KSP              _ksp;
  PC               _pc;
  PetscScalar      _kspTol;
  SbpOps          *_sbp;
  Mat              _B,_C; // composite matrices to make momentum balance simpler
  PetscErrorCode   initializeMomBalMats(); // computes B and C

  // for steady-state computations
  SbpOps          *_sbp_eta;
  KSP              _ksp_eta;
  PC               _pc_eta;
  PetscErrorCode   initializeSSMatrices(); // compute Bss and Css

  // runtime data
  double       _integrateTime,_writeTime,_linSolveTime,_factorTime,_startTime,_miscTime;
  PetscInt     _linSolveCount;

  // viewers and functions for file I/O
  PetscInt     _stepCount;

  // viewers:
  // 1st string = key naming relevant field, e.g. "slip"
  // 2nd PetscViewer = PetscViewer object for file IO
  // 3rd string = full file path name for output
  map <string,pair<PetscViewer,string> >  _viewers;
  PetscErrorCode writeDomain(const string outputDir);
  PetscErrorCode writeContext(const string outputDir);
  PetscErrorCode writeStep1D(PetscInt stepCount, string outputDir);
  PetscErrorCode writeStep2D(PetscInt stepCount, string outputDir);
  PetscErrorCode view(const double totRunTime);

  // constructor and destructor
  PowerLaw(Domain& D,string bcRType,string bcTType,string bcLType,string bcBType);
  ~PowerLaw();

  // initialize and set data
  PetscErrorCode loadSettings(const char *file); // load settings from input file
  PetscErrorCode checkInput(); // check input from file
  PetscErrorCode allocateFields(); // allocate space for member fields
  PetscErrorCode setMaterialParameters();
  PetscErrorCode loadFieldsFromFiles(); // load non-effective-viscosity parameters
  PetscErrorCode setUpSBPContext(Domain& D);
  PetscErrorCode setupKSP(KSP& ksp, PC& pc, Mat &A);


  // for steady state computations
  PetscErrorCode updateSSa(map<string,Vec>& varSS);
  PetscErrorCode updateSSb(map<string,Vec>& varSS,const PetscScalar time);
  PetscErrorCode guessSteadyStateEffVisc(const PetscScalar ess_t); // inititialize effective viscosity
  PetscErrorCode setSSRHS(map<string,Vec>& varSS,string bcRType,string bcTType,string bcLType,string bcBType);
  PetscErrorCode initializeSSMatrices(string bcRType,string bcTType,string bcLType,string bcBType);

  // methods for explicit time stepping
  PetscErrorCode initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx);
  PetscErrorCode updateFields(const PetscScalar time,const map<string,Vec>& varEx);
  PetscErrorCode updateTemperature(const Vec& T);
  PetscErrorCode updateGrainSize(const Vec& grainSize);
  PetscErrorCode computeMaxTimeStep(PetscScalar& maxTimeStep); // limited by Maxwell time
  PetscErrorCode computeViscStrainSourceTerms(Vec& source,Vec& gxy, Vec& gxz);
  PetscErrorCode computeViscStrainRates(const PetscScalar time,const Vec& gVxy,const Vec& gVxz,Vec& gVxy_t,Vec& gVxz_t);
  PetscErrorCode computeViscousStrainRateSAT(Vec &u, Vec &gL, Vec &gR, Vec &out);
  PetscErrorCode computeTotalStrains();
  PetscErrorCode computeStresses();
  PetscErrorCode computeSDev();
  PetscErrorCode computeDevViscStrainRates(); // deviatoric strains and strain rates
  PetscErrorCode computeViscosity(const PetscScalar viscCap);
  PetscErrorCode computeU();
  PetscErrorCode setRHS();
  PetscErrorCode changeBCTypes(string bcRTtype,string bcTTtype,string bcLTtype,string bcBTtype);
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
