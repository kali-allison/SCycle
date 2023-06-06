#ifndef HEATEQUATION_H_INCLUDED
#define HEATEQUATION_H_INCLUDED

#include <petscksp.h>
#include <string>
#include <cmath>
#include <assert.h>
#include <vector>
#include "genFuncs.hpp"
#include "domain.hpp"
#include "sbpOps.hpp"
#include "sbpOps_m_constGrid.hpp"
#include "sbpOps_m_varGrid.hpp"
#include "integratorContextEx.hpp"
#include "integratorContextImex.hpp"
#include "odeSolver.hpp"
#include "odeSolverImex.hpp"

using namespace std;

/*
 * Class implementing the heat equation.
 *
 * Possible forms:
 *   - steady state (dT/dt = 0)
 *   - transient (includes dT/dt term)
 *
 * Possible algorithms for integration in time:
 *   - Backward Euler (see functions whose name begins with "be")
 *   - Fully explicit time stepping (see functions whose name begins with "d_dt")
 *
 * Possible source terms:
 *   - viscous shear heating
 *   - frictional shear heating (either from a plane or a finite width shear zone)
 *   - radioactive decay
 */

class HeatEquation
{
private:
  // disable default copy constructor and assignment operator
  HeatEquation(const HeatEquation &that);
  HeatEquation& operator=(const HeatEquation &rhs);

public:

  // domain dimensions etc
  Domain              *_D;
  const PetscInt       _order,_Ny,_Nz;
  PetscInt             _Nz_lab; // # of points to LAB depth (must be <= Nz)
  const PetscScalar    _Ly,_Lz,_dy,_dz;
  PetscScalar          _Lz_lab; // depth of LAB (must be <= Lz)
  Vec                 *_y,*_z; // to handle variable grid spacing
  string               _heatEquationType;
  int                  _isMMS;
  int                  _loadICs;
  PetscScalar          _initTime,_initDeltaT;

  // IO information
  const char  *_file; // input ASCII file location
  string       _inputDir;
  string       _outputDir; // output file location
  string       _delim; // format is: var delim value (without the white space)

  // material parameters
  vector<double>  _rhoVals,_rhoDepths,_kVals,_kDepths,_cVals,_cDepths,_TVals,_TDepths;

  // heat fluxes
  Vec  _kTz_z0,_kTz; // surface and total heat flux

  // max temperature for writing out
  PetscScalar      _maxdT;
  Vec              _maxdTVec;

  // boundary conditions for steady-state (ss) and transient (trans) problems
  string _bcRType_ss,_bcTType_ss,_bcLType_ss,_bcBType_ss; // options: Dirichlet, Neumann
  string _bcRType_trans,_bcTType_trans,_bcLType_trans,_bcBType_trans; // options: Dirichlet, Neumann

  // viewers for 1D and 2D fields
  // 1st string = key naming relevant field, e.g. "slip"
  // 2nd PetscViewer = PetscViewer object for file IO
  // 3rd string = full file path name for output
  //~ map <string,pair<PetscViewer,string> >  _viewers2D;
  //~ PetscViewer                             _viewer1D_hdf5;

  // which factors to include: viscous and frictional shear heating, and radioactive heat generation
  string          _wViscShearHeating,_wFrictionalHeating,_wRadioHeatGen;

  // linear system data
  string          _sbpType;
  SbpOps*         _sbp;
  Vec             _bcR,_bcT,_bcL,_bcB; // boundary conditions when solving for dT
  string          _linSolver;
  PetscScalar     _kspTol;
  KSP             _kspSS,_kspTrans; // KSPs for steady state and transient problems
  PC              _pc;
  Mat             _I,_rcInv,_B,_pcMat; // intermediates for Backward Euler
  Mat             _D2ath;

  // scatters to take values from body field(s) to 1D fields
  // naming convention for key (string): body2<boundary>, example: "body2L>"
  map <string, VecScatter>  _scatters;

  // finite width shear zone
  Mat             _MapV; // maps slip velocity to full size vector for scaling Gw
  Vec             _Gw; // Green's function for shear heating, frictional heat
  Vec             _w; // width of shear zone (km)
  vector<double>  _wVals,_wDepths;
  PetscScalar     _wMax;

  // radiactive heat generation parameters
  vector<double>  _A0Vals,_A0Depths; // (kW/m^3) heat generation at z=0
  double          _Lrad; // (km) decay length scale

  // runtime data
  double          _linSolveTime,_factorTime,_beTime,_writeTime,_miscTime;
  PetscInt        _linSolveCount;

  // checkpoint settings
  //~ PetscInt _ckpt, _ckptNumber;


  // load settings from input file
  PetscErrorCode loadSettings(const char *file);
  PetscErrorCode allocateFields();
  PetscErrorCode setFields();
  PetscErrorCode loadFieldsFromFiles();
  PetscErrorCode checkInput();

  PetscErrorCode constructScatters(Vec& T, Vec& T_l);
  PetscErrorCode constructMapV();
  PetscErrorCode computeInitialSteadyStateTemp();
  PetscErrorCode setUpSteadyStateProblem();
  PetscErrorCode setUpTransientProblem();
  PetscErrorCode computeViscousShearHeating(const Vec& sdev, const Vec& dgdev);
  PetscErrorCode computeFrictionalShearHeating(const Vec& tau, const Vec& slipVel);
  PetscErrorCode setupKSP(Mat& A);
  PetscErrorCode setupKSP_SS(Mat& A);
  PetscErrorCode computeHeatFlux();

  Vec _Tamb,_dT,_T; // full domain: ambient temperature, change in temperature from ambiant, and total temperature
  Vec _k,_rho,_c; // thermal conductivity, density, heat capacity,
  Vec _Qrad,_Qfric,_Qvisc,_Q; // source terms: radioactive decay, frictional, viscous, total heat generation

  // constructor and destructor
  HeatEquation(Domain& D);
  ~HeatEquation();

  PetscErrorCode getTemp(Vec& T); // return total temperature
  PetscErrorCode setTemp(const Vec& T); // set temperature

  PetscErrorCode computeSteadyStateTemp(const PetscScalar time,const Vec slipVel,const Vec& tau, const Vec& sigmadev, const Vec& dgdev,Vec& T);
  PetscErrorCode initiateVarSS(map<string,Vec>& varSS);


  // compute rate
  PetscErrorCode initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& _varIm);
  PetscErrorCode updateFields(const PetscScalar time,const map<string,Vec>& varEx,const map<string,Vec>& varIm);


  // implicitly solve for temperature using backward Euler
  PetscErrorCode be(const PetscScalar time,const Vec slipVel,const Vec& tau, const Vec& sdev, const Vec& dgdev,Vec& T,const Vec& To,const PetscScalar dt);
  PetscErrorCode be_transient(const PetscScalar time,const Vec slipVel,const Vec& tau, const Vec& sdev, const Vec& dgdev,Vec& T,const Vec& To,const PetscScalar dt);
  PetscErrorCode be_steadyState(const PetscScalar time,const Vec slipVel,const Vec& tau, const Vec& sdev, const Vec& dgdev,Vec& T,const Vec& To,const PetscScalar dt);
  PetscErrorCode be_steadyStateMMS(const PetscScalar time,const Vec slipVel,const Vec& tau, const Vec& sigmadev, const Vec& dgdev,Vec& T,const Vec& To,const PetscScalar dt);

  PetscErrorCode d_dt_mms(const PetscScalar time,const Vec& T, Vec& dTdt);
  PetscErrorCode d_dt(const PetscScalar time,const Vec slipVel,const Vec& tau, const Vec& sdev, const Vec& dgdev, const Vec& T, Vec& dTdt);


  // IO commands
  PetscErrorCode view();
  PetscErrorCode writeDomain(const string outputDir);
  PetscErrorCode writeContext(const string outputDir, PetscViewer& viewer);
  PetscErrorCode writeStep1D(PetscViewer& viewer);
  PetscErrorCode writeStep2D(PetscViewer& viewer);
  PetscErrorCode writeCheckpoint(PetscViewer& viewer);
  PetscErrorCode loadCheckpoint();
  PetscErrorCode loadCheckpointSS();


  // MMS functions
  PetscErrorCode measureMMSError(const PetscScalar time);
  PetscErrorCode setMMSBoundaryConditions(const double time, string bcRType,string bcTType,string bcLType,string bcBType);

  static double zzmms_rho(const double y,const double z);
  static double zzmms_c(const double y,const double z);
  static double zzmms_h(const double y,const double z);

  static double zzmms_k(const double y,const double z);
  static double zzmms_k_y(const double y,const double z);
  static double zzmms_k_z(const double y,const double z);


  static double zzmms_f(const double y,const double z);
  static double zzmms_f_y(const double y,const double z);
  static double zzmms_f_yy(const double y,const double z);
  static double zzmms_f_z(const double y,const double z);
  static double zzmms_f_zz(const double y,const double z);

  static double zzmms_g(const double t);
  static double zzmms_g_t(const double t);
  static double zzmms_T(const double y,const double z,const double t);
  static double zzmms_T_y(const double y,const double z,const double t);
  static double zzmms_T_yy(const double y,const double z,const double t);
  static double zzmms_T_z(const double y,const double z,const double t);
  static double zzmms_T_zz(const double y,const double z,const double t);
  static double zzmms_T_t(const double y,const double z,const double t);
  static double zzmms_SSTsource(const double y,const double z,const double t);

  static double zzmms_dT(const double y,const double z,const double t);
  static double zzmms_dT_y(const double y,const double z,const double t);
  static double zzmms_dT_yy(const double y,const double z,const double t);
  static double zzmms_dT_z(const double y,const double z,const double t);
  static double zzmms_dT_zz(const double y,const double z,const double t);
  static double zzmms_dT_t(const double y,const double z,const double t);
  static double zzmms_SSdTsource(const double y,const double z,const double t);
};

#endif
