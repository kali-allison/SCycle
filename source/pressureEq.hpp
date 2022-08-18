#ifndef PRESSURE_HPP_INCLUDED
#define PRESSURE_HPP_INCLUDED

#include <petscksp.h>
#include <cmath>
#include <assert.h>
#include <vector>
#include <cmath>
#include "genFuncs.hpp"
#include "domain.hpp"
#include "fault.hpp"
#include "sbpOps.hpp"
#include "sbpOps_m_constGrid.hpp"
#include "sbpOps_m_varGrid.hpp"
#include "integratorContextEx.hpp"
#include "integratorContextImex.hpp"

/* This class solves for the uncoupled fluid pressure during earthquake cycle
 * simulations, and solves for the permeability changes due to fault slip and
 * pore pressure. Results show a significant change of fluid pressure during
 * earthquake cycles, implying possible existence of overpressure and fault-
 * valve behavior in Earth's crust.
 */

using namespace std;

/* Class to solve for pressure evolution along the 1D vertical strike-slip fault
 * Currently is a stand-alone solver, and does not couple to stress evolution on the fault
 * Permeability can be slip-dependent, or pressure-dependent
 */

class PressureEq
{
public:
  Domain *_D; // shallow copy of domain
  Vec _p = NULL;     // pressure
  string _permSlipDependent, _permPressureDependent;

private:
  const char *_file;      // input file
  string      _delim;     // format is: var delim value (without the white space)
  string      _outputDir; // directory for output
  string      _inputDir;  // directory for input
  const bool  _isMMS;     // true if running mms test
  string      _hydraulicTimeIntType; // time integration type (explicit vs implicit)

  int         _guessSteadyStateICs;
  PetscScalar _initTime, _initDeltaT;

  // domain properties
  const int _order;
  const PetscInt _N;        //number of nodes on fault
  const PetscScalar _L, _h; // length of fault, grid spacing on fault
  Vec _z;                   // vector of z-coordinates on fault (allows for variable grid spacing)

  // material properties
  Vec _n_p = NULL, _beta_p = NULL, _k_p = NULL, _eta_p = NULL, _rho_f = NULL;
  Vec _k_slip = NULL, _k_press = NULL;
  Vec _kL_p = NULL, _kT_p = NULL, _kmin_p = NULL, _kmax_p = NULL;
  Vec _kmin2_p = NULL, _sigma_p = NULL;
  PetscScalar _g; // gravitational acceleration
  PetscScalar _vL;
  PetscScalar _bcB_ratio;
  string _bcB_type;
  int _maxBeIteration;
  double _minBeDifference;

  // linear system
  string _linSolver;
  KSP _ksp;
  PetscScalar _kspTol;
  SbpOps *_sbp;
  int _linSolveCount;
  Vec _bcL = NULL, _bcT = NULL, _bcB = NULL, _bcB_gravity = NULL, _bcB_impose = NULL;
  Vec _p_t = NULL;

  // input fields
  vector<double> _n_pVals, _n_pDepths, _beta_pVals, _beta_pDepths, _k_pVals, _k_pDepths;
  vector<double> _eta_pVals, _eta_pDepths, _rho_fVals, _rho_fDepths;
  vector<double> _pVals, _pDepths, _dpVals, _dpDepths;
  vector<double> _kL_pVals, _kL_pDepths, _kT_pVals, _kT_pDepths, _kmin_pVals, _kmin_pDepths, _kmax_pVals, _kmax_pDepths;
  vector<double> _kmin2_pVals, _kmin2_pDepths, _sigma_pVals, _sigma_pDepths;
  vector<double> _sigmaNVals,_sigmaNDepths;
  Vec            _sN; // total normal stress

  VecScatter _scatters;

  // run time monitoring
  double _writeTime, _linSolveTime, _ptTime, _startTime, _miscTime;
  double _invTime;


  // viewers:
  // 1st string = key naming relevant field, e.g. "slip"
  // 2nd PetscViewer = PetscViewer object for file IO
  // 3rd string = full file path name for output
  map<string, pair<PetscViewer, string>> _viewers;

  // disable default copy constructor and assignment operator
  PressureEq(const PressureEq &that);
  PressureEq &operator=(const PressureEq &rhs);

  // private member functions
  PetscErrorCode computeVariableCoefficient(Vec &coeff);
  PetscErrorCode updateBoundaryCoefficient(const Vec &coeff);
  PetscErrorCode setUpSBP();
  PetscErrorCode computeInitialSteadyStatePressure(Domain &D);
  PetscErrorCode setUpBe(Domain &D);
  PetscErrorCode setupKSP(const Mat &A);
  PetscErrorCode updatePermPressureDependent();


  // constructor and destructor
public:
  PressureEq(Domain &D);
  ~PressureEq();

  // public member functions
  PetscErrorCode getPressure(Vec& P);
  PetscErrorCode setPressure(const Vec& P);
  PetscErrorCode getPermeability(Vec& K);
  PetscErrorCode setPremeability(const Vec& K);

  PetscErrorCode setFields(Domain &D);
  PetscErrorCode loadSettings(const char *file);
  PetscErrorCode checkInput();

  PetscErrorCode initiateIntegrand(const PetscScalar time, map<string, Vec> &varEx, map<string, Vec> &varIm);
  PetscErrorCode updateFields(const PetscScalar time, const map<string, Vec> &varEx);
  PetscErrorCode updateFields(const PetscScalar time, const map<string, Vec> &varEx, const map<string, Vec> &varIm);
  PetscErrorCode loadFieldsFromFiles();

  // ============ explicit time integration =======================
  PetscErrorCode d_dt(const PetscScalar time, const map<string, Vec> &varEx, map<string, Vec> &dvarEx);
  // time derivative of pressure
  PetscErrorCode dp_dt(const PetscScalar time, const map<string, Vec> &varEx, map<string, Vec> &dvarEx);
  PetscErrorCode dp_dt(const PetscScalar time, const Vec& P, Vec& dPdt);
  PetscErrorCode d_dt_mms(const PetscScalar time, const map<string, Vec> &varEx, map<string, Vec> &dvarEx);


  // ============= implicit time integration ======================
  PetscErrorCode d_dt(const PetscScalar time, const map<string, Vec> &varEx, map<string, Vec> &dvarEx, map<string, Vec> &varIm, const map<string, Vec> &varImo, const PetscScalar dt);
  // backward Euler
  PetscErrorCode be(const PetscScalar time, const map<string, Vec> &varEx, map<string, Vec> &dvarEx, map<string, Vec> &varIm, const map<string, Vec> &varImo, const PetscScalar dt);
  // time derivative of permeability
  PetscErrorCode dk_dt(const PetscScalar time, const map<string, Vec> &varEx, map<string, Vec> &dvarEx);
  PetscErrorCode dk_dt(const PetscScalar time, const Vec slipVel, const Vec &K, Vec &dKdt);

  // MMS test for backward Euler
  PetscErrorCode be_mms(const PetscScalar time, const map<string, Vec> &varEx, map<string, Vec> &dvarEx, map<string, Vec> &varIm, const map<string, Vec> &varImo, const PetscScalar dt);

  // IO
  PetscErrorCode view(const double totRunTime);
  PetscErrorCode writeContext(const string outputDir, PetscViewer& viewer);
  PetscErrorCode writeStep(PetscViewer& viewer);
  PetscErrorCode writeCheckpoint(PetscViewer& viewer);
  PetscErrorCode loadCheckpoint();

  // MMS error
  PetscErrorCode measureMMSError(const double totRunTime);
  static double zzmms_pSource1D(const double z, const double t);
  static double zzmms_pA1D(const double y, const double t);
  static double zzmms_pt1D(const double z, const double t);
};

#endif
