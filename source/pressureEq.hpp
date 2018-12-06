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
#include "sbpOps_c.hpp"
#include "sbpOps_fc.hpp"
#include "sbpOps_fc_coordTrans.hpp"
#include "integratorContextEx.hpp"
#include "integratorContextImex.hpp"

class PressureEq
{
private:
  const char *_file;      // input file
  std::string _delim;     // format is: var delim value (without the white space)
  std::string _outputDir; // directory for output
  std::string _inputDir;  // directory for input
  const bool _isMMS;      // true if running mms test
  // const bool        _nonlinear; // true if paramters are nolinear
  std::string _hydraulicTimeIntType; // time integration type (explicit vs implicit)

  int _guessSteadyStateICs;
  PetscScalar _initTime, _initDeltaT;

  // domain properties
  const int _order;
  const PetscInt _N;        //number of nodes on fault
  const PetscScalar _L, _h; // length of fault, grid spacing on fault
  Vec _z;                   // vector of z-coordinates on fault (allows for variable grid spacing)

  // material properties
  Vec _n_p, _beta_p, _k_p, _eta_p, _rho_f;
  Vec _kL_p, _kT_p, _kmin_p, _kmax_p;
  Vec _kmin2_p, _pstd_p;
  // Vec _meanK_p, _deltaK_p, _meanP_p, _stdP_p;
  PetscScalar _g; // gravitational acceleration
  PetscScalar _bcB_ratio;
  int _maxBeIteration;
  double _minBeDifference;

  // linear system
  std::string _linSolver;
  KSP _ksp;
  PetscScalar _kspTol;
  SbpOps *_sbp;
  std::string _sbpType;
  int _linSolveCount;
  Vec _bcL, _bcT, _bcB;
  Vec _p_t;

  // input fields
  std::vector<double> _n_pVals, _n_pDepths, _beta_pVals, _beta_pDepths, _k_pVals, _k_pDepths;
  std::vector<double> _eta_pVals, _eta_pDepths, _rho_fVals, _rho_fDepths;
  std::vector<double> _pVals, _pDepths, _dpVals, _dpDepths;
  std::vector<double> _kL_pVals, _kL_pDepths, _kT_pVals, _kT_pDepths, _kmin_pVals, _kmin_pDepths, _kmax_pVals, _kmax_pDepths;
  // std::vector<double> _meanK_pVals, _meanK_pDepths, _deltaK_pVals, _deltaK_pDepths, _meanP_pVals, _meanP_pDepths, _stdP_pVals, _stdP_pDepths;
  std::vector<double> _kmin2_pVals, _kmin2_pDepths, _pstd_pVals, _pstd_pDepths;
  std::vector<double>   _sigmaNVals,_sigmaNDepths;
  Vec                   _sN; // total normal stress

  // run time monitoring
  double _writeTime, _linSolveTime, _ptTime, _startTime, _miscTime;
  double _invTime;

  // // viewers
  // std::map <string,PetscViewer>  _viewers;

  // viewers:
  // 1st string = key naming relevant field, e.g. "slip"
  // 2nd PetscViewer = PetscViewer object for file IO
  // 3rd string = full file path name for output
  //~ std::map <string,PetscViewer>  _viewers;
  std::map<string, std::pair<PetscViewer, string>> _viewers;

  // disable default copy constructor and assignment operator
  PressureEq(const PressureEq &that);
  PressureEq &operator=(const PressureEq &rhs);

  PetscErrorCode setVecFromVectors(Vec &, vector<double> &, vector<double> &);
  PetscErrorCode setVecFromVectors(Vec &vec, vector<double> &vals, vector<double> &depths,
                                   const PetscScalar maxVal);

  PetscErrorCode computeVariableCoefficient(const Vec &p, Vec &coeff);
  PetscErrorCode setUpSBP();
  PetscErrorCode computeInitialSteadyStatePressure(Domain &D);
  PetscErrorCode setUpBe(Domain &D);
  PetscErrorCode setupKSP(const Mat &A);
  PetscErrorCode updatePermPressureDependent();

public:
  Domain *_D; // shallow copy of domain
  Vec _p;     // pressure

  std::string _permSlipDependent, _permPressureDependent;

  PressureEq(Domain &D);
  ~PressureEq();

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

  // explicit time integration
  PetscErrorCode d_dt(const PetscScalar time, const map<string, Vec> &varEx, map<string, Vec> &dvarEx);
  PetscErrorCode dp_dt(const PetscScalar time, const map<string, Vec> &varEx, map<string, Vec> &dvarEx);
  PetscErrorCode d_dt_mms(const PetscScalar time, const map<string, Vec> &varEx, map<string, Vec> &dvarEx);
  PetscErrorCode dp_dt(const PetscScalar time, const Vec& P, Vec& dPdt);
  // implicit time integration
  PetscErrorCode d_dt(const PetscScalar time, const map<string, Vec> &varEx, map<string, Vec> &dvarEx,
                      map<string, Vec> &varIm, const map<string, Vec> &varImo, const PetscScalar dt);
  PetscErrorCode be(const PetscScalar time, const map<string, Vec> &varEx, map<string, Vec> &dvarEx,
                    map<string, Vec> &varIm, const map<string, Vec> &varImo, const PetscScalar dt);
  PetscErrorCode be_mms(const PetscScalar time, const map<string, Vec> &varEx, map<string, Vec> &dvarEx,
                        map<string, Vec> &varIm, const map<string, Vec> &varImo, const PetscScalar dt);

  PetscErrorCode dk_dt(const PetscScalar time, const map<string, Vec> &varEx, map<string, Vec> &dvarEx);
  PetscErrorCode dk_dt(const PetscScalar time, const Vec slipVel, const Vec &K, Vec &dKdt);
  // IO
  PetscErrorCode view(const double totRunTime);
  PetscErrorCode writeContext(const std::string outputDir);
  PetscErrorCode writeStep(const PetscInt stepCount, const PetscScalar time);
  PetscErrorCode writeStep(const PetscInt stepCount, const PetscScalar time, const std::string outputDir);
  //~ PetscErrorCode view();

  // mms error
  PetscErrorCode measureMMSError(const double totRunTime);

  static double zzmms_pSource1D(const double z, const double t);
  static double zzmms_pA1D(const double y, const double t);
  static double zzmms_pt1D(const double z, const double t);
};

#endif
