#ifndef FAULT_HPP_INCLUDED
#define FAULT_HPP_INCLUDED

#include <assert.h>
#include <cmath>
#include <petscksp.h>
#include <vector>

#include "domain.hpp"
#include "genFuncs.hpp"
#include "rootFinderContext.hpp"
#include "rootFinder.hpp"

class RootFinder;

using namespace std;

/*
 * Class containing the implementation of rate-and-state friction. The fault
 * is a line of length N.
 *
 * Allowed geometries:
 *   - properties are symmetric about the fault and only one side is solved for
 *   - one side of the fault is perfectly rigid, and only the other side is solved for
 *
 * State variable: psi
 * Possible state variable evolution laws:
 *   - regularized slip law
 *   - regularized aging law
 *   - regularized flash heating (based on the slip law)
 *
 * This class implements a version of the fault for quasidynamic simulations,
 * called Fault_qd, and a version for inertial simulations, called Fault_fd.
 *
 */


// base class for one-sided fault
class Fault
{
private:
  // disable default copy constructor and assignment operator
  Fault(const Fault & that);
  Fault& operator=( const Fault& rhs);

public:
  Domain      *_D; // shallow copy of domain
  const char  *_inputFile; // input file
  string       _delim; // format is: var delim value (without the white space)
  string       _inputDir;
  string       _outputDir; // directory for output
  string       _stateLaw; // state evolution law
  PetscScalar  _faultTypeScale; // = 2 if symmetric fault, 1 if one side of fault is rigid
  PetscInt     _limitSlipVel; // if 0 no ceiling, if yes then slipVel limited to <= vL

  // domain properties
  const PetscInt     _N;  //number of nodes on fault
  const PetscScalar  _L; // length of fault, grid spacing on fault
  Vec                _z; // vector of z-coordinates on fault (allows for variable grid spacing)

  Vec          _tauQSP,_tauP,_strength, _prestress; // shear stress: quasistatic,not qs,fault strength, prestress
  PetscScalar  _prestressScalar;
  Vec          _slip,_slipVel, _slip0; // slip, slip velocity, initial slip
  Vec          _psi; // state variable

  // for locking the fault
  vector<double>   _lockedVals,_lockedDepths;
  Vec              _locked;

  // rate-and-state parameters
  PetscScalar      _f0,_v0;
  vector<double>   _aVals,_aDepths,_bVals,_bDepths,_DcVals,_DcDepths;
  Vec              _a,_b,_Dc;
  vector<double>   _cohesionVals,_cohesionDepths,_rhoVals,_rhoDepths,_muVals,_muDepths;
  Vec              _cohesion,_mu,_rho;
  vector<double>   _sigmaNVals,_sigmaNDepths;
  vector<double>   _stateVals,_stateDepths; // initial conditions for state variable
  PetscScalar      _sigmaN_cap,_sigmaN_floor; // allow cap and floor on normal stress
  Vec              _sNEff; // effective normal stress
  Vec              _sN; // total normal stress

  // flash heating parameters
  string           _VwType; // constant or function_of_Tw
  vector<double>   _TwVals,_TwDepths,_VwVals,_VwDepths;
  PetscScalar      _fw,_tau_c,_D_fh;
  Vec              _T,_k,_c,_Vw,_Tw;

  // tolerances for linear and nonlinear (for vel) solve
  PetscScalar      _rootTol;
  PetscInt         _rootIts,_maxNumIts; // total number of iterations

  // viewers:
  // 1st string = key naming relevant field, e.g. "slip"
  // 2nd PetscViewer = PetscViewer object for file IO
  // 3rd string = full file path name for output
  //~ map <string,PetscViewer>  _viewers;
  PetscViewer _viewer_hdf5;

  // runtime data
  double   _computeVelTime,_stateLawTime, _scatterTime;

  // for mapping from body fields to the fault
  VecScatter* _body2fault;

  // iterators for _var
  typedef vector<Vec>::iterator it_vec;
  typedef vector<Vec>::const_iterator const_it_vec;

  Fault(Domain& D,VecScatter& scatter2fault, const int& faultTypeScale);
  virtual ~Fault();

  // load settings from input file
  PetscErrorCode loadSettings(const char *file);
  PetscErrorCode checkInput(); // check input from file
  PetscErrorCode loadFieldsFromFiles();
  PetscErrorCode setFields(Domain&D);
  PetscErrorCode setThermalFields(const Vec& T, const Vec& k, const Vec& c);
  PetscErrorCode updateTemperature(const Vec& T);
  PetscErrorCode setVecFromVectors(Vec&, vector<double>&,vector<double>&);
  PetscErrorCode setVecFromVectors(Vec& vec, vector<double>& vals,vector<double>& depths, const PetscScalar maxVal);

  PetscErrorCode setSNEff(const Vec& p); // update effective normal stress to reflect new pore pressure
  PetscErrorCode setSN(const Vec& p); // update effective normal stress to reflect new pore pressure
  PetscErrorCode imposeSlipVelCeiling(); // if desired, limit slipVel to less than vL

  // for steady state computations
  PetscErrorCode guessSS(const PetscScalar vL);
  PetscErrorCode computePsiSS(const PetscScalar vL);

  // IO
  PetscErrorCode virtual view(const double totRunTime);
  PetscErrorCode virtual writeContext(const string outputDir, PetscViewer& viewer);
  PetscErrorCode virtual writeStep(PetscViewer& viewer);

  // checkpointing
  PetscErrorCode virtual writeCheckpoint(PetscViewer& viewer);
  PetscErrorCode virtual loadCheckpoint();
};


// quasi-dynamic implementation of one-sided fault
class Fault_qd: public Fault
{
private:
  // disable default copy constructor and assignment operator
  Fault_qd(const Fault_qd & that);
  Fault_qd& operator=( const Fault_qd& rhs);

public:
  Vec _eta_rad; // radiation damping term

  Fault_qd(Domain& D,VecScatter& scatter2fault, const int& faultTypeScale);
  ~Fault_qd();

  PetscErrorCode loadSettings(const char *file);

  // for interaction with mediator
  PetscErrorCode initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx);
  PetscErrorCode updateFields(const PetscScalar time,const map<string,Vec>& varEx);
  PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);
  PetscErrorCode getResid(const PetscInt ind,const PetscScalar vel,PetscScalar* out);
  PetscErrorCode computeVel();
  PetscErrorCode updateStrength();
  PetscErrorCode updateTauP();
  PetscErrorCode writeContext(const string outputDir, PetscViewer& viewer);
  PetscErrorCode writeCheckpoint(PetscViewer& viewer);
  PetscErrorCode loadCheckpoint();
  PetscErrorCode loadCheckpointSS();
};


// fully dynamic implementation of one-sided fault
class Fault_fd: public Fault
{
private:
  // disable default copy constructor and assignment operator
  Fault_fd(const Fault_fd & that);
  Fault_fd& operator=( const Fault_fd& rhs);

public:
  Vec            _Phi, _an, _fricPen;
  Vec            _u,_uPrev,_d2u; // d2u = (Dyy+Dzz)*u evaluated on the fault
  PetscScalar    _deltaT;
  Vec            _alphay;
  Vec            _tau0; // dU0/dy (stress at end of qd period)

  PetscScalar    _tCenterTau, _tStdTau, _zCenterTau, _zStdTau, _ampTau;
  string         _timeMode;

  Fault_fd(Domain&, VecScatter& scatter2fault, const int& faultTypeScale);
  ~Fault_fd();

  PetscErrorCode loadSettings(const char *file);
  PetscErrorCode setFields();

  // for interaction with mediator
  PetscErrorCode initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx);
  PetscErrorCode updateFields(const PetscScalar time,const map<string,Vec>& varEx);
  PetscErrorCode d_dt(const PetscScalar time,const PetscScalar deltaT, map<string,Vec>& varNext, const map<string,Vec>& var, const map<string,Vec>& varPrev);
  PetscErrorCode getResid(const PetscInt ind,const PetscScalar vel,PetscScalar* out);
  PetscErrorCode computeVel();
  PetscErrorCode computeStateEvolution(Vec& psiNext, const Vec& psi, const Vec& psiPrev);
  PetscErrorCode setPhi(const PetscScalar _deltaT);
  PetscErrorCode updatePrestress(const PetscScalar currT);
  PetscErrorCode writeCheckpoint(PetscViewer& viewer);
  PetscErrorCode loadCheckpoint();
  PetscErrorCode loadCheckpointSS();
};


// structs for root-finding pieces

// computing the slip velocity for the quasi-dynamic problem
struct ComputeVel_qd : public RootFinderContext
{
  // shallow copies of contextual fields
  const PetscScalar  *_a, *_b, *_sN, *_tauQS, *_eta, *_psi,*_locked,*_Co;
  const PetscInt      _N; // length of the arrays
  const PetscScalar   _v0,_vL;

  // constructor
  ComputeVel_qd(const PetscInt N, const PetscScalar* eta,const PetscScalar* tauQS,const PetscScalar* sN,const PetscScalar* psi,const PetscScalar* a,const PetscScalar* b,const PetscScalar& v0,const PetscScalar& vL,const PetscScalar *locked,const PetscScalar *Co);

  // command to perform root-finding process, once contextual variables have been set
  PetscErrorCode computeVel(PetscScalar* slipVelA, const PetscScalar rootTol, PetscInt& rootIts, const PetscInt maxNumIts);

  // function that matches root finder template
  PetscErrorCode getResid(const PetscInt Jj,const PetscScalar vel,PetscScalar* out);
  PetscErrorCode getResid(const PetscInt Jj,const PetscScalar slipVel,PetscScalar *out,PetscScalar *J);
};


// computing the slip velocity for the dynamic problem
struct ComputeVel_fd : public RootFinderContext
{
  // shallow copies of contextual fields
  const PetscScalar  *_locked, *_Phi, *_an, *_psi, *_fricPen, *_a, *_sNEff;
  const PetscInt      _N; // length of the arrays
  const PetscScalar   _v0, _vL;

  // constructor
  ComputeVel_fd(const PetscScalar* locked, const PetscInt N,const PetscScalar* Phi, const PetscScalar* an, const PetscScalar* psi, const PetscScalar* fricPen,const PetscScalar* a,const PetscScalar* sneff, const PetscScalar v0, const PetscScalar vL);

  // command to perform root-finding process, once contextual variables have been set
  PetscErrorCode computeVel(PetscScalar* slipVelA, const PetscScalar rootTol, PetscInt& rootIts, const PetscInt maxNumIts);

  // function that matches root finder template
  PetscErrorCode getResid(const PetscInt Jj,const PetscScalar vel,PetscScalar* out);
  PetscErrorCode getResid(const PetscInt Jj,const PetscScalar slipVel,PetscScalar *out,PetscScalar *J);
};


// computing the aging law for the dynamic problem
struct ComputeAging_fd : public RootFinderContext
{
  // shallow copies of contextual fields
  const PetscScalar  *_Dc, *_b, *_slipVel, *_slipVelPrev,*_psi, *_psiPrev;
  PetscScalar        *_psiNext;
  const PetscInt      _N; // length of the arrays
  const PetscScalar   _v0, _deltaT, _f0;

  // constructor
  ComputeAging_fd(const PetscInt N,const PetscScalar* Dc, const PetscScalar* b, PetscScalar* psiNext, const PetscScalar* psi, const PetscScalar* psiPrev, const PetscScalar* slipVel, const PetscScalar v0, const PetscScalar deltaT, const PetscScalar f0);

  // command to perform root-finding process, once contextual variables have been set
  PetscErrorCode computeLaw(const PetscScalar rootTol, PetscInt& rootIts, const PetscInt maxNumIts);

  // function that matches root finder template
  PetscErrorCode getResid(const PetscInt Jj,const PetscScalar vel,PetscScalar* out);
  PetscErrorCode getResid(const PetscInt Jj,const PetscScalar slipVel,PetscScalar *out,PetscScalar *J);
};


// computing the slipLaw for the dynamic problem
struct ComputeSlipLaw_fd : public RootFinderContext
{
  // shallow copies of contextual fields
  const PetscScalar  *_Dc, *_a, *_b, *_slipVel, *_slipVelPrev,*_psi, *_psiPrev;
  PetscScalar        *_psiNext;
  const PetscInt      _N; // length of the arrays
  const PetscScalar   _v0, _deltaT, _f0;

  // constructor and destructor
  ComputeSlipLaw_fd(const PetscInt N,const PetscScalar* Dc, const PetscScalar* a,const PetscScalar* b, PetscScalar* psiNext, const PetscScalar* psi, const PetscScalar* psiPrev,const PetscScalar* slipVel, const PetscScalar v0, const PetscScalar deltaT, const PetscScalar f0);

  // command to perform root-finding process, once contextual variables have been set
  PetscErrorCode computeLaw(const PetscScalar rootTol, PetscInt& rootIts, const PetscInt maxNumIts);

  // function that matches root finder template
  PetscErrorCode getResid(const PetscInt Jj,const PetscScalar vel,PetscScalar* out);
  PetscErrorCode getResid(const PetscInt Jj,const PetscScalar slipVel,PetscScalar *out,PetscScalar *J);
};


// computing the flashHeating for the dynamic problem
struct ComputeFlashHeating_fd : public RootFinderContext
{
  // shallow copies of contextual fields
  const PetscScalar  *_Dc, *_a, *_b, *_slipVel, *_slipVelPrev, *_Vw,*_psi, *_psiPrev;
  PetscScalar        *_psiNext;
  const PetscInt      _N; // length of the arrays
  const PetscScalar   _v0, _deltaT, _f0, _fw;

  // constructor and destructor
  ComputeFlashHeating_fd(const PetscInt N,const PetscScalar* Dc, const PetscScalar* a, const PetscScalar* b, PetscScalar* psiNext, const PetscScalar* psi, const PetscScalar* psiPrev, const PetscScalar* slipVel, const PetscScalar* Vw,const PetscScalar v0, const PetscScalar deltaT,const PetscScalar f0, const PetscScalar fw);

  // command to perform root-finding process, once contextual variables have been set
  PetscErrorCode computeLaw(const PetscScalar rootTol, PetscInt& rootIts, const PetscInt maxNumIts);

  // function that matches root finder template
  PetscErrorCode getResid(const PetscInt Jj,const PetscScalar vel,PetscScalar* out);
  PetscErrorCode getResid(const PetscInt Jj,const PetscScalar slipVel,PetscScalar *out,PetscScalar *J);
};


// common rate-and-state functions

// state evolution law: aging law, state variable: psi
PetscScalar agingLaw_psi(const PetscScalar& psi, const PetscScalar& slipVel, const PetscScalar& b, const PetscScalar& f0, const PetscScalar& v0, const PetscScalar& Dc);

PetscErrorCode agingLaw_psi_Vec(Vec& dstate, const Vec& psi, const Vec& slipVel, const Vec& a, const Vec& b, const PetscScalar& f0, const PetscScalar& v0, const Vec& Dc);


// state evolution law: aging law, state variable: theta
PetscScalar agingLaw_theta(const PetscScalar& theta, const PetscScalar& slipVel, const PetscScalar& Dc);

PetscErrorCode agingLaw_theta_Vec(Vec& dstate, const Vec& theta, const Vec& slipVel, const Vec& Dc);


// state evolution law: slip law, state variable: psi
PetscScalar slipLaw_psi(const PetscScalar& psi, const PetscScalar& slipVel, const PetscScalar& a, const PetscScalar& b, const PetscScalar& f0, const PetscScalar& v0, const PetscScalar& Dc);

PetscErrorCode slipLaw_psi_Vec(Vec& dstate, const Vec& psi, const Vec& slipVel,const Vec& a, const Vec& b, const PetscScalar& f0, const PetscScalar& v0, const Vec& Dc);


// state evolution law: slip law, state variable: theta
PetscScalar slipLaw_theta(const PetscScalar& state, const PetscScalar& slipVel, const PetscScalar& Dc);

PetscErrorCode slipLaw_theta_Vec(Vec& dstate, const Vec& theta, const Vec& slipVel, const Vec& Dc);


// flash heating: compute Vw
PetscScalar flashHeating_Vw(const PetscScalar& T, const PetscScalar& rho, const PetscScalar& c, const PetscScalar& k, const PetscScalar& D, const PetscScalar& Tw, const PetscScalar& tau_c);


// flash heating: slip law, state variable: psi
PetscScalar flashHeating_psi(const PetscScalar& psi, const PetscScalar& slipVel, const PetscScalar& Vw, const PetscScalar& fw, const PetscScalar& Dc,const PetscScalar& a,const PetscScalar& b, const PetscScalar& f0, const PetscScalar& v0);

PetscErrorCode flashHeating_psi_Vec(Vec &dpsi,const Vec& psi, const Vec& slipVel, const Vec& T, const Vec& rho, const Vec& c, const Vec& k, Vec& Vw, const PetscScalar& D, const Vec& Tw, const PetscScalar& tau_c, const PetscScalar& fw, const Vec& Dc,const Vec& a,const Vec& b, const PetscScalar& f0, const PetscScalar& v0, const string _VwType);


// frictional strength, regularized form, for state variable psi
PetscErrorCode strength_psi_Vec(Vec& strength, const Vec& psi, const Vec& slipVel, const Vec& a,  const Vec& sN, const PetscScalar& v0);

PetscScalar strength_psi(const PetscScalar& sN, const PetscScalar& psi, const PetscScalar& slipVel, const PetscScalar& a, const PetscScalar& v0);


#endif
