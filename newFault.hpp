#ifndef NEWFAULT_HPP_INCLUDED
#define NEWFAULT_HPP_INCLUDED

#include <petscksp.h>
#include <cmath>
#include <assert.h>
#include <vector>
#include <cmath>
#include "genFuncs.hpp"
#include "domain.hpp"
#include "rootFinderContext.hpp"

class RootFinder;


// base class for one-sided fault
class NewFault
{
  private:
    // disable default copy constructor and assignment operator
    NewFault(const NewFault & that);
    NewFault& operator=( const NewFault& rhs);

  public:
    Domain           *_D; // shallow copy of domain
    const char       *_inputFile; // input file
    std::string       _delim; // format is: var delim value (without the white space)
    std::string       _outputDir; // directory for output
    std::string       _stateLaw; // state evolution law

    // domain properties
    const PetscInt     _N;  //number of nodes on fault
    const PetscScalar  _L; // length of fault, grid spacing on fault
    Vec                _z; // vector of z-coordinates on fault (allows for variable grid spacing)

    Vec            _tauQSP;
    Vec            _tauP; // not quasi-static

    // rate-and-state parameters
    PetscScalar           _f0,_v0;
    std::vector<double>   _aVals,_aDepths,_bVals,_bDepths,_DcVals,_DcDepths;
    Vec                   _a,_b,_Dc;
    std::vector<double>   _cohesionVals,_cohesionDepths,_rhoVals,_rhoDepths,_muVals,_muDepths;
    Vec                   _cohesion,_mu,_rho;
    Vec                   _dPsi,_psi;
    std::vector<double>   _sigmaNVals,_sigmaNDepths;
    PetscScalar           _sigmaN_cap,_sigmaN_floor; // allow cap and floor on normal stress
    Vec                   _sNEff; // effective normal stress
    Vec                   _sN; // total normal stress
    Vec                   _slip,_slipVel;

    PetscScalar           _fw,_Vw,_tau_c,_Tw,_D_fh; // flash heating parameters
    Vec                   _T,_k,_c; // for flash heating

    // tolerances for linear and nonlinear (for vel) solve
    PetscScalar    _rootTol;
    PetscInt       _rootIts,_maxNumIts; // total number of iterations

    // viewers:
    // 1st string = key naming relevant field, e.g. "slip"
    // 2nd PetscViewer = PetscViewer object for file IO
    // 3rd string = full file path name for output
    //~ std::map <string,PetscViewer>  _viewers;
    std::map <string,std::pair<PetscViewer,string> >  _viewers;

    // runtime data
    double               _computeVelTime,_stateLawTime;

    // for mapping from body fields to the fault
    VecScatter* _body2fault;

    // iterators for _var
    typedef std::vector<Vec>::iterator it_vec;
    typedef std::vector<Vec>::const_iterator const_it_vec;


    NewFault(Domain& D,VecScatter& scatter2fault);
    virtual ~NewFault();

    // load settings from input file
    PetscErrorCode loadSettings(const char *file);
    PetscErrorCode checkInput(); // check input from file
    PetscErrorCode loadFieldsFromFiles(std::string inputDir);
    PetscErrorCode setFields(Domain&D);
    PetscErrorCode setThermalFields(const Vec& T, const Vec& k, const Vec& c);
    PetscErrorCode updateTemperature(const Vec& T);
    PetscErrorCode setVecFromVectors(Vec&, vector<double>&,vector<double>&);
    PetscErrorCode setVecFromVectors(Vec& vec, vector<double>& vals,vector<double>& depths,
      const PetscScalar maxVal);

    PetscErrorCode setTauQS(const Vec& sxy);
    PetscErrorCode setSNEff(const Vec& p); // update effective normal stress to reflect new pore pressure
    PetscErrorCode setSN(const Vec& p); // update effective normal stress to reflect new pore pressure

    // for steady state computations
    PetscErrorCode computeTauRS(Vec& tauRS, const PetscScalar vL);

    // IO
    PetscErrorCode virtual view(const double totRunTime);
    PetscErrorCode virtual writeContext(const std::string outputDir);
    PetscErrorCode writeStep(const PetscInt stepCount, const PetscScalar time);
    PetscErrorCode virtual writeStep(const PetscInt stepCount, const PetscScalar time, const std::string outputDir);
};



// quasi-dynamic implementation of one-sided fault
class NewFault_qd: public NewFault
{
  private:

    // disable default copy constructor and assignment operator
    NewFault_qd(const NewFault_qd & that);
    NewFault_qd& operator=( const NewFault_qd& rhs);

  public:

    Vec _eta_rad; // radiation damping term

    NewFault_qd(Domain& D,VecScatter& scatter2fault);
    ~NewFault_qd();

    PetscErrorCode loadSettings(const char *file);

    // for interaction with mediator
    PetscErrorCode initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx);
    PetscErrorCode updateFields(const PetscScalar time,const map<string,Vec>& varEx);
    PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);

    PetscErrorCode getResid(const PetscInt ind,const PetscScalar vel,PetscScalar* out);
    PetscErrorCode computeVel();

    PetscErrorCode writeContext(const std::string outputDir);
};




// fully dynamic implementation of one-sided fault
class NewFault_dyn: public NewFault
{
  private:

    // disable default copy constructor and assignment operator
    NewFault_dyn(const NewFault_dyn & that);
    NewFault_dyn& operator=( const NewFault_dyn& rhs);

  public:
   Vec                  _Phi, _an, _constraints_factor;
    Vec                 _slipPrev;
    Vec                 _rhoLocal;
    IS                  _is;
    PetscScalar         _deltaT;

    PetscScalar         _alphay, _alphaz;

    PetscScalar           _tCenterTau, _tStdTau, _zCenterTau, _zStdTau, _ampTau;

    NewFault_dyn(Domain&, VecScatter& scatter2fault);
    ~NewFault_dyn();

    PetscErrorCode loadSettings(const char *file);

    // for interaction with mediator
    PetscErrorCode initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx);
    PetscErrorCode initiateIntegrand_dyn(map<string,Vec>& varEx, Vec _rhoVec);
    PetscErrorCode updateFields(const PetscScalar time,const map<string,Vec>& varEx);
    PetscErrorCode d_dt(const PetscScalar time, map<string,Vec>& varEx,map<string,Vec>& dvarEx,PetscScalar _deltaT);

    PetscErrorCode getResid(const PetscInt ind,const PetscScalar vel,PetscScalar* out);
    PetscErrorCode computeVel();
    PetscErrorCode computeAgingLaw();
    PetscErrorCode setPhi(map<string,Vec>& varEx, map<string,Vec>& dvarEx, const PetscScalar _deltaT);
    PetscErrorCode updateTau(const PetscScalar currT);
};


// structs for root-finding pieces

// computing the slip velocity for the quasi-dynamic problem
struct ComputeVel_qd : public RootFinderContext
{
  // shallow copies of contextual fields
  const PetscScalar  *_a, *_b, *_sN, *_tauQS, *_eta, *_psi;
  const PetscInt      _N; // length of the arrays
  const PetscScalar   _v0;

  // constructor and destructor
  ComputeVel_qd(const PetscInt N, const PetscScalar* eta,const PetscScalar* tauQS,const PetscScalar* sN,const PetscScalar* psi,const PetscScalar* a,const PetscScalar* b,const PetscScalar& v0);
  //~ ~ComputeVel_qd(); // use default destructor, as this class consists entirely of shallow copies

  // command to perform root-finding process, once contextual variables have been set
  PetscErrorCode computeVel(PetscScalar* slipVelA, const PetscScalar rootTol, PetscInt& rootIts, const PetscInt maxNumIts);

  // function that matches root finder template
  PetscErrorCode getResid(const PetscInt Jj,const PetscScalar vel,PetscScalar* out);
  PetscErrorCode getResid(const PetscInt Jj,const PetscScalar slipVel,PetscScalar *out,PetscScalar *J);
};


// computing the slip velocity for the dynamic problem
struct ComputeVel_dyn : public RootFinderContext
{
  // shallow copies of contextual fields
  const PetscScalar  *_Phi, *_an, *_psi, *_constraints_factor, *_a, *_sNEff;
  const PetscInt      _N; // length of the arrays
  const PetscScalar   _v0;

  // constructor and destructor
  ComputeVel_dyn(const PetscInt N,const PetscScalar* Phi, const PetscScalar* an, const PetscScalar* psi, const PetscScalar* constraints_factor,const PetscScalar* a,const PetscScalar* sneff, const PetscScalar v0);
  //~ ~ComputeVel_qd(); // use default destructor, as this class consists entirely of shallow copies

  // command to perform root-finding process, once contextual variables have been set
  PetscErrorCode computeVel(PetscScalar* slipVelA, const PetscScalar rootTol, PetscInt& rootIts, const PetscInt maxNumIts);

  // function that matches root finder template
  PetscErrorCode getResid(const PetscInt Jj,const PetscScalar vel,PetscScalar* out);
  PetscErrorCode getResid(const PetscInt Jj,const PetscScalar slipVel,PetscScalar *out,PetscScalar *J){return 1;};
};

// computing the slip velocity for the dynamic problem
struct ComputeAging_dyn : public RootFinderContext
{
  // shallow copies of contextual fields
  const PetscScalar  *_Dc, *_b, *_slipVel, *_slipPrev;
  PetscScalar        *_psi;
  const PetscInt      _N; // length of the arrays
  const PetscScalar   _v0, _deltaT, _f0;

  // constructor and destructor
  ComputeAging_dyn(const PetscInt N,const PetscScalar* Dc, const PetscScalar* b, PetscScalar* psi, const PetscScalar* slipVel,const PetscScalar* slipPrev, const PetscScalar v0, const PetscScalar deltaT, const PetscScalar f0);
  //~ ~ComputeVel_qd(); // use default destructor, as this class consists entirely of shallow copies

  // command to perform root-finding process, once contextual variables have been set
  PetscErrorCode computeAging(const PetscScalar rootTol, PetscInt& rootIts, const PetscInt maxNumIts);

  // function that matches root finder template
  PetscErrorCode getResid(const PetscInt Jj,const PetscScalar vel,PetscScalar* out);
  PetscErrorCode getResid(const PetscInt Jj,const PetscScalar slipVel,PetscScalar *out,PetscScalar *J){return 1;};
};


// common rate-and-state functions

// state evolution law: aging law, state variable: psi
PetscScalar agingLaw_psi(const PetscScalar& psi, const PetscScalar& slipVel, const PetscScalar& b, const PetscScalar& f0, const PetscScalar& v0, const PetscScalar& Dc);

// applies the aging law to a Vec
PetscScalar agingLaw_psi_Vec(Vec& dstate, const Vec& psi, const Vec& slipVel, const Vec& a, const Vec& b, const PetscScalar& f0, const PetscScalar& v0, const Vec& Dc);

// state evolution law: aging law, state variable: theta
PetscScalar agingLaw_theta(const PetscScalar& theta, const PetscScalar& slipVel, const PetscScalar& Dc);

// applies the aging law to a Vec
PetscScalar agingLaw_theta_Vec(Vec& dstate, const Vec& theta, const Vec& slipVel, const Vec& Dc);

// state evolution law: slip law, state variable: psi
PetscScalar slipLaw_psi(const PetscScalar& psi, const PetscScalar& slipVel, const PetscScalar& a, const PetscScalar& b, const PetscScalar& f0, const PetscScalar& v0, const PetscScalar& Dc);

// applies the state law to a Vec
PetscScalar slipLaw_psi_Vec(Vec& dstate, const Vec& psi, const Vec& slipVel,const Vec& a, const Vec& b, const PetscScalar& f0, const PetscScalar& v0, const Vec& Dc);

// state evolution law: slip law, state variable: theta
PetscScalar slipLaw_theta(const PetscScalar& state, const PetscScalar& slipVel, const PetscScalar& Dc);

// applies the state law to a Vec
PetscScalar slipLaw_theta_Vec(Vec& dstate, const Vec& theta, const Vec& slipVel, const Vec& Dc);

// frictional strength, regularized form, for state variable psi
PetscScalar strength_psi(const PetscScalar& sN, const PetscScalar& psi, const PetscScalar& slipVel, const PetscScalar& a, const PetscScalar& v0);


#include "rootFinder.hpp"

#endif
