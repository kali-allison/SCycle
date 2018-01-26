#ifndef NEWFAULT_HPP_INCLUDED
#define NEWFAULT_HPP_INCLUDED

#include <petscksp.h>
#include <cmath>
#include <assert.h>
#include <vector>
#include <cmath>
#include "genFuncs.hpp"
#include "domain.hpp"
#include "heatEquation.hpp"
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
    const char       *_inputFile; // input file
    std::string       _delim; // format is: var delim value (without the white space)
    std::string       _outputDir; // directory for output
    std::string       _stateLaw; // state evolution law

    // domain properties
    const PetscInt     _N;  //number of nodes on fault
    const PetscScalar  _L; // length of fault, grid spacing on fault
    Vec                _z; // vector of z-coordinates on fault (allows for variable grid spacing)

    // rate-and-state parameters
    PetscScalar           _f0,_v0;
    std::vector<double>   _aVals,_aDepths,_bVals,_bDepths,_DcVals,_DcDepths;
    Vec                   _a,_b,_Dc;
    std::vector<double>   _cohesionVals,_cohesionDepths;
    Vec                   _cohesion;
    Vec                   _dPsi,_psi;
    std::vector<double>   _sigmaNVals,_sigmaNDepths;
    PetscScalar           _sigmaN_cap,_sigmaN_floor; // allow cap and floor on normal stress
    Vec                   _sNEff; // effective normal stress
    Vec                   _sN; // total normal stress
    Vec                   _eta_rad; // radiation damping parameter
    PetscScalar           _muVal,_rhoVal; // if constant
    Vec                   _slip,_slipVel;

    Vec            _tauQSP;
    Vec            _tauP; // not quasi-static

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
    PetscErrorCode setVecFromVectors(Vec&, vector<double>&,vector<double>&);
    PetscErrorCode setVecFromVectors(Vec& vec, vector<double>& vals,vector<double>& depths,
      const PetscScalar maxVal);

    PetscErrorCode setTauQS(const Vec& tau);
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

    NewFault_qd(Domain& D,VecScatter& scatter2fault);
    ~NewFault_qd();

    // for interaction with mediator
    PetscErrorCode initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx);
    PetscErrorCode updateFields(const PetscScalar time,const map<string,Vec>& varEx);
    PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);

    PetscErrorCode getResid(const PetscInt ind,const PetscScalar vel,PetscScalar* out);
    PetscErrorCode computeVel();
};


// structs for root-finding pieces

// computing the slip velocity for the quasi-dynamic problem
struct ComputeVel_qd : public RootFinderContext
{
  // shallow copies of contextual fields
  const PetscScalar  *_a, *_b, *_sN, *_tauQS, *_eta, *_psi;
  PetscInt      _N; // length of the arrays
  PetscScalar   _v0;

  // constructor and destructor
  ComputeVel_qd(const PetscInt N, const PetscScalar* eta,const PetscScalar* tauQS,const PetscScalar* sN,const PetscScalar* psi,const PetscScalar* a,const PetscScalar* b,const PetscScalar& v0);
  //~ ~ComputeVel_qd(); // use default destructor

  // command to perform root-finding process, once contextual variables have been set
  PetscErrorCode computeVel(PetscScalar* slipVelA, const PetscScalar rootTol, PetscInt& rootIts, const PetscInt maxNumIts);

  // function that matches root finder template
  PetscErrorCode getResid(const PetscInt Jj,const PetscScalar vel,PetscScalar* out);
  PetscErrorCode getResid(const PetscInt Jj,const PetscScalar slipVel,PetscScalar *out,PetscScalar *J);
};



// common rate-and-state functions

// state evolution law: aging law, state variable: psi
PetscScalar agingLaw_psi(const PetscScalar& psi, const PetscScalar& slipVel, const PetscScalar& a, const PetscScalar& b, const PetscScalar& f0, const PetscScalar& v0, const PetscScalar& Dc)
{
  PetscScalar A = exp( (double) (f0-psi)/b );
  PetscScalar dstate = 0.;
  if ( ~isinf(A) ) {
    dstate = (PetscScalar) (b*v0/Dc)*( A - slipVel/v0 );
  }
  assert(!isnan(dstate));
  assert(!isinf(dstate));
  return dstate;
}

// applies the aging law to a Vec
PetscScalar agingLaw_psi_Vec(Vec& dstate, const Vec& psi, const Vec& slipVel, const Vec& a, const Vec& b, const PetscScalar& f0, const PetscScalar& v0, const Vec& Dc)
{
  PetscErrorCode ierr = 0;

  PetscScalar *psiA,*dstateA,*slipVelA,*aA,*bA,*DcA = 0;
  VecGetArray(dstate,&dstateA);
  VecGetArray(psi,&psiA);
  VecGetArray(slipVel,&slipVelA);
  VecGetArray(a,&aA);
  VecGetArray(b,&bA);
  VecGetArray(Dc,&DcA);
  PetscInt Jj = 0; // local array index
  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(psi,&Istart,&Iend); // local portion of global Vec index
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    dstateA[Jj] = agingLaw_psi(psiA[Jj], slipVelA[Jj], aA[Jj], bA[Jj], f0, v0, DcA[Jj]);
    Jj++;
  }
  VecRestoreArray(dstate,&dstateA);
  VecRestoreArray(psi,&psiA);
  VecRestoreArray(slipVel,&slipVelA);
  VecRestoreArray(a,&aA);
  VecRestoreArray(b,&bA);
  VecRestoreArray(Dc,&DcA);

  return ierr;
}

// state evolution law: aging law, state variable: theta
PetscScalar agingLaw_theta(const PetscScalar& theta, const PetscScalar& slipVel, const PetscScalar& Dc)
{
  PetscScalar dstate = 1. - theta*slipVel/Dc;

  assert(!isnan(dstate));
  assert(!isinf(dstate));
  return dstate;
}

// applies the aging law to a Vec
PetscScalar agingLaw_theta_Vec(Vec& dstate, const Vec& theta, const Vec& slipVel, const Vec& Dc)
{
  PetscErrorCode ierr = 0;

  PetscScalar *thetaA,*dstateA,*slipVelA,*DcA = 0;
  VecGetArray(dstate,&dstateA);
  VecGetArray(theta,&thetaA);
  VecGetArray(slipVel,&slipVelA);
  VecGetArray(Dc,&DcA);
  PetscInt Jj = 0; // local array index
  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(theta,&Istart,&Iend); // local portion of global Vec index
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    dstateA[Jj] = agingLaw_theta(thetaA[Jj], slipVelA[Jj], DcA[Jj]);
    Jj++;
  }
  VecRestoreArray(dstate,&dstateA);
  VecRestoreArray(theta,&thetaA);
  VecRestoreArray(slipVel,&slipVelA);
  VecRestoreArray(Dc,&DcA);

  return ierr;
}

// state evolution law: slip law, state variable: psi
PetscScalar slipLaw_psi(const PetscScalar& psi, const PetscScalar& slipVel, const PetscScalar& a, const PetscScalar& b, const PetscScalar& f0, const PetscScalar& v0, const PetscScalar& Dc)
{
  PetscScalar fss = f0 + (a-b)*log(slipVel/v0);
  PetscScalar f = psi + a*log(slipVel/v0);
  PetscScalar dstate = -slipVel/Dc *(f - fss);

  assert(!isnan(dstate));
  assert(!isinf(dstate));
  return dstate;
}

// applies the state law to a Vec
PetscScalar slipLaw_psi_Vec(Vec& dstate, const Vec& psi, const Vec& slipVel,const Vec& a, const Vec& b, const PetscScalar& f0, const PetscScalar& v0, const Vec& Dc)
{
  PetscErrorCode ierr = 0;

  PetscScalar *psiA,*dstateA,*slipVelA,*aA,*bA,*DcA = 0;
  VecGetArray(dstate,&dstateA);
  VecGetArray(psi,&psiA);
  VecGetArray(slipVel,&slipVelA);
  VecGetArray(a,&aA);
  VecGetArray(b,&bA);
  VecGetArray(Dc,&DcA);
  PetscInt Jj = 0; // local array index
  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(psi,&Istart,&Iend); // local portion of global Vec index
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    dstateA[Jj] = slipLaw_psi(psiA[Jj], slipVelA[Jj], aA[Jj], bA[Jj], f0, v0, DcA[Jj]);
    Jj++;
  }
  VecRestoreArray(dstate,&dstateA);
  VecRestoreArray(psi,&psiA);
  VecRestoreArray(slipVel,&slipVelA);
  VecRestoreArray(a,&aA);
  VecRestoreArray(b,&bA);
  VecRestoreArray(Dc,&DcA);

  return ierr;
}

// state evolution law: slip law, state variable: theta
PetscScalar slipLaw_theta(const PetscScalar& state, const PetscScalar& slipVel, const PetscScalar& Dc)
{
  PetscScalar A = state*slipVel/Dc;
  PetscScalar dstate = 0.;
  if (A != 0.) { dstate = -A*log(A); }

  assert(!isnan(dstate));
  assert(!isinf(dstate));
  return dstate;
}

// applies the state law to a Vec
PetscScalar slipLaw_theta_Vec(Vec& dstate, const Vec& theta, const Vec& slipVel, const Vec& Dc)
{
  PetscErrorCode ierr = 0;

  PetscScalar *thetaA,*dstateA,*slipVelA,*DcA = 0;
  VecGetArray(dstate,&dstateA);
  VecGetArray(theta,&thetaA);
  VecGetArray(slipVel,&slipVelA);
  VecGetArray(Dc,&DcA);
  PetscInt Jj = 0; // local array index
  PetscInt Istart, Iend;
  ierr = VecGetOwnershipRange(theta,&Istart,&Iend); // local portion of global Vec index
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    dstateA[Jj] = slipLaw_theta(thetaA[Jj], slipVelA[Jj], DcA[Jj]);
    Jj++;
  }
  VecRestoreArray(dstate,&dstateA);
  VecRestoreArray(theta,&thetaA);
  VecRestoreArray(slipVel,&slipVelA);
  VecRestoreArray(Dc,&DcA);

  return ierr;
}


// frictional strength, regularized form, for state variable psi
PetscScalar strength_psi(const PetscScalar& sN, const PetscScalar& psi, const PetscScalar& slipVel, const PetscScalar& a, const PetscScalar& b, const PetscScalar& v0)
{
  PetscScalar strength = (PetscScalar) a*sN*asinh( (double) (slipVel/2./v0)*exp(psi/a) );
  return strength;
}






#include "rootFinder.hpp"





#endif
