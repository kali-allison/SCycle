#ifndef FAULT_HPP_INCLUDED
#define FAULT_HPP_INCLUDED

#include <petscksp.h>
#include <cmath>
#include <assert.h>
#include <vector>
#include <cmath>
#include "genFuncs.hpp"
#include "domain.hpp"
#include "heatEquation.hpp"
#include "rootFinderContext.hpp"

// defines if state is in terms of psi (=1) or in terms of theta (=0)
//~ #define STATE_PSI 1

class RootFinder;


// base class
class Fault: public RootFinderContext
{

  //~protected:
  public:
    const char       *_file; // input file
    std::string       _delim; // format is: var delim value (without the white space)
    std::string       _outputDir; // directory for output
    const bool        _isMMS; // true if running mms test
    bool              _bcLTauQS;
    std::string       _stateLaw; // state evolution law

    // domain properties
    const PetscInt     _N;  //number of nodes on fault
    const PetscScalar  _L,_h; // length of fault, grid spacing on fault
    Vec                _z; // vector of z-coordinates on fault (allows for variable grid spacing)
    const std::string  _problemType; // symmetric (only y>0) or full

    // tolerances for linear and nonlinear (for vel) solve
    PetscScalar    _rootTol;
    PetscInt       _rootIts,_maxNumIts; // total number of iterations

   // fields that are identical on split nodes
   PetscScalar           _f0,_v0,_vL;
   PetscScalar           _fw,_Vw,_tau_c,_Tw,_D; // flash heating parameters
   Vec                   _T,_k,_rho,_c; // for flash heating
   std::vector<double>   _aVals,_aDepths,_bVals,_bDepths,_DcVals,_DcDepths;
   Vec                   _a,_b,_Dc;
   std::vector<double>   _cohesionVals,_cohesionDepths;
   Vec                   _cohesion;
   Vec                   _dPsi,_psi,_theta,_dTheta;


    // fields that differ on the split nodes
    std::vector<double>  _impedanceVals,_impedanceDepths;
    std::vector<double>  _sigmaNVals,_sigmaNDepths;
    PetscScalar          _sigmaN_cap,_sigmaN_floor; // allow cap and floor on normal stress
    Vec                  _sNEff; // effective normal stress
    Vec                  _sN; // total normal stress
    Vec                  _zP;
    //~ PetscScalar   *_muArrPlus,*_csArrPlus;
    Vec                 *_muVecP,*_csVecP;
    Vec                  _slip,_slipVel;

    // viewers:
    // 1st string = key naming relevant field, e.g. "slip"
    // 2nd PetscViewer = PetscViewer object for file IO
    // 3rd string = full file path name for output
    //~ std::map <string,PetscViewer>  _viewers;
    std::map <string,std::pair<PetscViewer,string> >  _viewers;

    // runtime data
    double               _computeVelTime,_stateLawTime;


    PetscErrorCode setFrictionFields(Domain&D);

    // disable default copy constructor and assignment operator
    Fault(const Fault & that);
    Fault& operator=( const Fault& rhs);

    PetscErrorCode setVecFromVectors(Vec&, vector<double>&,vector<double>&);
    PetscErrorCode  setVecFromVectors(Vec& vec, vector<double>& vals,vector<double>& depths,
      const PetscScalar maxVal);

  //~public:
    Vec            _tauQSP;
    Vec            _tauP; // not quasi-static

    // iterators for _var
    typedef std::vector<Vec>::iterator it_vec;
    typedef std::vector<Vec>::const_iterator const_it_vec;

    Fault(Domain& D, HeatEquation& He);
    virtual ~Fault();


    // state evolution equations
    PetscErrorCode agingLaw_theta(const PetscInt ind,const PetscScalar state,PetscScalar &dstate);
    PetscErrorCode agingLaw_psi(const PetscInt ind,const PetscScalar state,PetscScalar &dstate);
    PetscErrorCode slipLaw_theta(const PetscInt ind,const PetscScalar state,PetscScalar &dstate);
    PetscErrorCode slipLaw_psi(const PetscInt ind,const PetscScalar state,PetscScalar &dstate);
    PetscErrorCode flashHeating_psi(const PetscInt ind,const PetscScalar state,PetscScalar &dstate);
    PetscErrorCode stronglyVWLaw_theta(const PetscInt ind,const PetscScalar state,PetscScalar &dstate);


    PetscErrorCode virtual computeVel() = 0;
    PetscErrorCode virtual getResid(const PetscInt ind,const PetscScalar vel,PetscScalar *out) = 0;
    PetscErrorCode virtual getResid(const PetscInt ind,const PetscScalar vel,PetscScalar *out,PetscScalar *J) = 0;
    PetscErrorCode virtual d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx) = 0;
    PetscErrorCode virtual d_dt_WaveEq(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& dvarEx, PetscScalar _deltaT) = 0;
    PetscErrorCode virtual initiateIntegrand(const PetscScalar time, map<string,Vec>& varEx) = 0;
    PetscErrorCode virtual updateFields(const PetscScalar time,const map<string,Vec>& varEx) = 0;

    PetscErrorCode virtual setTauQS(const Vec& sigma_xyPlus,const Vec& sigma_xyMinus) = 0;
    PetscErrorCode virtual setFaultDisp(Vec const &uhatPlus,const Vec &uhatMinus) = 0;
    PetscErrorCode virtual setSNEff(const Vec& p) = 0; // update effective normal stress to reflect new pore pressure
    PetscErrorCode virtual setSN(const Vec& p) = 0; // update effective normal stress to reflect new pore pressure
    PetscErrorCode virtual setTemp(const Vec& T) = 0;


    // for steady state computations
    PetscErrorCode getTauRS(Vec& tauRS, const PetscScalar vL);
    PetscErrorCode computeVss(const Vec tau); // compute slip vel from tau
    PetscScalar    getTauSS(PetscInt& ind); // return steady-state shear stress
    PetscErrorCode initiateVarSS(map<string,Vec>& varSS); // tau and slip vel in varSS
    PetscErrorCode updateSS(map<string,Vec>& varSS);

    // IO
    PetscErrorCode virtual view(const double totRunTime);
    PetscErrorCode virtual writeContext(const std::string outputDir) = 0;
    PetscErrorCode virtual writeStep(const PetscInt stepCount, const PetscScalar time) = 0;
    PetscErrorCode virtual writeStep(const PetscInt stepCount, const PetscScalar time, const std::string outputDir) = 0;

    // load settings from input file
    PetscErrorCode loadSettings(const char *file);
    PetscErrorCode loadFieldsFromFiles(std::string inputDir);
    PetscErrorCode checkInput(); // check input from file
    PetscErrorCode setHeatParams(const Vec& k,const Vec& rho,const Vec& c);

    // mms test
    PetscErrorCode virtual measureMMSError(const double totRunTime) { return 0; };

};




class SymmFault: public Fault
{

  //~protected:

  private:

    PetscErrorCode setSplitNodeFields();


    // disable default copy constructor and assignment operator
    SymmFault(const SymmFault & that);
    SymmFault& operator=( const SymmFault& rhs);

  public:



    SymmFault(Domain&D, HeatEquation& He);
    ~SymmFault();

    // for interaction with mediator
    PetscErrorCode initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx);
    PetscErrorCode updateFields(const PetscScalar time,const map<string,Vec>& varEx);
    PetscErrorCode d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);
    PetscErrorCode d_dt_WaveEq(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& dvarEx, PetscScalar _deltaT);
    PetscErrorCode virtual d_dt_eqCycle(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);
    PetscErrorCode virtual d_dt_mms(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx);

    PetscErrorCode virtual writeStep(const PetscInt stepCount, const PetscScalar time);
    PetscErrorCode virtual writeStep(const PetscInt stepCount, const PetscScalar time, const std::string outputDir);
    PetscErrorCode virtual writeContext(const std::string outputDir);

    PetscErrorCode getResid(const PetscInt ind,const PetscScalar vel,PetscScalar* out);
    PetscErrorCode getResid(const PetscInt ind,const PetscScalar vel,PetscScalar* out,PetscScalar* J);
    PetscErrorCode computeVel();

    // don't technically need the 2nd argument
    PetscErrorCode setTemp(const Vec& T);
    PetscErrorCode getTau(Vec& tau);
    PetscErrorCode setTauQS(const Vec& sigma_xyPlus,const Vec& sigma_xyMinus);
    PetscErrorCode setFaultDisp(Vec const &uhatPlus,const Vec &uhatMinus);
    PetscErrorCode setSNEff(const Vec& p); // update effective normal stress to reflect new pore pressure
    PetscErrorCode setSN(const Vec& p); // set total normal stress from pore pressure and effective normal stress

};





/*
class FullFault: public Fault
{

  //~protected:
  public:

    // fields that exist on left split nodes
    Vec            _zM;
    PetscScalar   *_muArrMinus,*_csArrMinus;
    PetscInt       _arrSize; // size of _muArrMinus
    Vec            _uP,_uM,_velPlus,_velMinus;


    // time-integrated variables
    //~std::vector<Vec>    _var;

    // viewers
    PetscViewer    _uPlusViewer,_uMV,_velPlusViewer,_velMinusViewer,
                   _tauQSMinusViewer,_stateViewer;


    PetscErrorCode setSplitNodeFields();
    PetscErrorCode computeVel();

    // disable default copy constructor and assignment operator
    FullFault(const FullFault & that);
    FullFault& operator=( const FullFault& rhs);

  //~public:

    Vec            _tauQSMinus;

    // iterators for _var
    typedef std::vector<Vec>::iterator it_vec;
    typedef std::vector<Vec>::const_iterator const_it_vec;

    FullFault(Domain&D, HeatEquation& He);
    ~FullFault();

    PetscErrorCode getResid(const PetscInt ind,const PetscScalar vel,PetscScalar* out);
    PetscErrorCode getResid(const PetscInt ind,const PetscScalar vel,PetscScalar* out,PetscScalar* J);

    PetscErrorCode d_dt(const map<string,Vec>& varEx,map<string,Vec>& dvarEx);


    PetscErrorCode setTauQS(const Vec& sigma_xyPlus,const Vec& sigma_xyMinus);
    PetscErrorCode setFaultDisp(Vec const &uhatPlus,const Vec &uhatMinus);

    PetscErrorCode writeStep(const std::string outputDir,const PetscInt step);
    PetscErrorCode writeContext(const std::string outputDir);
};
*/


#include "rootFinder.hpp"





#endif
