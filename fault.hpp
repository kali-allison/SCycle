#ifndef FAULT_HPP_INCLUDED
#define FAULT_HPP_INCLUDED

#include <petscksp.h>
#include <cmath>
#include <assert.h>
#include <vector>
#include <cmath>
#include "genFuncs.hpp"
#include "domain.hpp"
#include "rootFinderContext.hpp"

// defines if state is in terms of psi (=1) or in terms of theta (=0)
#define STATE_PSI 1

class RootFinder;


// base class
class Fault: public RootFinderContext
{

  //~protected:
  public:
    const char       *_file;
    std::string       _delim; // format is: var delim value (without the white space)
    std::string       _stateLaw; // state evolution law

    // domain properties
    const PetscInt     _N;  //number of nodes on fault
    const PetscInt     _sizeMuArr;
    const PetscScalar  _L,_h; // length of fault, grid spacing on fault
    Vec                _z; // vector of z-coordinates on fault (allows for variable grid spacing)
    const std::string  _problemType; // symmetric (only y>0) or full
    const PetscScalar  _depth,_width; // basin dimensions needed for fault properties (sigma_N, b)

    // tolerances for linear and nonlinear (for vel) solve
    PetscScalar    _rootTol;
    PetscInt       _rootIts,_maxNumIts; // total number of iterations

   // fields that are identical on split nodes
   PetscScalar    _f0,_v0,_vL;
   std::vector<double> _aVals,_aDepths,_bVals,_bDepths,_DcVals,_DcDepths;
   Vec            _a,_b,_Dc;
   std::vector<double> _cohesionVals,_cohesionDepths;
   Vec            _cohesion;
   Vec            _state,_dPsi;


    // fields that differ on the split nodes
    std::vector<double> _sigmaNVals,_sigmaNDepths;
    Vec            _sigma_N;

    Vec            _zP;
    //~ PetscScalar   *_muArrPlus,*_csArrPlus;
    Vec           *_muVecP,*_csVecP;
    Vec            _slip,_slipVel;

    // viewers
    PetscViewer    _slipViewer,_slipVelViewer,_tauQSPlusViewer,_stateViewer;


    PetscErrorCode setFrictionFields(Domain&D);

    // disable default copy constructor and assignment operator
    Fault(const Fault & that);
    Fault& operator=( const Fault& rhs);

    PetscErrorCode setVecFromVectors(Vec&, vector<double>&,vector<double>&);

  //~public:
    Vec            _tauQSP;

    // iterators for _var
    typedef std::vector<Vec>::iterator it_vec;
    typedef std::vector<Vec>::const_iterator const_it_vec;

    Fault(Domain&D);
    ~Fault();

    PetscErrorCode agingLaw(const PetscInt ind,const PetscScalar state,PetscScalar &dstate);
    PetscErrorCode slipLaw(const PetscInt ind,const PetscScalar state,PetscScalar &dstate);
    PetscErrorCode stronglyVWLaw(const PetscInt ind,const PetscScalar state,PetscScalar &dstate);
    PetscErrorCode virtual computeVel() = 0;
    PetscErrorCode virtual getResid(const PetscInt ind,const PetscScalar vel,PetscScalar *out) = 0;
    PetscErrorCode virtual d_dt(const_it_vec varBegin,it_vec dvarBegin) = 0;

    PetscErrorCode virtual setTauQS(const Vec& sigma_xyPlus,const Vec& sigma_xyMinus) = 0;
    PetscErrorCode virtual setFaultDisp(Vec const &uhatPlus,const Vec &uhatMinus) = 0;

    PetscScalar getTauInf(PetscInt& ind);

    PetscErrorCode virtual writeContext(const std::string outputDir) = 0;
    PetscErrorCode virtual writeStep(const std::string outputDir,const PetscInt step) = 0;

    // load settings from input file
    PetscErrorCode loadSettings(const char *file);
    PetscErrorCode loadFieldsFromFiles(std::string inputDir);

    // check input from file
    PetscErrorCode checkInput();
};




class SymmFault: public Fault
{

  //~protected:

  public:

    PetscErrorCode setSplitNodeFields();
    PetscErrorCode computeVel();

    // disable default copy constructor and assignment operator
    SymmFault(const SymmFault & that);
    SymmFault& operator=( const SymmFault& rhs);

  //~public:



    SymmFault(Domain&D);
    ~SymmFault();

    PetscErrorCode getResid(const PetscInt ind,const PetscScalar vel,PetscScalar *out);
    PetscErrorCode d_dt(const_it_vec varBegin,it_vec dvarBegin);

    // don't technically need the 2nd argument
    PetscErrorCode setTauQS(const Vec& sigma_xyPlus,const Vec& sigma_xyMinus);
    PetscErrorCode setFaultDisp(Vec const &uhatPlus,const Vec &uhatMinus);

    PetscErrorCode writeStep(const std::string outputDir,const PetscInt step);
    PetscErrorCode writeContext(const std::string outputDir);
};






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

    FullFault(Domain&D);
    ~FullFault();

    PetscErrorCode getResid(const PetscInt ind,const PetscScalar vel,PetscScalar *out);

    PetscErrorCode d_dt(const_it_vec varBegin,it_vec dvarBegin);


    PetscErrorCode setTauQS(const Vec& sigma_xyPlus,const Vec& sigma_xyMinus);
    PetscErrorCode setFaultDisp(Vec const &uhatPlus,const Vec &uhatMinus);

    PetscErrorCode writeStep(const std::string outputDir,const PetscInt step);
    PetscErrorCode writeContext(const std::string outputDir);
};



#include "rootFinder.hpp"





#endif
