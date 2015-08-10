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

class RootFinder;

/* TO DO:
 *   - Split setFields into setFrictionalFields, which can be defined
 *     in Fault, and setSplitNodeFields, which will be defined separately
 *     in SymmFault and FullFault.
 *
 */


// base class
class Fault: public RootFinderContext
{

  //~protected:
  public:

    // domain properties
    const PetscInt     _N;  //number of nodes on fault
    const PetscInt     _sizeMuArr;
    const PetscScalar  _L,_h; // length of fault, grid spacing on fault
    const PetscScalar  _Dc;
    const std::string  _problemType; // symmetric (only y>0) or full
    const PetscScalar  _depth,_width; // basin dimensions needed for fault properties (sigma_N, b)

    // tolerances for linear and nonlinear (for vel) solve
    PetscScalar    _rootTol;
    PetscInt       _rootIts,_maxNumIts; // total number of iterations

   // fields that are identical on split nodes
   PetscScalar    _seisDepth,_f0,_v0,_vL;
   PetscScalar    _aVal,_bBasin,_bAbove,_bBelow;
   Vec            _a,_b;
   Vec            _psi,_tempPsi,_dPsi;


    // fields that differ on the split nodes
    PetscScalar    _sigma_N_min,_sigma_N_max;
    Vec            _sigma_N;

    Vec            _zPlus;
    PetscScalar   *_muArrPlus,*_csArrPlus;
    Vec            _slip,_slipVel;

    // viewers
    PetscViewer    _slipViewer,_slipVelViewer,_tauQSPlusViewer,_psiViewer;


    PetscErrorCode setFrictionFields();

    // disable default copy constructor and assignment operator
    Fault(const Fault & that);
    Fault& operator=( const Fault& rhs);

  //~public:
    std::vector<Vec>    _var;
    Vec            _tauQSPlus;

    // iterators for _var
    typedef typename std::vector<Vec>::iterator it_vec;
    typedef typename std::vector<Vec>::const_iterator const_it_vec;

    Fault(Domain&D);
    ~Fault();

    PetscErrorCode agingLaw(const PetscInt ind,const PetscScalar psi,PetscScalar *dPsi);
    PetscErrorCode virtual computeVel() = 0;
    PetscErrorCode virtual getResid(const PetscInt ind,const PetscScalar vel,PetscScalar *out) = 0;
    PetscErrorCode virtual d_dt(const_it_vec varBegin,const_it_vec varEnd,
                                it_vec dvarBegin,it_vec dvarEnd) = 0;

    PetscErrorCode virtual setTauQS(const Vec& sigma_xyPlus,const Vec& sigma_xyMinus) = 0;
    PetscErrorCode virtual setFaultDisp(Vec const &uhatPlus,const Vec &uhatMinus) = 0;

    PetscScalar getTauInf(PetscInt& ind);

    PetscErrorCode virtual writeContext(const std::string outputDir) = 0;
    PetscErrorCode virtual writeStep(const std::string outputDir,const PetscInt step) = 0;
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
    PetscErrorCode d_dt(const_it_vec varBegin,const_it_vec varEnd,
                     it_vec dvarBegin,it_vec dvarEnd);

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
    Vec            _zMinus;
    PetscScalar   *_muArrMinus,*_csArrMinus;
    PetscInt       _arrSize; // size of _muArrMinus
    Vec            _uPlus,_uMinus,_velPlus,_velMinus;


    // time-integrated variables
    //~std::vector<Vec>    _var;

    // viewers
    PetscViewer    _uPlusViewer,_uMinusViewer,_velPlusViewer,_velMinusViewer,
                   _tauQSMinusViewer,_psiViewer;


    PetscErrorCode setSplitNodeFields();
    PetscErrorCode computeVel();

    // disable default copy constructor and assignment operator
    FullFault(const FullFault & that);
    FullFault& operator=( const FullFault& rhs);

  //~public:

    Vec            _tauQSMinus;

    // iterators for _var
    typedef typename std::vector<Vec>::iterator it_vec;
    typedef typename std::vector<Vec>::const_iterator const_it_vec;

    FullFault(Domain&D);
    ~FullFault();

    PetscErrorCode getResid(const PetscInt ind,const PetscScalar vel,PetscScalar *out);

    PetscErrorCode d_dt(const_it_vec varBegin,const_it_vec varEnd,
                     it_vec dvarBegin,it_vec dvarEnd);


    PetscErrorCode setTauQS(const Vec& sigma_xyPlus,const Vec& sigma_xyMinus);
    PetscErrorCode setFaultDisp(Vec const &uhatPlus,const Vec &uhatMinus);

    PetscErrorCode writeStep(const std::string outputDir,const PetscInt step);
    PetscErrorCode writeContext(const std::string outputDir);
};



#include "rootFinder.hpp"





#endif
