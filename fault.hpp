#ifndef FAULT_HPP_INCLUDED
#define FAULT_HPP_INCLUDED

#include <petscksp.h>
#include <cmath>
#include <assert.h>
#include <vector>
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

    // tolerances for linear and nonlinear (for vel) solve
    PetscScalar    _rootTol;
    PetscInt       _rootIts,_maxNumIts; // total number of iterations

    // elastic coefficients and frictional parameters:
    PetscScalar    _seisDepth,_cs,_f0,_v0,_vp;
    PetscScalar    _aVal,_bAbove,_bBelow;
    Vec            _sigma_N,_a,_b;

    // fields that exist on fault
    Vec            _psi,_tempPsi,_dPsi;
    PetscScalar    _sigma_N_val;

    Vec            _zPlus;
    PetscScalar   *_muArrPlus,*_csArrPlus;
    Vec            _uPlus,_velPlus;

    // viewers
    PetscViewer    _uPlusViewer,_velPlusViewer,_tauQSplusViewer,_psiViewer;


    PetscErrorCode setFrictionFields();
    PetscErrorCode agingLaw(const PetscInt ind,const PetscScalar psi,PetscScalar *dPsi);

    // disable default copy constructor and assignment operator
    Fault(const Fault & that);
    Fault& operator=( const Fault& rhs);

  //~public:

    std::vector<Vec>    _var;
    Vec            _tauQSplus;

    // iterators for _var
    typedef typename std::vector<Vec>::iterator it_vec;
    typedef typename std::vector<Vec>::const_iterator const_it_vec;

    Fault(Domain&D);
    ~Fault();

    PetscErrorCode virtual getResid(const PetscInt ind,const PetscScalar vel,PetscScalar *out) = 0;
    PetscErrorCode virtual d_dt(const_it_vec varBegin,const_it_vec varEnd,
                                it_vec dvarBegin,it_vec dvarEnd) = 0;


    PetscScalar getTauInf(PetscInt& ind);

    PetscErrorCode virtual writeContext(const std::string outputDir) = 0;
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



    PetscErrorCode setTauQS(const Vec& sigma_xyPlus);
    PetscErrorCode setFaultDisp(Vec const &uhatPlus);

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
    Vec            _uMinus,_velMinus,_vel;


    // viewers
    PetscViewer    _uMinusViewer,_velMinusViewer,_tauQSminusViewer,_psiViewer;


    PetscErrorCode setSplitNodeFields();
    PetscErrorCode computeVel();

    // disable default copy constructor and assignment operator
    FullFault(const FullFault & that);
    FullFault& operator=( const FullFault& rhs);

  //~public:

    Vec            _tauQSminus;

    // iterators for _var
    typedef typename std::vector<Vec>::iterator it_vec;
    typedef typename std::vector<Vec>::const_iterator const_it_vec;

    FullFault(Domain&D);
    ~FullFault();

    PetscErrorCode getResid(const PetscInt ind,const PetscScalar vel,PetscScalar *out);
    PetscErrorCode d_dt(const_it_vec varBegin,const_it_vec varEnd,
                     it_vec dvarBegin,it_vec dvarEnd);


    PetscErrorCode setTauQS(const Vec& sigma_xyPlus,const Vec& sigma_xyMinus);
    PetscErrorCode setFaultDisp(Vec const &uhatPlus,Vec const &uhatMinus);


    PetscErrorCode writeStep(const std::string outputDir,const PetscInt step);
    PetscErrorCode writeContext(const std::string outputDir);
};



#include "rootFinder.hpp"





#endif
