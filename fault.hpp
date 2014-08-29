#ifndef FAULT_HPP_INCLUDED
#define FAULT_HPP_INCLUDED

#include <petscksp.h>
#include <cmath>
#include <assert.h>
//~#include "userContext.h"
#include "domain.hpp"

using namespace std;

class RootFinder;

class Fault
{

  //~protected:
  public:

    // domain properties
    const PetscInt     _N; //number of nodes on fault
    const PetscScalar  _L,_h; // length of fault, grid spacing on fault
    const PetscScalar  _Dc;

    // tolerances for linear and nonlinear (for vel) solve
    PetscScalar    _rootTol;
    PetscInt       _rootIts,_maxNumIts; // total number of iterations

    // elastic coefficients and frictional parameters
    PetscScalar    _depth,_seisDepth,_cs,_f0,_v0,_vp;
    PetscScalar    _bAbove,_bBelow;
    PetscScalar    _muIn,_muOut,_rhoIn,_rhoOut,*_muArr,*_rhoArr;// for basin
    Vec            _eta,_sigma_N,_a,_b;

    Vec            _bcRShift;

    // fields that exist on fault
    Vec            _faultDisp,_vel;
    Vec            _psi,_tempPsi,_dPsi;
    PetscScalar     _sigma_N_val;

    // viewers
    PetscViewer   _faultDispViewer,_velViewer,_tauViewer,_psiViewer;

    PetscErrorCode computeVel();
    PetscErrorCode agingLaw(const PetscInt ind,const PetscScalar psi,PetscScalar *dPsi);

    // disable default copy constructor and assignment operator
    Fault(const Fault & that);
    Fault& operator=( const Fault& rhs);

  //~public:

    RootFinder    *_rootAlg; // algorithm used to solve for velocity on fault
    Vec            _var[2];
    Vec            _tau;


    Fault(Domain&D);
    ~Fault();

    PetscErrorCode getResid(const PetscInt ind,const PetscScalar vel,PetscScalar *out);
    PetscErrorCode d_dt(Vec const*var,Vec*dvar);


    PetscErrorCode setFields();
    PetscErrorCode setTau(const Vec&sigma_xy);
    PetscErrorCode setFaultDisp(Vec const &bcF);
    const Vec& getBcRShift() const;

    PetscErrorCode writeContext(const string outputDir);
    PetscErrorCode writeStep(const string outputDir,const PetscInt step);
    PetscErrorCode read();
};

#include "rootFinder.hpp"





#endif
