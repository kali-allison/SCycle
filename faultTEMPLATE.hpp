#ifndef FAULT_HPP_INCLUDED
#define FAULT_HPP_INCLUDED

#include <petscksp.h>
#include <cmath>
#include <assert.h>
#include "userContext.h"

using namespace std;

template <class ROOTALG>
class Fault
{

  protected:

    // domain properties
    const PetscInt    _N; //number of nodes on fault
    const PetscScalar  _L,_h; // length of fault, grid spacing on fault
    const PetscScalar _Dc;

    // tolerances for linear and nonlinear (for vel) solve
    PetscScalar    _rootTol;
    PetscInt       _rootIts,_maxNumIts; // total number of iterations
    //~Bisect         _rootFinder;

    // elastic coefficients and frictional parameters
    PetscScalar    _depth,_seisDepth,_cs,_f0,_v0,_vp,_tau_inf;
    PetscScalar    _muIn,_muOut,_D,_W,_rhoIn,_rhoOut,*_muArr;// for basin
    Vec            _eta,_sigma_N,_a,_b;

    Vec            _bcRShift;

    // fields that exist on fault
    Vec            _faultDisp,_vel;
    Vec            _psi,_tempPsi,_dPsi;
    Vec            _tau;

    Vec            _var[2];


    // disable default copy constructor and assignment operator
    Fault(const Fault & that);
    Fault& operator=( const Fault& rhs);

    PetscErrorCode computeVel();

    PetscErrorCode agingLaw(const PetscInt ind,const PetscScalar psi,PetscScalar *dPsi);
    PetscErrorCode stressMstrength(const PetscInt ind,const PetscScalar vel,PetscScalar *out);

  public:

    ROOTALG _rootAlg; // algorithm used to solve for velocity on fault


    Fault(UserContext&D);
    ~Fault();

    PetscErrorCode d_dt();

    PetscErrorCode setFields();
    PetscErrorCode setTau(const Vec&sigma_xy);
    const Vec& getBcRShift() const;

    PetscErrorCode write();
    PetscErrorCode read();
};




//================= constructor and destructor ========================
template<class ROOTALG> Fault<ROOTALG>::Fault(UserContext&D)
//~Fault::Fault(UserContext&D)
: _N(D.N),_L(D.Lz),_h(_L/(_N-1.)),_Dc(D.D_c),
  _rootTol(D.rootTol),_rootIts(0),_maxNumIts(1e8),
  _rootAlg(_maxNumIts,_rootTol),
  _depth(D.H),_seisDepth(D.H),_cs(D.cs),_f0(D.f0),_v0(D.v0),_vp(D.vp),_tau_inf(D.tau_inf),
  _D(D.D),_W(D.W),_rhoIn(D.rhoIn),_rhoOut(D.rhoOut),_muArr(D.muArr)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting constructor in fault.cpp.\n");
#endif

  // fields that exist on the fault
  VecCreate(PETSC_COMM_WORLD,&_tau);
  VecSetSizes(_tau,PETSC_DECIDE,_N);
  VecSetFromOptions(_tau);     PetscObjectSetName((PetscObject) _tau, "tau");
  VecDuplicate(_tau,&_psi); PetscObjectSetName((PetscObject) _psi, "psi");
  VecDuplicate(_tau,&_tempPsi); PetscObjectSetName((PetscObject) _tempPsi, "tempPsi");
  VecDuplicate(_tau,&_dPsi); PetscObjectSetName((PetscObject) _dPsi, "dPsi");
  VecDuplicate(_tau,&_faultDisp); PetscObjectSetName((PetscObject) _faultDisp, "faultDisp");
  VecDuplicate(_tau,&_vel); PetscObjectSetName((PetscObject) _vel, "vel");

  _var[1] = _vel;
  _var[2] = _psi;

  // frictional fields
  VecDuplicate(_tau,&_eta); PetscObjectSetName((PetscObject) _eta, "eta");
  VecDuplicate(_tau,&_sigma_N); PetscObjectSetName((PetscObject) _sigma_N, "sigma_N");
  VecDuplicate(_tau,&_a); PetscObjectSetName((PetscObject) _a, "_a");
  VecDuplicate(_tau,&_b); PetscObjectSetName((PetscObject) _b, "_b");

  VecDuplicate(_tau,&_bcRShift); PetscObjectSetName((PetscObject) _bcRShift, "_b");


#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending constructor in fault.cpp.\n");
#endif
}

template<class ROOTALG> Fault<ROOTALG>::~Fault()
//~Fault::~Fault()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting destructor in fault.cpp.\n");
#endif

  // fields that exist on the fault
  VecDestroy(&_tau);
  VecDestroy(&_psi);
  VecDestroy(&_tempPsi);
  VecDestroy(&_dPsi);
  VecDestroy(&_faultDisp);
  VecDestroy(&_vel);

  VecDestroy(&_bcRShift);

  // frictional fields
  VecDestroy(&_eta);
  VecDestroy(&_sigma_N);
  VecDestroy(&_a);
  VecDestroy(&_b);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending destructor in fault.cpp.\n");
#endif
}


//==================== protected member functions ======================

template<class ROOTALG> PetscErrorCode Fault<ROOTALG>::computeVel()
//~PetscErrorCode Fault::computeVel()
{
  PetscErrorCode ierr = 0;
  Vec            left,right,out;
  PetscScalar    outVal,leftVal,rightVal;
  PetscInt       Ii,Istart,Iend,its;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting computeVel in fault.cpp\n");CHKERRQ(ierr);
#endif

  ierr = VecDuplicate(_tau,&right);CHKERRQ(ierr);
  ierr = VecCopy(_tau,right);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(right,right,_eta);CHKERRQ(ierr);
  ierr = VecAbs(right);CHKERRQ(ierr);

  ierr = VecDuplicate(right,&left);CHKERRQ(ierr);
  ierr = VecSet(left,0.0);CHKERRQ(ierr); // assumes right-lateral fault

  ierr = VecDuplicate(left,&out);CHKERRQ(ierr);
  //~PetscErrorCode (*frictionLaw)(const PetscInt,const PetscScalar,PetscScalar *, void *) = &stressMstrength;

  ierr = VecGetOwnershipRange(left,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr = VecGetValues(left,1,&Ii,&leftVal);CHKERRQ(ierr);
    ierr = VecGetValues(right,1,&Ii,&rightVal);CHKERRQ(ierr);
    if (leftVal==rightVal) { outVal = leftVal; }
    else {
      _rootAlg.findRoot(this,Ii,&outVal);
      //~ierr = bisect((*frictionLaw),Ii,leftVal,rightVal,&outVal,&its,_rootTol,1e5,&D);CHKERRQ(ierr);
      //~_rootIts += its;
    }
    ierr = VecSetValue(_vel,Ii,outVal,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_vel);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_vel);CHKERRQ(ierr);

  ierr = VecDestroy(&left);CHKERRQ(ierr);
  ierr = VecDestroy(&right);CHKERRQ(ierr);
  ierr = VecDestroy(&out);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending computeVel in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


template<class ROOTALG> PetscErrorCode Fault<ROOTALG>::agingLaw(const PetscInt ind,const PetscScalar psi,PetscScalar *dPsi)
//~PetscErrorCode Fault::agingLaw(const PetscInt ind,const PetscScalar psi,PetscScalar *dPsi)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    b,vel;

  //~double startTime = MPI_Wtime();

  ierr = VecGetOwnershipRange(_psi,&Istart,&Iend);
  if ( (ind>=Istart) & (ind<Iend) ) {
    ierr = VecGetValues(_b,1,&ind,&b);CHKERRQ(ierr);
    ierr = VecGetValues(_vel,1,&ind,&vel);CHKERRQ(ierr);
  }
  else {
    SETERRQ(PETSC_COMM_WORLD,1,"Attempting to access nonlocal array values in agingLaw\n");
  }

  //~if (b==0) { *dPsi = 0; }
  if ( isinf(exp(1/b)) ) { *dPsi = 0; }
  else if ( b <= 1e-3 ) { *dPsi = 0; }
  else {
    *dPsi = (PetscScalar) (b*_v0/_Dc)*( exp((double) ( (_f0-psi)/b) ) - (vel/_v0) );
  }


  if (isnan(*dPsi)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isnan(*dPsi) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%.9e,b=%.9e,f0=%.9e,D_c=%.9e,v0=%.9e,vel=%.9e\n",psi,b,_f0,_Dc,_v0,vel);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(b*D->v0/D->D_c)=%.9e\n",(b*_v0/_Dc));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"exp((double) ( (D->f0-psi)/b) )=%.9e\n",exp((double) ( (_f0-psi)/b) ));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(vel/D->v0)=%.9e\n",(vel/_v0));
    CHKERRQ(ierr);
  }
  else if (isinf(*dPsi)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isinf(*dPsi) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%.9e,b=%.9e,f0=%.9e,D_c=%.9e,v0=%.9e,vel=%.9e\n",psi,b,_f0,_Dc,_v0,vel);
    CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(b*D->v0/D->D_c)=%.9e\n",(b*_v0/_Dc));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"exp((double) ( (D->f0-psi)/b) )=%.9e\n",exp((double) ( (_f0-psi)/b) ));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(vel/D->v0)=%.9e\n",(vel/_v0));
  }

  assert(!isnan(*dPsi));
  assert(!isinf(*dPsi));

  //~double endTime = MPI_Wtime();
  //~D->agingLawTime = D->agingLawTime + (endTime-startTime);

  return ierr;
}

template<class ROOTALG> PetscErrorCode Fault<ROOTALG>::stressMstrength(const PetscInt ind,const PetscScalar vel,PetscScalar *out)
//~PetscErrorCode stressMstrength(const PetscInt ind,const PetscScalar vel,PetscScalar *out, void * ctx)
{
  PetscErrorCode ierr = 0;
  PetscScalar    psi,a,sigma_n,eta,tau;
  PetscInt       Istart,Iend;

#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting stressMstrength in fault.hpp\n");CHKERRQ(ierr);
#endif

  ierr = VecGetOwnershipRange(_psi,&Istart,&Iend);
  if ( (ind>=Istart) & (ind<Iend) ) {
    ierr = VecGetValues(_tempPsi,1,&ind,&psi);CHKERRQ(ierr);
    ierr = VecGetValues(_a,1,&ind,&a);CHKERRQ(ierr);
    ierr = VecGetValues(_sigma_N,1,&ind,&sigma_n);CHKERRQ(ierr);
    ierr = VecGetValues(_eta,1,&ind,&eta);CHKERRQ(ierr);
    ierr = VecGetValues(_tau,1,&ind,&tau);CHKERRQ(ierr);
  }
  else {
    SETERRQ(PETSC_COMM_WORLD,1,"Attempting to access nonlocal array values in stressMstrength\n");
  }

   *out = (PetscScalar) a*sigma_n*asinh( (double) (vel/2/_v0)*exp(psi/a) ) + eta*vel - tau;

#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"    psi=%g,a=%g,sigma_n=%g,eta=%g,tau=%g,vel=%g\n",psi,a,sigma_n,eta,tau,vel);
#endif
  if (isnan(*out)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isnan(*out) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%g,a=%g,sigma_n=%g,eta=%g,tau=%g,vel=%g\n",psi,a,sigma_n,eta,tau,vel);
    CHKERRQ(ierr);
  }
  else if (isinf(*out)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isinf(*out) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%g,a=%g,sigma_n=%g,eta=%g,tau=%g,vel=%g\n",psi,a,sigma_n,eta,tau,vel);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(vel/2/_v0)=%.9e\n",vel/2/_v0);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"exp(psi/a)=%.9e\n",exp(psi/a));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"eta*vel=%.9e\n",eta*vel);
    CHKERRQ(ierr);
  }

  assert(!isnan(*out));
  assert(!isinf(*out));

#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending stressMstrength in fault.hpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

//==================== set/get functions ===============================
template<class ROOTALG> PetscErrorCode Fault<ROOTALG>::setFields()
//~PetscErrorCode Fault::setFields()
{
  PetscErrorCode ierr = 0;
  PetscInt       Ii,Istart,Iend;
  PetscScalar    v,z;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting setFields in fault.cpp\n");CHKERRQ(ierr);
#endif

  ierr = VecSet(_psi,_f0);CHKERRQ(ierr);
  ierr = VecCopy(_psi,_tempPsi);CHKERRQ(ierr);
  ierr = VecSet(_sigma_N,50.);CHKERRQ(ierr);
  ierr = VecSet(_a,0.015);CHKERRQ(ierr);

  // Set b
  PetscScalar L2 = 1.5*_seisDepth;  //This is depth at which increase stops and fault is purely velocity strengthening
  PetscInt    N1 = _seisDepth/_h;
  PetscInt    N2 = L2/_h;
  ierr = VecGetOwnershipRange(_b,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii < N1+1) {
      v=0.02;
      ierr = VecSetValues(_b,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    else if (Ii>N1 && Ii<=N2) {
      v = 0.02/(_seisDepth-L2);v = v*Ii*_h - v*L2;
      ierr = VecSetValues(_b,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    else {
      v = 0.0;
      ierr = VecSetValues(_b,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_b);CHKERRQ(ierr);
  //ierr = VecSet(D.b,0.02);CHKERRQ(ierr); // for spring-slider!!!!!!!!!!!!!!!!


  // tau, eta, gRShift
  PetscScalar a,b,eta,tau_inf,sigma_N,bcRShift;
  ierr = VecGetOwnershipRange(_tau,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr =  VecGetValues(_a,1,&Ii,&a);CHKERRQ(ierr);
    ierr =  VecGetValues(_b,1,&Ii,&b);CHKERRQ(ierr);
    ierr =  VecGetValues(_sigma_N,1,&Ii,&sigma_N);CHKERRQ(ierr);

    tau_inf = sigma_N*a*asinh( (double) 0.5*_vp*exp(_f0/a)/_v0 );
    z = ((double) Ii)*_h;
    if (z < _depth) { eta = 0.5*sqrt(_rhoIn*_muArr[Ii]); }
    else { eta = 0.5*sqrt(_rhoOut*_muArr[Ii]); }
    bcRShift = tau_inf*_L/_muArr[Ii];

    ierr = VecSetValue(_tau,Ii,tau_inf,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(_eta,Ii,eta,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(_bcRShift,Ii,bcRShift,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_tau);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_eta);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_bcRShift);CHKERRQ(ierr);

  ierr = VecAssemblyEnd(_tau);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_eta);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcRShift);CHKERRQ(ierr);


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending setFields in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

template<class ROOTALG> PetscErrorCode Fault<ROOTALG>::setTau(const Vec&sigma_xy)
//~PetscErrorCode Fault::setTau(const Vec&sigma_xy)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting setTau in lithosphere.cpp.\n");CHKERRQ(ierr);
#endif

  PetscInt       Ii,Istart,Iend;
  PetscScalar    v;

  ierr = VecGetOwnershipRange(sigma_xy,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii<_N) {
      ierr = VecGetValues(sigma_xy,1,&Ii,&v);CHKERRQ(ierr);
      ierr = VecSetValues(_tau,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_tau);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_tau);CHKERRQ(ierr);


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending setTau in lithosphere.c\n");CHKERRQ(ierr);
#endif
  return ierr;
}

template<class ROOTALG> const Vec& Fault<ROOTALG>::getBcRShift() const
//~const Vec& Fault::getBcRShift() const
{
  return _bcRShift;
}











#endif
