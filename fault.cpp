#include "fault.hpp"

using namespace std;




Fault::Fault(Domain&D)
: _N(D._Nz),_sizeMuArr(D._Ny*D._Nz),_L(D._Lz),_h(_L/(_N-1.)),_Dc(D._Dc),
  _problemType(D._problemType),
  _depth(D._depth),_width(D._width),
  _rootTol(D._rootTol),_rootIts(0),_maxNumIts(1e8),
  _seisDepth(D._seisDepth),_f0(D._f0),_v0(D._v0),_vp(D._vp),
  _aVal(D._aVal),_bBasin(D._bBasin),_bAbove(D._bAbove),_bBelow(D._bBelow),
  _a(NULL),_b(NULL),
  _psi(NULL),_tempPsi(NULL),_dPsi(NULL),
  _sigma_N(D._sigma_N),
  _muArrPlus(D._muArrPlus),_csArrPlus(D._csArrPlus),_uPlus(NULL),_velPlus(NULL),
  _uPlusViewer(NULL),_velPlusViewer(NULL),_tauQSplusViewer(NULL),
  _psiViewer(NULL),
  _tauQSplus(NULL)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting Fault::Fault in fault.cpp.\n");
#endif

  // fields that exist on the fault
  VecCreate(PETSC_COMM_WORLD,&_tauQSplus);
  VecSetSizes(_tauQSplus,PETSC_DECIDE,_N);
  VecSetFromOptions(_tauQSplus);     PetscObjectSetName((PetscObject) _tauQSplus, "tau");
  VecDuplicate(_tauQSplus,&_psi); PetscObjectSetName((PetscObject) _psi, "psi");
  VecDuplicate(_tauQSplus,&_tempPsi); PetscObjectSetName((PetscObject) _tempPsi, "tempPsi");
  VecDuplicate(_tauQSplus,&_dPsi); PetscObjectSetName((PetscObject) _dPsi, "dPsi");
  VecDuplicate(_tauQSplus,&_uPlus); PetscObjectSetName((PetscObject) _uPlus, "_uPlus");
  VecDuplicate(_tauQSplus,&_velPlus); PetscObjectSetName((PetscObject) _velPlus, "velPlus");

  // set up initial conditions for integration (shallow copy)
  _var.push_back(_psi);
  _var.push_back(_uPlus);


  // frictional fields
  VecDuplicate(_tauQSplus,&_zPlus); PetscObjectSetName((PetscObject) _zPlus, "_zPlus");
  VecDuplicate(_tauQSplus,&_sigma_N); PetscObjectSetName((PetscObject) _sigma_N, "sigma_N");
  VecDuplicate(_tauQSplus,&_a); PetscObjectSetName((PetscObject) _a, "_a");
  VecDuplicate(_tauQSplus,&_b); PetscObjectSetName((PetscObject) _b, "_b");


  setFrictionFields();

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending Fault::Fault in fault.cpp.\n");
#endif
}

Fault::~Fault()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting Fault::~Fault in fault.cpp.\n");
#endif

  // fields that exist on the fault
  VecDestroy(&_tauQSplus);
  VecDestroy(&_psi);
  VecDestroy(&_tempPsi);
  VecDestroy(&_dPsi);
  VecDestroy(&_uPlus);
  VecDestroy(&_velPlus);


  // frictional fields
  VecDestroy(&_zPlus);
  VecDestroy(&_sigma_N);
  VecDestroy(&_a);
  VecDestroy(&_b);


  PetscViewerDestroy(&_uPlusViewer);
  PetscViewerDestroy(&_velPlusViewer);
  PetscViewerDestroy(&_tauQSplusViewer);
  PetscViewerDestroy(&_psiViewer);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending Fault::~Fault in fault.cpp.\n");
#endif
}


PetscErrorCode Fault::setFrictionFields()
{
  PetscErrorCode ierr = 0;
  PetscInt       Ii,Istart,Iend;
  PetscScalar    v;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting Fault::setFrictionFields in fault.cpp\n");CHKERRQ(ierr);
#endif

  // set depth-independent fields
  ierr = VecSet(_psi,_f0);CHKERRQ(ierr);
  ierr = VecCopy(_psi,_tempPsi);CHKERRQ(ierr);
  ierr = VecSet(_a,_aVal);CHKERRQ(ierr);


  // Set b

  PetscScalar r = 0,z = 0;
  PetscScalar rbar = 0.25*_width*_width;
  PetscScalar rw = 1+0.25*_width*_width/_depth/_depth;
  PetscScalar L2 = 1.5*_seisDepth;  // depth below which fault is purely velocity-strengthening
  PetscInt    N1 = _seisDepth/_h;
  PetscInt    N2 = L2/_h;
  ierr = VecGetOwnershipRange(_b,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    z = _h*(Ii-_N*(Ii/_N));
    r=(0.25*_width*_width/_depth/_depth)*z*z;
    if (Ii < N1+1) {
      v = 0.5*(_bAbove-_bBasin)*(tanh((double)(r-rbar)/rw)+1) + _bBasin;
      ierr = VecSetValues(_b,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    else if (Ii>N1 && Ii<=N2) {
      v = (double) (Ii*_h-_seisDepth)*(_bAbove-_bBelow)/(_seisDepth-L2) + _bAbove;
      ierr = VecSetValues(_b,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    else {
      ierr = VecSetValues(_b,1,&Ii,&_bBelow,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_b);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending Fault::setFrictionFields in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}



PetscScalar Fault::getTauInf(PetscInt& ind)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 2
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting Fault::getTauInf in fault.cpp for ind=%i\n",ind);CHKERRQ(ierr);
#endif

  PetscInt       Istart,Iend;
  PetscScalar    a,sigma_N;

  // throw error if value requested is not stored locally
  ierr = VecGetOwnershipRange(_tauQSplus,&Istart,&Iend);CHKERRQ(ierr);
  assert(ind>=Istart && ind<Iend);

  ierr =  VecGetValues(_a,1,&ind,&a);CHKERRQ(ierr);
  ierr =  VecGetValues(_sigma_N,1,&ind,&sigma_N);CHKERRQ(ierr);

#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending Fault::getTauInf in fault.cpp for ind=%i\n",ind);CHKERRQ(ierr);
#endif
  return sigma_N*a*asinh( (double) 0.5*_vp*exp(_f0/a)/_v0 );
}




//================= Functions assuming only + side exists ========================
SymmFault::SymmFault(Domain&D)
: Fault(D)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting SymmFault::SymmFault in fault.cpp.\n");
#endif

  // vectors were allocated in Fault constructor, just need to set values.

  setSplitNodeFields();

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending SymmFault::SymmFault in fault.cpp.\n");
#endif
}

SymmFault::~SymmFault()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting SymmFault::~SymmFault in fault.cpp.\n");
#endif

  // this is covered by the Fault destructor.

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending SymmFault::~SymmFault in fault.cpp.\n");
#endif
}

// assumes right-lateral fault
PetscErrorCode SymmFault::computeVel()
{
  PetscErrorCode ierr = 0;
  Vec            left,right,out;
  Vec            tauQS,eta;
  PetscScalar    outVal,leftVal,rightVal;
  PetscInt       Ii,Istart,Iend;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmFault::computeVel in fault.cpp\n");CHKERRQ(ierr);
#endif

  //~ierr = VecDuplicate(_tauQSplus,&right);CHKERRQ(ierr);
  //~ierr = VecCopy(_tauQSplus,right);CHKERRQ(ierr);
  //~ierr = VecPointwiseDivide(right,right,_zPlus);CHKERRQ(ierr);
  //~ierr = VecScale(right,2.0);CHKERRQ(ierr);
  //~ierr = VecAbs(right);CHKERRQ(ierr);

  // constructing right boundary: right = tauQS/eta
  //   tauQS = tauQSPlus
  //   eta = zPlus/2
  //   -> right = 2*tauQS/zPlus
  ierr = VecDuplicate(_tauQSplus,&tauQS);CHKERRQ(ierr);
  ierr = VecDuplicate(_tauQSplus,&eta);CHKERRQ(ierr);

  ierr = VecCopy(_tauQSplus,tauQS);CHKERRQ(ierr);
  ierr = VecCopy(_zPlus,eta);
  ierr = VecScale(eta,0.5);CHKERRQ(ierr);

  // set up boundaries and output for rootfinder algorithm
  ierr = VecDuplicate(_tauQSplus,&right);CHKERRQ(ierr);
  ierr = VecCopy(tauQS,right);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(right,right,eta);CHKERRQ(ierr);

  ierr = VecDuplicate(right,&left);CHKERRQ(ierr);
  ierr = VecSet(left,0.0);CHKERRQ(ierr);

  ierr = VecDuplicate(left,&out);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(left,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr = VecGetValues(left,1,&Ii,&leftVal);CHKERRQ(ierr);
    ierr = VecGetValues(right,1,&Ii,&rightVal);CHKERRQ(ierr);
    if (abs(leftVal-rightVal)<1e-14) { outVal = leftVal; }
    else {
      Bisect rootAlg(_maxNumIts,_rootTol);
      ierr = rootAlg.setBounds(leftVal,rightVal);CHKERRQ(ierr);
      ierr = rootAlg.findRoot(this,Ii,&outVal);CHKERRQ(ierr);
      _rootIts += rootAlg.getNumIts();
    }
    ierr = VecSetValue(_velPlus,Ii,outVal,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_velPlus);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_velPlus);CHKERRQ(ierr);

  ierr = VecDestroy(&tauQS);CHKERRQ(ierr);
  ierr = VecDestroy(&eta);CHKERRQ(ierr);
  ierr = VecDestroy(&left);CHKERRQ(ierr);
  ierr = VecDestroy(&right);CHKERRQ(ierr);
  ierr = VecDestroy(&out);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmFault::computeVel in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}



// populate fields on the fault
PetscErrorCode SymmFault::setSplitNodeFields()
{
  PetscErrorCode ierr = 0;
  PetscInt       Ii,Istart,Iend;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmFault::setSplitNodeFields in fault.cpp\n");CHKERRQ(ierr);
#endif


  // tau, eta, bcRShift, sigma_N
  PetscScalar a,b,zPlus,tau_inf,sigma_N;

/*  // set normal stress
  PetscScalar z = 0, g=9.8;
  PetscScalar rhoIn = _muArrPlus[0]/(_csArrPlus[0]*_csArrPlus[0]);
  PetscScalar rhoOut = _muArrPlus[_N-1]/(_csArrPlus[_N-1]*_csArrPlus[_N-1]);
  PetscScalar *sigmaArr;
  PetscInt    *sigmaInds;
  ierr = PetscMalloc(_N*sizeof(PetscScalar),&sigmaArr);CHKERRQ(ierr);
  ierr = PetscMalloc(_N*sizeof(PetscInt),&sigmaInds);CHKERRQ(ierr);
  for (Ii=0;Ii<_N;Ii++) {
    sigmaInds[Ii] = Ii;

    z = ((double) Ii)*_h;
    //~// gradient following lithostatic - hydrostatic
    if (Ii<=_depth/_h) {
      sigmaArr[Ii] = rhoIn*g*z - g*z;
    }
    else if (Ii>_depth/_h) {
      sigmaArr[Ii] = rhoOut*g*(z-_depth) + rhoIn*g*_depth - g*z;
    }

    // normal stress is > 0 at Earth's surface
    sigmaArr[Ii] += _sigma_N_min;

    // cap to represent fluid overpressurization (Lapusta and Rice, 2000)
    // (in the paper, the max is 50 MPa)
    sigmaArr[Ii] =(PetscScalar) min((double) sigmaArr[Ii],_sigma_N_max);
  }
  ierr = VecSetValues(_sigma_N,_N,sigmaInds,sigmaArr,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_sigma_N);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_sigma_N);CHKERRQ(ierr);
*/


  ierr = VecGetOwnershipRange(_tauQSplus,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr =  VecGetValues(_a,1,&Ii,&a);CHKERRQ(ierr);
    ierr =  VecGetValues(_b,1,&Ii,&b);CHKERRQ(ierr);
    ierr =  VecGetValues(_sigma_N,1,&Ii,&sigma_N);CHKERRQ(ierr);

    //eta = 0.5*sqrt(_rhoArr[Ii]*_muArr[Ii]);
    zPlus = _muArrPlus[Ii]/_csArrPlus[Ii];

    tau_inf = sigma_N*a*asinh( (double) 0.5*_vp*exp(_f0/a)/_v0 );
    //~bcRshift = tau_inf*_L/_muArrPlus[_sizeMuArr-_N+Ii]; // use last values of muArr

    ierr = VecSetValue(_tauQSplus,Ii,tau_inf,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(_zPlus,Ii,zPlus,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_tauQSplus);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_zPlus);CHKERRQ(ierr);

  ierr = VecAssemblyEnd(_tauQSplus);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_zPlus);CHKERRQ(ierr);

  //~PetscFree(sigmaInds);
  //~PetscFree(sigmaArr);


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmFault::setSplitNodeFields in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode SymmFault::setFaultDisp(Vec const &bcF, Vec const &bcFminus)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmFault::setSymmFaultDisp in fault.cpp\n");CHKERRQ(ierr);
#endif

  ierr = VecCopy(bcF,_uPlus);CHKERRQ(ierr);
  ierr = VecScale(_uPlus,2.0);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmFault::setSymmFault in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode SymmFault::setTauQS(const Vec&sigma_xy,const Vec& sigma_xyMinus)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmFault::setTauQS in lithosphere.cpp.\n");CHKERRQ(ierr);
#endif

  PetscInt       Ii,Istart,Iend;
  PetscScalar    v;

  ierr = VecGetOwnershipRange(sigma_xy,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii<_N) {
      ierr = VecGetValues(sigma_xy,1,&Ii,&v);CHKERRQ(ierr);
      ierr = VecSetValues(_tauQSplus,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_tauQSplus);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_tauQSplus);CHKERRQ(ierr);



#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmFault::setTauQS in lithosphere.c\n");CHKERRQ(ierr);
#endif
  return ierr;
}



PetscErrorCode SymmFault::getResid(const PetscInt ind,const PetscScalar vel,PetscScalar *out)
{
  PetscErrorCode ierr = 0;
  PetscScalar    psi,a,sigma_N,zPlus,tauQS;
  PetscInt       Istart,Iend;

#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmFault::getResid in fault.cpp\n");CHKERRQ(ierr);
#endif

  ierr = VecGetOwnershipRange(_psi,&Istart,&Iend);
  assert(ind>=Istart && ind<Iend);
  ierr = VecGetValues(_tempPsi,1,&ind,&psi);CHKERRQ(ierr);
  ierr = VecGetValues(_a,1,&ind,&a);CHKERRQ(ierr);
  ierr = VecGetValues(_sigma_N,1,&ind,&sigma_N);CHKERRQ(ierr);
  ierr = VecGetValues(_zPlus,1,&ind,&zPlus);CHKERRQ(ierr);
  ierr = VecGetValues(_tauQSplus,1,&ind,&tauQS);CHKERRQ(ierr);

  if (a==0) { *out = 0.5*zPlus*vel - tauQS; }
  else { *out = (PetscScalar) a*sigma_N*asinh( (double) (vel/2./_v0)*exp(psi/a) ) + 0.5*zPlus*vel - tauQS; }
#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"    psi=%g,a=%g,sigma_n=%g,z=%g,tau=%g,vel=%g\n",psi,a,sigma_N,zPlus,tauQS,vel);
#endif
  if (isnan(*out)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isnan(*out) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%g,a=%g,sigma_n=%g,eta=%g,tau=%g,vel=%g\n",psi,a,sigma_N,zPlus,tauQS,vel);
    CHKERRQ(ierr);
  }
  else if (isinf(*out)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isinf(*out) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%g,a=%g,sigma_n=%g,eta=%g,tau=%g,vel=%g\n",psi,a,sigma_N,zPlus,tauQS,vel);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(vel/2/_v0)=%.9e\n",vel/2/_v0);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"exp(psi/a)=%.9e\n",exp(psi/a));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"eta*vel=%.9e\n",zPlus*vel);
    CHKERRQ(ierr);
  }

  assert(!isnan(*out));
  assert(!isinf(*out));

#if VERBOSE > 3
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmFault::getResid in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode SymmFault::agingLaw(const PetscInt ind,const PetscScalar psi,PetscScalar *dPsi)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    b,vel;


  ierr = VecGetOwnershipRange(_psi,&Istart,&Iend);
  assert( ind>=Istart && ind<Iend);
  ierr = VecGetValues(_b,1,&ind,&b);CHKERRQ(ierr);
  ierr = VecGetValues(_velPlus,1,&ind,&vel);CHKERRQ(ierr);

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

  return ierr;
}





PetscErrorCode SymmFault::d_dt(const_it_vec varBegin,const_it_vec varEnd,
                        it_vec dvarBegin,it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmFault::d_dt in fault.cpp\n");CHKERRQ(ierr);
#endif

  PetscScalar    val,psiVal;
  PetscInt       Ii,Istart,Iend;

  assert(varBegin+1 != varEnd);

  ierr = VecCopy(*(varBegin),_tempPsi);CHKERRQ(ierr);
  ierr = computeVel();CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(_velPlus,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr = VecGetValues(*(varBegin),1,&Ii,&psiVal);
    ierr = agingLaw(Ii,psiVal,&val);CHKERRQ(ierr);
    ierr = VecSetValue(*(dvarBegin),Ii,val,INSERT_VALUES);CHKERRQ(ierr);

    ierr = VecGetValues(_velPlus,1,&Ii,&val);CHKERRQ(ierr);
    ierr = VecSetValue(*(dvarBegin+1),Ii,val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(*dvarBegin);CHKERRQ(ierr); ierr = VecAssemblyBegin(*(dvarBegin+1));CHKERRQ(ierr);
  ierr = VecAssemblyEnd(*dvarBegin);CHKERRQ(ierr);   ierr = VecAssemblyEnd(*(dvarBegin+1));CHKERRQ(ierr);

#if VERBOSE > 1
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmFault::d_dt in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode SymmFault::writeContext(const string outputDir)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmFault::writeContext in fault.cpp\n");CHKERRQ(ierr);
#endif

  PetscViewer    viewer;

  std::string str = outputDir + "a";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_a,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "b";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_b,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "zPlus";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_zPlus,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  //~str = outputDir + "sigma_N";
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //~ierr = VecView(_sigma_N,viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

#if VERBOSE > 1
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmFault::writeContext in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode SymmFault::writeStep(const string outputDir,const PetscInt step)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
   ierr = PetscPrintf(PETSC_COMM_WORLD,"starting SymmFault::writeStep in fault.cpp at step %i\n",step);CHKERRQ(ierr);
#endif

  if (step==0) {

      PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"uPlus").c_str(),FILE_MODE_WRITE,&_uPlusViewer);
      ierr = VecView(_uPlus,_uPlusViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_uPlusViewer);CHKERRQ(ierr);


      PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"velPlus").c_str(),FILE_MODE_WRITE,&_velPlusViewer);
      ierr = VecView(_velPlus,_velPlusViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_velPlusViewer);CHKERRQ(ierr);

      PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"tauQSplus").c_str(),FILE_MODE_WRITE,&_tauQSplusViewer);
      ierr = VecView(_tauQSplus,_tauQSplusViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_tauQSplusViewer);CHKERRQ(ierr);

      PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"psi").c_str(),FILE_MODE_WRITE,&_psiViewer);
      ierr = VecView(_psi,_psiViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_psiViewer);CHKERRQ(ierr);

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"uPlus").c_str(),
                                   FILE_MODE_APPEND,&_uPlusViewer);CHKERRQ(ierr);

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"velPlus").c_str(),
                                   FILE_MODE_APPEND,&_velPlusViewer);CHKERRQ(ierr);

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"tauQSplus").c_str(),
                                   FILE_MODE_APPEND,&_tauQSplusViewer);CHKERRQ(ierr);

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"psi").c_str(),
                                   FILE_MODE_APPEND,&_psiViewer);CHKERRQ(ierr);
  }
  else {
    ierr = VecView(_uPlus,_uPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_velPlus,_velPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_tauQSplus,_tauQSplusViewer);CHKERRQ(ierr);
    ierr = VecView(_psi,_psiViewer);CHKERRQ(ierr);
  }

#if VERBOSE > 1
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmFault::writeStep in fault.cpp at step %i\n",step);CHKERRQ(ierr);
#endif
  return ierr;
}











//================= FullFault Functions (both + and - sides) ===========
FullFault::FullFault(Domain&D)
: Fault(D),_zMinus(NULL),_muArrMinus(D._muArrMinus),_csArrMinus(D._csArrMinus),
  _uMinus(NULL),_velMinus(NULL),_vel(NULL)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting FullFault::FullFault in fault.cpp.\n");
#endif

  // allocate space for new fields that exist on left ndoes
  VecDuplicate(_tauQSplus,&_tauQSminus); PetscObjectSetName((PetscObject) _tauQSminus, "tauQSminus");
  VecDuplicate(_tauQSplus,&_uMinus);  PetscObjectSetName((PetscObject) _uMinus, "uMinus");
  VecDuplicate(_tauQSplus,&_velMinus); PetscObjectSetName((PetscObject) _velMinus, "velMinus");
  VecDuplicate(_tauQSplus,&_vel); PetscObjectSetName((PetscObject) _velMinus, "velMinus");
  VecDuplicate(_tauQSplus,&_zMinus); PetscObjectSetName((PetscObject) _zMinus, "zMinus");

  _var.push_back(_uMinus);

  setSplitNodeFields();

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending FullFault::FullFault in fault.cpp.\n");
#endif
}

FullFault::~FullFault()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting FullFault::~FullFault in fault.cpp.\n");
#endif

  // fields that exist on the - side of the split nodes
  VecDestroy(&_tauQSminus);

  VecDestroy(&_zMinus);
  VecDestroy(&_uMinus);
  VecDestroy(&_velMinus);

  PetscViewerDestroy(&_uMinusViewer);
  PetscViewerDestroy(&_velMinusViewer);
  PetscViewerDestroy(&_tauQSminusViewer);


#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending FullFault::~FullFault in fault.cpp.\n");
#endif
}


//==================== protected member functions ======================
// compute vel (vel = velPlus - velMinus), assuming right-lateral fault
PetscErrorCode FullFault::computeVel()
{
  PetscErrorCode ierr = 0;
  Vec            left=NULL,right=NULL,out=NULL;
  Vec            zSum=NULL,temp=NULL,tauQS=NULL,eta=NULL;
  PetscScalar    outVal=0,leftVal=0,rightVal=0;
  PetscInt       Ii,Istart,Iend;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullFault::computeVel in fault.cpp\n");CHKERRQ(ierr);
#endif

  // constructing right boundary: right = tauQS/eta
  // for full fault:
  //   tauQS = [zMinus*tauQSplus + zPlus*tauQSminus]/(zPlus + zMinus)
  //   eta = zPlus*zMinus/(zPlus+zMinus)
  ierr = VecDuplicate(_tauQSplus,&tauQS);CHKERRQ(ierr);
  ierr = VecDuplicate(_tauQSplus,&eta);CHKERRQ(ierr);

  // zSum = zPlus + zMinus
  ierr = VecDuplicate(_zPlus,&zSum);CHKERRQ(ierr);
  ierr = VecCopy(_zPlus,zSum);CHKERRQ(ierr);
  ierr = VecAXPY(zSum,1.0,_zMinus);CHKERRQ(ierr);

  // construct eta
  ierr = VecPointwiseMult(eta,_zPlus,_zMinus);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(eta,eta,zSum);CHKERRQ(ierr);

  // construct tauQS
  ierr = VecPointwiseDivide(tauQS,_zMinus,zSum);CHKERRQ(ierr);
  ierr = VecPointwiseMult(tauQS,tauQS,_tauQSplus);CHKERRQ(ierr);
  ierr = VecDuplicate(_tauQSplus,&temp);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(temp,_zPlus,zSum);CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp,temp,_tauQSminus);CHKERRQ(ierr);
  ierr = VecAXPY(tauQS,1.0,temp);CHKERRQ(ierr); // tauQS = tauQS + temp2
  ierr = VecDestroy(&temp);CHKERRQ(ierr);

  // for symmetric material properties: (here for debugging purposes)
  //   tauQS = tauQSPlus
  //   eta = zPlus/2
  //   -> right = 2*tauQS/zPlus
  //~ierr = VecCopy(_tauQSplus,tauQS);CHKERRQ(ierr);
  //~ierr = VecCopy(_zPlus,eta);
  //~ierr = VecScale(eta,0.5);CHKERRQ(ierr);


  // set up boundaries and output for rootfinder algorithm
  ierr = VecDuplicate(_tauQSplus,&right);CHKERRQ(ierr);
  ierr = VecCopy(tauQS,right);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(right,right,eta);CHKERRQ(ierr);

  ierr = VecDuplicate(right,&left);CHKERRQ(ierr);
  ierr = VecSet(left,0.0);CHKERRQ(ierr);

  ierr = VecDuplicate(left,&out);CHKERRQ(ierr);
  ierr = VecSet(out,0.0);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(left,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr = VecGetValues(left,1,&Ii,&leftVal);CHKERRQ(ierr);
    ierr = VecGetValues(right,1,&Ii,&rightVal);CHKERRQ(ierr);
    if (abs(leftVal-rightVal)<1e-14) { outVal = leftVal; }
    else {
      // construct fresh each time so the boundaries etc are correct
      Bisect rootAlg(_maxNumIts,_rootTol);
      ierr = rootAlg.setBounds(leftVal,rightVal);CHKERRQ(ierr);
      ierr = rootAlg.findRoot(this,Ii,&outVal);CHKERRQ(ierr);
      _rootIts += rootAlg.getNumIts();
    }
    ierr = VecSetValue(_vel,Ii,outVal,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_vel);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_vel);CHKERRQ(ierr);



  // compute velPlus
  // velPlus = (+tauQSplus - tauQSminus + zMinus*vel)/(zPlus + zMinus)
  Vec tauSum=NULL, velCorr=NULL;
  VecDuplicate(_vel,&tauSum);
  VecCopy(_tauQSplus,tauSum);
  VecAXPY(tauSum,-1.0,_tauQSminus);
  VecPointwiseDivide(tauSum,tauSum,zSum);
  VecDuplicate(_vel,&velCorr);
  VecPointwiseDivide(velCorr,_zMinus,zSum);
  VecPointwiseMult(velCorr,velCorr,_vel);
  VecCopy(tauSum,_velPlus);
  VecAXPY(_velPlus,1.0,velCorr);

  // from symmetric fault (here for debugging purposes)
  //~ierr = VecCopy(_vel,_velPlus);CHKERRQ(ierr);
  //~ierr = VecScale(_velPlus,0.5);CHKERRQ(ierr);

  // compute velMinus
  ierr = VecCopy(_velPlus,_velMinus);CHKERRQ(ierr);
  ierr = VecAXPY(_velMinus,-1.0,_vel);CHKERRQ(ierr);


  // clean up memory (this step is a common source of memory leaks)
  ierr = VecDestroy(&tauSum);CHKERRQ(ierr);
  ierr = VecDestroy(&velCorr);CHKERRQ(ierr);
  ierr = VecDestroy(&zSum);CHKERRQ(ierr);
  ierr = VecDestroy(&tauQS);CHKERRQ(ierr);
  ierr = VecDestroy(&eta);CHKERRQ(ierr);
  ierr = VecDestroy(&left);CHKERRQ(ierr);
  ierr = VecDestroy(&right);CHKERRQ(ierr);
  ierr = VecDestroy(&out);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullFault::computeVel in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode FullFault::setSplitNodeFields()
{
  PetscErrorCode ierr = 0;
  PetscInt       Ii,Istart,Iend;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullFault::setSplitNodeFields in fault.cpp\n");CHKERRQ(ierr);
#endif


  // tauQSplus/minus, zPlus/Minus, bcRShift, sigma_N
  PetscScalar a,b,zPlus,zMinus,tau_inf,sigma_N;
  ierr = VecGetOwnershipRange(_tauQSplus,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr =  VecGetValues(_a,1,&Ii,&a);CHKERRQ(ierr);
    ierr =  VecGetValues(_b,1,&Ii,&b);CHKERRQ(ierr);
    ierr =  VecGetValues(_sigma_N,1,&Ii,&sigma_N);CHKERRQ(ierr);

    //~eta = 0.5*sqrt(_rhoArr[Ii]*_muArr[Ii]);
    zPlus = _muArrPlus[Ii]/_csArrPlus[Ii];
    zMinus = _muArrMinus[Ii]/_csArrMinus[Ii];

    tau_inf = sigma_N*a*asinh( (double) 0.5*_vp*exp(_f0/a)/_v0 );

    ierr = VecSetValue(_tauQSplus,Ii,tau_inf,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(_tauQSminus,Ii,tau_inf,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(_zPlus,Ii,zPlus,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(_zMinus,Ii,zMinus,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(_sigma_N,Ii,sigma_N,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(_tauQSplus);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_tauQSminus);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_zPlus);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_zMinus);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_sigma_N);CHKERRQ(ierr);

  ierr = VecAssemblyEnd(_tauQSplus);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_tauQSminus);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_zPlus);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_zMinus);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_sigma_N);CHKERRQ(ierr);



#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullFault::setSplitNodeFields in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


// initialize uPlus and uMinus from lithosphere's data
PetscErrorCode FullFault::setFaultDisp(Vec const &bcLplus,Vec const &bcLminus)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullFault::setFullFaultDisp in fault.cpp\n");CHKERRQ(ierr);
#endif

    ierr = VecCopy(bcLplus,_uPlus);CHKERRQ(ierr);
    ierr = VecCopy(bcLminus,_uMinus);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullFault::setFullFaultDisp in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode FullFault::setTauQS(const Vec& sigma_xyPlus,const Vec& sigma_xyMinus)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullFault::setTauQS in lithosphere.cpp.\n");CHKERRQ(ierr);
#endif

  PetscInt       Ii,Istart,Iend;
  PetscScalar    v;

  ierr = VecGetOwnershipRange(sigma_xyPlus,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii<_N) {
      ierr = VecGetValues(sigma_xyPlus,1,&Ii,&v);CHKERRQ(ierr);
      ierr = VecSetValues(_tauQSplus,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_tauQSplus);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_tauQSplus);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(sigma_xyMinus,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii<_N) {
      ierr = VecGetValues(sigma_xyMinus,1,&Ii,&v);CHKERRQ(ierr);
      ierr = VecSetValues(_tauQSminus,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
ierr = VecAssemblyBegin(_tauQSminus);CHKERRQ(ierr);
ierr = VecAssemblyEnd(_tauQSminus);CHKERRQ(ierr);


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullFault::setTauQS in lithosphere.c\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode FullFault::getResid(const PetscInt ind,const PetscScalar vel,PetscScalar *out)
{
  PetscErrorCode ierr = 0;
  PetscScalar    psi,a,sigma_N,zPlus,zMinus,tauQSplus,tauQSminus;
  PetscInt       Istart,Iend;

#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullFault::getResid in fault.cpp\n");CHKERRQ(ierr);
#endif

  ierr = VecGetOwnershipRange(_psi,&Istart,&Iend);
  assert( ind>=Istart && ind<Iend );

  ierr = VecGetValues(_tempPsi,1,&ind,&psi);CHKERRQ(ierr);
  ierr = VecGetValues(_a,1,&ind,&a);CHKERRQ(ierr);
  ierr = VecGetValues(_sigma_N,1,&ind,&sigma_N);CHKERRQ(ierr);

  ierr = VecGetValues(_zPlus,1,&ind,&zPlus);CHKERRQ(ierr);
  ierr = VecGetValues(_tauQSplus,1,&ind,&tauQSplus);CHKERRQ(ierr);

  ierr = VecGetValues(_tauQSminus,1,&ind,&tauQSminus);CHKERRQ(ierr);
  ierr = VecGetValues(_zMinus,1,&ind,&zMinus);CHKERRQ(ierr);


  *out = (PetscScalar) a*sigma_N*asinh( (double) (vel/2/_v0)*exp(psi/a) )
         - zMinus/(zPlus+zMinus)*tauQSplus
         - zPlus/(zPlus+zMinus)*tauQSminus
         + zPlus*zMinus/(zPlus+zMinus)*vel;

  // from symmetric fault (here for debugging purposes)
  //~*out = (PetscScalar) a*sigma_N*asinh( (double) (vel/2/_v0)*exp(psi/a) ) + 0.5*zPlus*vel - tauQSplus;


#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"    psi=%g,a=%g,sigma_n=%g,zPlus=%g,tau=%g,vel=%g,out=%g\n",psi,a,sigma_N,zPlus,tauQSplus,vel,out);
#endif
  if (isnan(*out)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isnan(*out) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%g,a=%g,sigma_n=%g,eta=%g,tau=%g,vel=%g\n",psi,a,sigma_N,zPlus,tauQSplus,vel);
    CHKERRQ(ierr);
  }
  else if (isinf(*out)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isinf(*out) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%g,a=%g,sigma_n=%g,eta=%g,tau=%g,vel=%g\n",psi,a,sigma_N,zPlus,tauQSplus,vel);
    CHKERRQ(ierr);
  }

  assert(!isnan(*out));
  assert(!isinf(*out));

#if VERBOSE > 3
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullFault::getResid in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}



PetscErrorCode FullFault::agingLaw(const PetscInt ind,const PetscScalar psi,PetscScalar *dPsi)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    b,vel;


  ierr = VecGetOwnershipRange(_psi,&Istart,&Iend);
  assert( ind>=Istart && ind<Iend);
  ierr = VecGetValues(_b,1,&ind,&b);CHKERRQ(ierr);
  ierr = VecGetValues(_vel,1,&ind,&vel);CHKERRQ(ierr);

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

  return ierr;
}



PetscErrorCode FullFault::d_dt(const_it_vec varBegin,const_it_vec varEnd,
                               it_vec dvarBegin,it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 3
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullFault::d_dt in fault.cpp\n");CHKERRQ(ierr);
#endif

  PetscScalar    val,psiVal;
  PetscInt       Ii,Istart,Iend;

  assert(varBegin+1 != varEnd);

  ierr = VecCopy(*(varBegin),_tempPsi);CHKERRQ(ierr);
  ierr = computeVel();CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(_vel,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr = VecGetValues(*varBegin,1,&Ii,&psiVal);
    ierr = agingLaw(Ii,psiVal,&val);CHKERRQ(ierr);
    ierr = VecSetValue(*dvarBegin,Ii,val,INSERT_VALUES);CHKERRQ(ierr);

    ierr = VecGetValues(_velPlus,1,&Ii,&val);CHKERRQ(ierr);
    ierr = VecSetValue(*(dvarBegin+1),Ii,val,INSERT_VALUES);CHKERRQ(ierr);

    ierr = VecGetValues(_velMinus,1,&Ii,&val);CHKERRQ(ierr);
    ierr = VecSetValue(*(dvarBegin+2),Ii,val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(*dvarBegin);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(*(dvarBegin+1));CHKERRQ(ierr);
  ierr = VecAssemblyBegin(*(dvarBegin+2));CHKERRQ(ierr);

  ierr = VecAssemblyEnd(*dvarBegin);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(*(dvarBegin+1));CHKERRQ(ierr);
  ierr = VecAssemblyEnd(*(dvarBegin+2));CHKERRQ(ierr);

  return ierr;
#if VERBOSE > 3
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullFault::d_dt in fault.cpp\n");CHKERRQ(ierr);
#endif
}


PetscErrorCode FullFault::writeContext(const string outputDir)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullFault::writeContext in fault.cpp\n");CHKERRQ(ierr);
#endif

 //~ierr = PetscViewerHDF5PushGroup(viewer, "/frictionContext");CHKERRQ(ierr);
//~
  //~ierr = VecView(a, viewer);CHKERRQ(ierr);
  //~ierr = VecView(b, viewer);CHKERRQ(ierr);
  //~ierr = VecView(eta, viewer);CHKERRQ(ierr);
  //~ierr = VecView(sigma_N, viewer);CHKERRQ(ierr);
  //~ierr = MatView(mu, viewer);CHKERRQ(ierr);
//~
//~
  //~ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);

  PetscViewer    viewer;

  std::string str = outputDir + "a";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_a,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "b";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_b,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "zPlus";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_zPlus,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

    str = outputDir + "zMinus";
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(_zMinus,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  //~str = outputDir + "sigma_N";
  //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //~ierr = VecView(_sigma_N,viewer);CHKERRQ(ierr);
  //~ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

#if VERBOSE > 1
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullFault::writeContext in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode FullFault::writeStep(const string outputDir,const PetscInt step)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
   ierr = PetscPrintf(PETSC_COMM_WORLD,"starting FullFault::writeStep in fault.cpp at step %i\n",step);CHKERRQ(ierr);
#endif

  if (step==0) {

      PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"uPlus").c_str(),FILE_MODE_WRITE,&_uPlusViewer);
      ierr = VecView(_uPlus,_uPlusViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_uPlusViewer);CHKERRQ(ierr);

      PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"velPlus").c_str(),FILE_MODE_WRITE,&_velPlusViewer);
      ierr = VecView(_velPlus,_velPlusViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_velPlusViewer);CHKERRQ(ierr);

      PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"tauQSplus").c_str(),FILE_MODE_WRITE,&_tauQSplusViewer);
      ierr = VecView(_tauQSplus,_tauQSplusViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_tauQSplusViewer);CHKERRQ(ierr);

      PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"psi").c_str(),FILE_MODE_WRITE,&_psiViewer);
      ierr = VecView(_psi,_psiViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_psiViewer);CHKERRQ(ierr);

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"uPlus").c_str(),
                                   FILE_MODE_APPEND,&_uPlusViewer);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"velPlus").c_str(),
                                   FILE_MODE_APPEND,&_velPlusViewer);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"tauQSplus").c_str(),
                                   FILE_MODE_APPEND,&_tauQSplusViewer);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"psi").c_str(),
                                   FILE_MODE_APPEND,&_psiViewer);CHKERRQ(ierr);

        PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"uMinus").c_str(),FILE_MODE_WRITE,&_uMinusViewer);
        ierr = VecView(_uMinus,_uMinusViewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&_uMinusViewer);CHKERRQ(ierr);

        PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"tauQSminus").c_str(),FILE_MODE_WRITE,&_tauQSminusViewer);
        ierr = VecView(_tauQSminus,_tauQSminusViewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&_tauQSminusViewer);CHKERRQ(ierr);

        PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"velMinus").c_str(),FILE_MODE_WRITE,&_velMinusViewer);
        ierr = VecView(_velMinus,_velMinusViewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&_velMinusViewer);CHKERRQ(ierr);

        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"uMinus").c_str(),
                                   FILE_MODE_APPEND,&_uMinusViewer);CHKERRQ(ierr);
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"velMinus").c_str(),
                                   FILE_MODE_APPEND,&_velMinusViewer);CHKERRQ(ierr);
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"tauQSminus").c_str(),
                                   FILE_MODE_APPEND,&_tauQSminusViewer);CHKERRQ(ierr);
  }
  else {
    ierr = VecView(_uPlus,_uPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_velPlus,_velPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_tauQSplus,_tauQSplusViewer);CHKERRQ(ierr);
    ierr = VecView(_psi,_psiViewer);CHKERRQ(ierr);

      ierr = VecView(_uMinus,_uMinusViewer);CHKERRQ(ierr);
      ierr = VecView(_velMinus,_velMinusViewer);CHKERRQ(ierr);
      ierr = VecView(_tauQSminus,_tauQSminusViewer);CHKERRQ(ierr);
  }

#if VERBOSE > 1
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullFault::writeStep in fault.cpp at step %i\n",step);CHKERRQ(ierr);
#endif
  return ierr;
}



