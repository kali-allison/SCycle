#include "fault.hpp"

using namespace std;




Fault::Fault(Domain&D)
: _file(D._file),_delim(D._delim),
  _N(D._Nz),_sizeMuArr(D._Ny*D._Nz),_L(D._Lz),_h(D._dz),_z(NULL),
  _problemType(D._problemType),
  _depth(D._depth),_width(D._width),
  _rootTol(D._rootTol),_rootIts(0),_maxNumIts(1e8),
  _f0(D._f0),_v0(D._v0),_vL(D._vL),
  _a(NULL),_b(NULL),_Dc(NULL),
  _state(NULL),_dPsi(NULL),
  _sigma_N(NULL),
  _muVecP(&D._muVecP),_csVecP(&D._csVecP),
  _slip(NULL),_slipVel(NULL),
  _slipViewer(NULL),_slipVelViewer(NULL),_tauQSPlusViewer(NULL),
  _stateViewer(NULL),
  _tauQSP(NULL)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting Fault::Fault in fault.cpp.\n");
#endif

  // set a, b, normal stress, and Dc
  loadSettings(_file);
  checkInput();



  // fields that exist on the fault
  VecCreate(PETSC_COMM_WORLD,&_tauQSP);
  VecSetSizes(_tauQSP,PETSC_DECIDE,_N);
  VecSetFromOptions(_tauQSP);     PetscObjectSetName((PetscObject) _tauQSP, "tau");  VecSet(_tauQSP,0.0);
  VecDuplicate(_tauQSP,&_state); PetscObjectSetName((PetscObject) _state, "psi"); VecSet(_state,0.0);
  VecDuplicate(_tauQSP,&_dPsi); PetscObjectSetName((PetscObject) _dPsi, "dPsi"); VecSet(_dPsi,0.0);
  VecDuplicate(_tauQSP,&_slip); PetscObjectSetName((PetscObject) _slip, "_slip"); VecSet(_slip,0.0);
  VecDuplicate(_tauQSP,&_slipVel); PetscObjectSetName((PetscObject) _slipVel, "_slipVel");
  VecSet(_slipVel,0.0);


  // frictional fields
  VecDuplicate(_tauQSP,&_Dc); PetscObjectSetName((PetscObject) _Dc, "_Dc");
  VecDuplicate(_tauQSP,&_sigma_N); PetscObjectSetName((PetscObject) _sigma_N, "_sigma_N");
  VecDuplicate(_tauQSP,&_zP); PetscObjectSetName((PetscObject) _zP, "_zP");
  VecDuplicate(_tauQSP,&_a); PetscObjectSetName((PetscObject) _a, "_a");
  VecDuplicate(_tauQSP,&_b); PetscObjectSetName((PetscObject) _b, "_b");

  // initialize _z
  VecDuplicate(_tauQSP,&_z);
  PetscInt    Istart,Iend;
  VecGetOwnershipRange(_tauQSP,&Istart,&Iend);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    PetscScalar z = Ii-_N*(Ii/_N);
    if (Ii < _N) {
      VecGetValues(D._z,1,&Ii,&z);
      VecSetValue(_z,Ii,z,INSERT_VALUES);
    }
  }
  VecAssemblyBegin(_z);
  VecAssemblyEnd(_z);

  setFrictionFields(D);


#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending Fault::Fault in fault.cpp.\n");
#endif
}



// Check that required fields have been set by the input file
PetscErrorCode Fault::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting Domain::checkInputPlus in domain.cpp.\n");CHKERRQ(ierr);
  #endif

  assert(_DcVals.size() == _DcDepths.size() );
  assert(_aVals.size() == _aDepths.size() );
  assert(_bVals.size() == _bDepths.size() );
  assert(_sigmaNVals.size() == _sigmaNDepths.size() );


#if VERBOSE > 1
ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending Domain::checkInputPlus in domain.cpp.\n");CHKERRQ(ierr);
#endif
  //~}
  return ierr;
}


Fault::~Fault()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting Fault::~Fault in fault.cpp.\n");
#endif

  // fields that exist on the fault
  VecDestroy(&_tauQSP);
  VecDestroy(&_state);
  VecDestroy(&_dPsi);
  VecDestroy(&_slip);
  VecDestroy(&_slipVel);


  // frictional fields
  VecDestroy(&_Dc);
  VecDestroy(&_zP);
  VecDestroy(&_a);
  VecDestroy(&_b);
  VecDestroy(&_sigma_N);


  PetscViewerDestroy(&_slipViewer);
  PetscViewerDestroy(&_slipVelViewer);
  PetscViewerDestroy(&_tauQSPlusViewer);
  PetscViewerDestroy(&_stateViewer);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending Fault::~Fault in fault.cpp.\n");
#endif
}

// Fills vec with the linear interpolation between the pairs of points (vals,depths).
PetscErrorCode Fault::setVecFromVectors(Vec& vec, vector<double>& vals,vector<double>& depths)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    v,z,z0,z1,v0,v1;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting Fault::setVecFromVectors in fault.cpp\n");CHKERRQ(ierr);
#endif

// build structure from generalized input
  size_t vecLen = depths.size();
  ierr = VecGetOwnershipRange(vec,&Istart,&Iend);CHKERRQ(ierr);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++)
  {
    //~ z = _h*(Ii-_N*(Ii/_N));
    //~ PetscScalar z2 = 0;
    VecGetValues(_z,1,&Ii,&z);CHKERRQ(ierr);
    //~ PetscPrintf(PETSC_COMM_WORLD,"%i: z = %g, z2 = %g\n",Ii,z,z2);
    for (size_t ind = 0; ind < vecLen-1; ind++) {
      z0 = depths[0+ind];
      z1 = depths[0+ind+1];
      v0 = vals[0+ind];
      v1 = vals[0+ind+1];
      if (z>=z0 && z<=z1) { v = (v1 - v0)/(z1-z0) * (z-z0) + v0; }
      ierr = VecSetValues(vec,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);



#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending Fault::setVecFromVectors in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode Fault::setFrictionFields(Domain&D)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting Fault::setFrictionFields in fault.cpp\n");CHKERRQ(ierr);
#endif

  // set depth-independent fields
  #if STATE_PSI == 1
    ierr = VecSet(_state,_f0);CHKERRQ(ierr); // in terms of psi
  #endif
  #if STATE_PSI == 0
    ierr = VecSet(_state,1e9);CHKERRQ(ierr);
  #endif

  // set a using a vals
  if (_N == 1) {
    VecSet(_b,_bVals[0]);
    VecSet(_a,_aVals[0]);
    VecSet(_sigma_N,_sigmaNVals[0]);
    VecSet(_Dc,_DcVals[0]);
  }
  else {
    ierr = setVecFromVectors(_a,_aVals,_aDepths);CHKERRQ(ierr);
    ierr = setVecFromVectors(_b,_bVals,_bDepths);CHKERRQ(ierr);
    ierr = setVecFromVectors(_sigma_N,_sigmaNVals,_sigmaNDepths);CHKERRQ(ierr);
    ierr = setVecFromVectors(_Dc,_DcVals,_DcDepths);CHKERRQ(ierr);
  }


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
  ierr = VecGetOwnershipRange(_tauQSP,&Istart,&Iend);CHKERRQ(ierr);
  assert(ind>=Istart && ind<Iend);

  ierr =  VecGetValues(_a,1,&ind,&a);CHKERRQ(ierr);
  ierr =  VecGetValues(_sigma_N,1,&ind,&sigma_N);CHKERRQ(ierr);

#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending Fault::getTauInf in fault.cpp for ind=%i\n",ind);CHKERRQ(ierr);
#endif
  return sigma_N*a*asinh( (double) 0.5*_vL*exp(_f0/a)/_v0 );
}


PetscErrorCode Fault::agingLaw(const PetscInt ind,const PetscScalar state,PetscScalar &dstate)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    b,slipVel,Dc;


  ierr = VecGetOwnershipRange(_state,&Istart,&Iend);
  assert( ind>=Istart && ind<Iend);
  ierr = VecGetValues(_Dc,1,&ind,&Dc);CHKERRQ(ierr);
  ierr = VecGetValues(_b,1,&ind,&b);CHKERRQ(ierr);
  ierr = VecGetValues(_slipVel,1,&ind,&slipVel);CHKERRQ(ierr);
  slipVel = abs(slipVel); // state evolution is not sensitive to direction of slip


  // if in terms of theta
  #if STATE_PSI == 0
    dstate = 1 - state*slipVel/Dc;
  #endif

  // if in terms of psi
  #if STATE_PSI == 1
    if ( isinf(exp((_f0-state)/b)) ) { dstate = 0; } // new criteria
    //~if ( isinf(exp(1.0/b)) ) { dstate = 0; } // old criteria
    else if ( b <= 1e-3 ) { dstate = 0; }
    else {
      dstate = (PetscScalar) (b*_v0/Dc)*( exp((double) ( (_f0-state)/b) ) - (slipVel/_v0) );
    }

    if (isnan(dstate)) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"isnan(dstate) evaluated to true\n");
      ierr = PetscPrintf(PETSC_COMM_WORLD,"state=%.9e,b=%.9e,f0=%.9e,D_c=%.9e,v0=%.9e,vel=%.9e\n",state,b,_f0,Dc,_v0,slipVel);
      CHKERRQ(ierr);
    }
    else if (isinf(dstate)) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"isinf(*dPsi) evaluated to true\n");
      ierr = PetscPrintf(PETSC_COMM_WORLD,"state=%.9e,b=%.9e,f0=%.9e,D_c=%.9e,v0=%.9e,vel=%.9e\n",state,b,_f0,Dc,_v0,slipVel);
      CHKERRQ(ierr);
    }
  #endif
  assert(!isnan(dstate));
  assert(!isinf(dstate));

  return ierr;
}

PetscErrorCode Fault::slipLaw(const PetscInt ind,const PetscScalar state,PetscScalar &dstate)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    b,slipVel,Dc;


  ierr = VecGetOwnershipRange(_state,&Istart,&Iend);
  assert( ind>=Istart && ind<Iend);
  ierr = VecGetValues(_Dc,1,&ind,&Dc);CHKERRQ(ierr);
  ierr = VecGetValues(_b,1,&ind,&b);CHKERRQ(ierr);
  ierr = VecGetValues(_slipVel,1,&ind,&slipVel);CHKERRQ(ierr);
  slipVel = abs(slipVel); // state evolution is not sensitive to direction of slip

    //~PetscScalar fss = _f0 + log(slipVel/_v0);

  // if in terms of theta
  #if STATE_PSI == 0
    PetscScalar A = state*slipVel/Dc;
    dstate = -A*log(A);
  #endif
  #if STATE_PSI == 1
    PetscPrintf(PETSC_COMM_WORLD,"WARNING: Fault::slipLaw not written for state variable psi!\n\n");
    assert(0);
  #endif
  if (isnan(dstate)) {
    PetscPrintf(PETSC_COMM_WORLD,"state = %e, slipVel=%e,Dc = %e\n",state,slipVel,Dc);
  }
  assert(!isnan(dstate));
  assert(!isinf(dstate));

  /* // if in terms of psi
  if ( isinf(exp(1/b)) ) { *dPsi = 0; }
  else if ( b <= 1e-3 ) { *dPsi = 0; }
  else {
    *dPsi = (PetscScalar) (b*_v0/Dc)*( exp((double) ( (_f0-psi)/b) ) - (slipVel/_v0) );
  }
  assert(!isnan(*dPsi));
  assert(!isinf(*dPsi));
  */

  return ierr;
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
  PetscScalar    outVal,leftVal,rightVal,temp;
  PetscInt       Ii,Istart,Iend;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmFault::computeVel in fault.cpp\n");CHKERRQ(ierr);
#endif

  //~ierr = VecDuplicate(_tauQSP,&right);CHKERRQ(ierr);
  //~ierr = VecCopy(_tauQSP,right);CHKERRQ(ierr);
  //~ierr = VecPointwiseDivide(right,right,_zP);CHKERRQ(ierr);
  //~ierr = VecScale(right,2.0);CHKERRQ(ierr);
  //~ierr = VecAbs(right);CHKERRQ(ierr);

  // constructing right boundary: right = tauQS/eta
  //   tauQS = tauQSPlus
  //   eta = zPlus/2
  //   -> right = 2*tauQS/zPlus
  ierr = VecDuplicate(_tauQSP,&tauQS);CHKERRQ(ierr);
  ierr = VecDuplicate(_tauQSP,&eta);CHKERRQ(ierr);

  ierr = VecCopy(_tauQSP,tauQS);CHKERRQ(ierr);
  ierr = VecCopy(_zP,eta);
  ierr = VecScale(eta,0.5);CHKERRQ(ierr);

  // set up boundaries and output for rootfinder algorithm
  ierr = VecDuplicate(_tauQSP,&right);CHKERRQ(ierr);
  ierr = VecCopy(tauQS,right);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(right,right,eta);CHKERRQ(ierr);

  ierr = VecDuplicate(right,&left);CHKERRQ(ierr);
  ierr = VecSet(left,0.0);CHKERRQ(ierr);

  ierr = VecDuplicate(left,&out);CHKERRQ(ierr);


  ierr = VecGetOwnershipRange(left,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr = VecGetValues(left,1,&Ii,&leftVal);CHKERRQ(ierr);
    ierr = VecGetValues(right,1,&Ii,&rightVal);CHKERRQ(ierr);

    if (isnan(leftVal) || isnan(rightVal)) {
      PetscPrintf(PETSC_COMM_WORLD,"\n\nError:left or right evaluated to nan.\n");
      assert(0);
    }
    // correct for left-lateral fault motion
    if (leftVal>rightVal) {
      temp = leftVal;
      rightVal = leftVal;
      leftVal = temp;
    }

    if (abs(leftVal-rightVal)<1e-14) { outVal = leftVal; }
    else {
      Bisect rootAlg(_maxNumIts,_rootTol);
      ierr = rootAlg.setBounds(leftVal,rightVal);CHKERRQ(ierr);
      ierr = rootAlg.findRoot(this,Ii,&outVal);CHKERRQ(ierr);
      _rootIts += rootAlg.getNumIts();
    }
    ierr = VecSetValue(_slipVel,Ii,outVal,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_slipVel);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_slipVel);CHKERRQ(ierr);


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

  // create properly sized vectors for mu and cs
  Vec muV; VecDuplicate(_a,&muV);
  Vec csV; VecDuplicate(_a,&csV);
  PetscScalar mu,cs;
  ierr = VecGetOwnershipRange(*_muVecP,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii < _N) {
    ierr =  VecGetValues(*_muVecP,1,&Ii,&mu);CHKERRQ(ierr);
    ierr =  VecGetValues(*_csVecP,1,&Ii,&cs);CHKERRQ(ierr);

    ierr = VecSetValue(muV,Ii,mu,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(csV,Ii,cs,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(muV);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(csV);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(muV);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(csV);CHKERRQ(ierr);



  // tau, eta, bcRShift, sigma_N
  PetscScalar a,b,zPlus,tau_inf,sigma_N;
  ierr = VecGetOwnershipRange(_a,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr =  VecGetValues(_a,1,&Ii,&a);CHKERRQ(ierr);
    ierr =  VecGetValues(_b,1,&Ii,&b);CHKERRQ(ierr);
    ierr =  VecGetValues(_sigma_N,1,&Ii,&sigma_N);CHKERRQ(ierr);

    //eta = 0.5*sqrt(_rhoArr[Ii]*_muArr[Ii]);
    //~ zPlus = _muArrPlus[Ii]/_csArrPlus[Ii];
    ierr =  VecGetValues(muV,1,&Ii,&mu);CHKERRQ(ierr);
    ierr =  VecGetValues(csV,1,&Ii,&cs);CHKERRQ(ierr);
    zPlus = mu/cs;

    tau_inf = sigma_N*a*asinh( (double) 0.5*_vL*exp(_f0/a)/_v0 );

    ierr = VecSetValue(_tauQSP,Ii,tau_inf,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(_zP,Ii,zPlus,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_tauQSP);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_zP);CHKERRQ(ierr);

  ierr = VecAssemblyEnd(_tauQSP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_zP);CHKERRQ(ierr);

  //~ VecView(*_csVecP,PETSC_VIEWER_STDOUT_WORLD);
  //~ VecView(*_muVecP,PETSC_VIEWER_STDOUT_WORLD);
  //~ VecView(*_csVecP,PETSC_VIEWER_STDOUT_WORLD);
  //~ VecView(_zP,PETSC_VIEWER_STDOUT_WORLD);
  //~ VecView(_tauQSP,PETSC_VIEWER_STDOUT_WORLD);
  //~ assert(0);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmFault::setSplitNodeFields in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

// what is this function for??
PetscErrorCode SymmFault::setFaultDisp(Vec const &bcF, Vec const &bcFminus)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmFault::setSymmFaultDisp in fault.cpp\n");CHKERRQ(ierr);
#endif

  // bcF holds displacement at y=0+
  ierr = VecCopy(bcF,_slip);CHKERRQ(ierr);
  ierr = VecScale(_slip,2.0);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmFault::setSymmFault in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode SymmFault::setTauQS(const Vec&sigma_xyPlus,const Vec& sigma_xyMinus)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmFault::setTauQS in lithosphere.cpp.\n");CHKERRQ(ierr);
#endif

  PetscInt       Ii,Istart,Iend;
  PetscScalar    v;

  ierr = VecGetOwnershipRange(sigma_xyPlus,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii<_N) {
      ierr = VecGetValues(sigma_xyPlus,1,&Ii,&v);CHKERRQ(ierr);
      ierr = VecSetValues(_tauQSP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_tauQSP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_tauQSP);CHKERRQ(ierr);


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmFault::setTauQS in lithosphere.c\n");CHKERRQ(ierr);
#endif
  return ierr;
}



PetscErrorCode SymmFault::getResid(const PetscInt ind,const PetscScalar slipVel,PetscScalar *out)
{
  PetscErrorCode ierr = 0;
  PetscScalar    state,a,sigma_N,zPlus,tauQS;
  PetscInt       Istart,Iend;

#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmFault::getResid in fault.cpp\n");CHKERRQ(ierr);
#endif

  ierr = VecGetOwnershipRange(_state,&Istart,&Iend);
  assert(ind>=Istart && ind<Iend);
  ierr = VecGetValues(_state,1,&ind,&state);CHKERRQ(ierr);
  ierr = VecGetValues(_a,1,&ind,&a);CHKERRQ(ierr);
  ierr = VecGetValues(_sigma_N,1,&ind,&sigma_N);CHKERRQ(ierr);
  ierr = VecGetValues(_zP,1,&ind,&zPlus);CHKERRQ(ierr);
  ierr = VecGetValues(_tauQSP,1,&ind,&tauQS);CHKERRQ(ierr);

  // frictional strength of fault
  // in terms of psi
  #if STATE_PSI == 1
    PetscScalar strength = (PetscScalar) a*sigma_N*asinh( (double) (slipVel/2./_v0)*exp(state/a) );
  #endif

  // in terms of theta
  #if STATE_PSI == 0
    PetscScalar b,Dc=0;
    ierr = VecGetValues(_b,1,&ind,&b);CHKERRQ(ierr);
    ierr = VecGetValues(_Dc,1,&ind,&Dc);CHKERRQ(ierr);
    PetscScalar psi = _f0 + b*log(state*_v0/Dc);
    PetscScalar strength = (PetscScalar) a*sigma_N*asinh( (double) (slipVel/2./_v0)*exp(psi/a) );
  #endif

  // stress on fault
  PetscScalar stress = tauQS - 0.5*zPlus*slipVel;

  *out = strength - stress;

#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"    psi=%g,a=%g,sigma_n=%g,z=%g,tau=%g,vel=%g\n",state,a,sigma_N,zPlus,tauQS,slipVel);
#endif
  if (isnan(*out)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isnan(*out) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%g,a=%g,sigma_n=%g,z=%g,tau=%g,vel=%g\n",state,a,sigma_N,zPlus,tauQS,slipVel);
    CHKERRQ(ierr);
  }
  else if (isinf(*out)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isinf(*out) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%g,a=%g,sigma_n=%g,z=%g,tau=%g,vel=%g\n",state,a,sigma_N,zPlus,tauQS,slipVel);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(vel/2/_v0)=%.9e\n",slipVel/2/_v0);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"exp(psi/a)=%.9e\n",exp(state/a));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"z*vel=%.9e\n",zPlus*slipVel);
    CHKERRQ(ierr);
  }

  assert(!isnan(*out));
  assert(!isinf(*out));

#if VERBOSE > 3
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmFault::getResid in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}





PetscErrorCode SymmFault::d_dt(const_it_vec varBegin,const_it_vec varEnd,
                        it_vec dvarBegin,it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmFault::d_dt in fault.cpp\n");CHKERRQ(ierr);
#endif

  PetscScalar    val,stateVal;
  PetscInt       Ii,Istart,Iend;

  assert(varBegin+1 != varEnd);

  ierr = VecCopy(*(varBegin),_state);CHKERRQ(ierr);
  ierr = VecCopy(*(varBegin+1),_slip);CHKERRQ(ierr);
  ierr = computeVel();CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(_slipVel,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr = VecGetValues(*(varBegin),1,&Ii,&stateVal);
    ierr = agingLaw(Ii,stateVal,val);CHKERRQ(ierr);
    //~ierr = slipLaw(Ii,stateVal,val);CHKERRQ(ierr);
    ierr = VecSetValue(*(dvarBegin),Ii,val,INSERT_VALUES);CHKERRQ(ierr);

    ierr = VecGetValues(_slipVel,1,&Ii,&val);CHKERRQ(ierr);
    ierr = VecSetValue(*(dvarBegin+1),Ii,val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(*dvarBegin);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(*(dvarBegin+1));CHKERRQ(ierr);

  ierr = VecAssemblyEnd(*dvarBegin);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(*(dvarBegin+1));CHKERRQ(ierr);

  // force fault to remain locked
  //~ierr = VecSet(*dvarBegin,0.0);CHKERRQ(ierr);
  //~ierr = VecSet(*(dvarBegin+1),0.0);CHKERRQ(ierr);


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
  ierr = VecView(_zP,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // output normal stress vector
  str =  outputDir + "sigma_N";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_sigma_N,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // output critical distance
  str =  outputDir + "Dc";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_Dc,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);


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
      PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"slip").c_str(),FILE_MODE_WRITE,&_slipViewer);
      ierr = VecView(_slip,_slipViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_slipViewer);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"slip").c_str(),
                                   FILE_MODE_APPEND,&_slipViewer);CHKERRQ(ierr);

      PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"slipVel").c_str(),FILE_MODE_WRITE,&_slipVelViewer);
      ierr = VecView(_slipVel,_slipVelViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_slipVelViewer);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"slipVel").c_str(),
                                   FILE_MODE_APPEND,&_slipVelViewer);CHKERRQ(ierr);

      PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"tauQSPlus").c_str(),FILE_MODE_WRITE,&_tauQSPlusViewer);
      ierr = VecView(_tauQSP,_tauQSPlusViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_tauQSPlusViewer);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"tauQSPlus").c_str(),
                                   FILE_MODE_APPEND,&_tauQSPlusViewer);CHKERRQ(ierr);

      PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"state").c_str(),FILE_MODE_WRITE,&_stateViewer);
      ierr = VecView(_state,_stateViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_stateViewer);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"state").c_str(),
                                   FILE_MODE_APPEND,&_stateViewer);CHKERRQ(ierr);
  }
  else {
    ierr = VecView(_slip,_slipViewer);CHKERRQ(ierr);
    ierr = VecView(_slipVel,_slipVelViewer);CHKERRQ(ierr);
    ierr = VecView(_tauQSP,_tauQSPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_state,_stateViewer);CHKERRQ(ierr);
  }

#if VERBOSE > 1
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmFault::writeStep in fault.cpp at step %i\n",step);CHKERRQ(ierr);
#endif
  return ierr;
}











//================= FullFault Functions (both + and - sides) ===========
FullFault::FullFault(Domain&D)
: Fault(D),_zM(NULL),_muArrMinus(D._muArrMinus),_csArrMinus(D._csArrMinus),
  _arrSize(D._Ny*D._Nz),
  _uP(NULL),_uM(NULL),_velPlus(NULL),_velMinus(NULL),
  _uPlusViewer(NULL),_uMV(NULL),_velPlusViewer(NULL),_velMinusViewer(NULL),
 _tauQSMinusViewer(NULL),_stateViewer(NULL),_tauQSMinus(NULL)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting FullFault::FullFault in fault.cpp.\n");
#endif

  // allocate space for new fields that exist on left ndoes
  VecDuplicate(_tauQSP,&_tauQSMinus); PetscObjectSetName((PetscObject) _tauQSMinus, "tauQSminus");
  VecDuplicate(_tauQSP,&_uP);  PetscObjectSetName((PetscObject) _uP, "uPlus");
  VecDuplicate(_tauQSP,&_uM);  PetscObjectSetName((PetscObject) _uM, "uMinus");
  VecDuplicate(_tauQSP,&_velMinus); PetscObjectSetName((PetscObject) _velMinus, "velMinus");
  VecDuplicate(_tauQSP,&_velPlus); PetscObjectSetName((PetscObject) _velPlus, "velPlus");
  VecDuplicate(_tauQSP,&_zM); PetscObjectSetName((PetscObject) _zM, "zMinus");

  setSplitNodeFields();

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending FullFault::FullFault in fault.cpp.\n");
#endif
}




PetscErrorCode Fault::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting loadData in domain.cpp, loading from file: %s.\n", file);CHKERRQ(ierr);
#endif
  PetscMPIInt rank,size;
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);


  ifstream infile( file );
  string line,var;
  size_t pos = 0;
  while (getline(infile, line))
  {
    istringstream iss(line);
    pos = line.find(_delim); // find position of the delimiter
    var = line.substr(0,pos);

    if (var.compare("DcVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_DcVals);
    }
    else if (var.compare("DcDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_DcDepths);
    }

    else if (var.compare("sigmaNVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_sigmaNVals);
    }
    else if (var.compare("sigmaNDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_sigmaNDepths);
    }

    else if (var.compare("aVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_aVals);
    }
    else if (var.compare("aDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_aDepths);
    }
    else if (var.compare("bVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_bVals);
    }
    else if (var.compare("bDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_bDepths);
    }
  }

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending loadData in domain.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}




FullFault::~FullFault()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting FullFault::~FullFault in fault.cpp.\n");
#endif

  // fields that exist on the - side of the split nodes
  VecDestroy(&_tauQSMinus);

  VecDestroy(&_zM);
  VecDestroy(&_uM);
  VecDestroy(&_uP);
  VecDestroy(&_velMinus);
  VecDestroy(&_velPlus);

  PetscViewerDestroy(&_uPlusViewer);
  PetscViewerDestroy(&_uMV);
  PetscViewerDestroy(&_velMinusViewer);
  PetscViewerDestroy(&_velPlusViewer);
  PetscViewerDestroy(&_tauQSMinusViewer);


#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending FullFault::~FullFault in fault.cpp.\n");
#endif
}


//==================== protected member functions ======================
// compute slipVel ( = velPlus - velMinus), assuming right-lateral fault
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
  ierr = VecDuplicate(_tauQSP,&tauQS);CHKERRQ(ierr);
  ierr = VecDuplicate(_tauQSP,&eta);CHKERRQ(ierr);

  // zSum = zPlus + zMinus
  ierr = VecDuplicate(_zP,&zSum);CHKERRQ(ierr);
  ierr = VecCopy(_zP,zSum);CHKERRQ(ierr);
  ierr = VecAXPY(zSum,1.0,_zM);CHKERRQ(ierr);

  // construct eta
  ierr = VecPointwiseMult(eta,_zP,_zM);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(eta,eta,zSum);CHKERRQ(ierr);

  // construct tauQS
  ierr = VecPointwiseDivide(tauQS,_zM,zSum);CHKERRQ(ierr);
  ierr = VecPointwiseMult(tauQS,tauQS,_tauQSP);CHKERRQ(ierr);
  ierr = VecDuplicate(_tauQSP,&temp);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(temp,_zP,zSum);CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp,temp,_tauQSMinus);CHKERRQ(ierr);
  ierr = VecAXPY(tauQS,1.0,temp);CHKERRQ(ierr); // tauQS = tauQS + temp2
  ierr = VecDestroy(&temp);CHKERRQ(ierr);


  // set up boundaries and output for rootfinder algorithm
  ierr = VecDuplicate(_tauQSP,&right);CHKERRQ(ierr);
  ierr = VecCopy(tauQS,right);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(right,right,eta);CHKERRQ(ierr);


  //~// from SymmFault, here for debugging purposes
  //~ierr = VecDuplicate(_tauQSP,&tauQS);CHKERRQ(ierr);
  //~ierr = VecDuplicate(_tauQSP,&eta);CHKERRQ(ierr);
  //~ierr = VecCopy(_tauQSP,tauQS);CHKERRQ(ierr);
  //~ierr = VecCopy(_zP,eta);
  //~ierr = VecScale(eta,0.5);CHKERRQ(ierr);

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
    ierr = VecSetValue(_slipVel,Ii,outVal,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_slipVel);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_slipVel);CHKERRQ(ierr);

  // compute velPlus
  // velPlus = (+tauQSplus - tauQSminus + zMinus*vel)/(zPlus + zMinus)
  Vec tauSum=NULL, velCorr=NULL;
  VecDuplicate(_slipVel,&tauSum);
  VecCopy(_tauQSP,tauSum);
  VecAXPY(tauSum,-1.0,_tauQSMinus);
  VecPointwiseDivide(tauSum,tauSum,zSum);
  VecDuplicate(_slipVel,&velCorr);
  VecPointwiseDivide(velCorr,_zM,zSum);
  VecPointwiseMult(velCorr,velCorr,_slipVel);
  VecCopy(tauSum,_velPlus);
  VecAXPY(_velPlus,1.0,velCorr);

  // compute velMinus
  ierr = VecCopy(_velPlus,_velMinus);CHKERRQ(ierr);
  ierr = VecAXPY(_velMinus,-1.0,_slipVel);CHKERRQ(ierr);


  //~// from SymmFault, here for debugging purposes
  //~ierr = VecCopy(_slipVel,_velPlus);CHKERRQ(ierr);
  //~ierr = VecScale(_velPlus,0.5);CHKERRQ(ierr);
  //~ierr = VecCopy(_velPlus,_velMinus);CHKERRQ(ierr);
  //~ierr = VecScale(_velMinus,-1.0);CHKERRQ(ierr);


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


  // tauQSPlus/Minus, zPlus/Minus, bcRShift, sigma_N
  PetscScalar a,b,zPlus,zMinus,tau_inf,sigma_N,mu,cs;
  ierr = VecGetOwnershipRange(_tauQSP,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr =  VecGetValues(_a,1,&Ii,&a);CHKERRQ(ierr);
    ierr =  VecGetValues(_b,1,&Ii,&b);CHKERRQ(ierr);
    ierr =  VecGetValues(_sigma_N,1,&Ii,&sigma_N);CHKERRQ(ierr);

    ierr =  VecGetValues(*_muVecP,1,&Ii,&mu);CHKERRQ(ierr);
    ierr =  VecGetValues(*_csVecP,1,&Ii,&cs);CHKERRQ(ierr);
    zPlus = mu/cs;
    PetscPrintf(PETSC_COMM_WORLD,"FullFault::SsetSplitNodeFields not set up!!\n");
    assert(0);
    zMinus = mu/cs;

    tau_inf = sigma_N*a*asinh( (double) 0.5*_vL*exp(_f0/a)/_v0 );

    ierr = VecSetValue(_tauQSP,Ii,tau_inf,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(_tauQSMinus,Ii,tau_inf,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(_zP,Ii,zPlus,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(_zM,Ii,zMinus,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(_sigma_N,Ii,sigma_N,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(_tauQSP);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_tauQSMinus);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_zP);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_zM);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_sigma_N);CHKERRQ(ierr);

  ierr = VecAssemblyEnd(_tauQSP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_tauQSMinus);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_zP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_zM);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_sigma_N);CHKERRQ(ierr);



#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullFault::setSplitNodeFields in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


// initialize uPlus and uMinus from lithosphere's data
PetscErrorCode FullFault::setFaultDisp(Vec const &bcLPlus,Vec const &bcRMinus)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullFault::setFullFaultDisp in fault.cpp\n");CHKERRQ(ierr);
#endif

    ierr = VecCopy(bcLPlus,_uP);CHKERRQ(ierr);
    ierr = VecCopy(bcRMinus,_uM);CHKERRQ(ierr);

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

  PetscInt       Ii,Istart,Iend,Jj;
  PetscScalar    v;

  ierr = VecGetOwnershipRange(sigma_xyPlus,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii<_N) {
      ierr = VecGetValues(sigma_xyPlus,1,&Ii,&v);CHKERRQ(ierr);
      ierr = VecSetValues(_tauQSP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_tauQSP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_tauQSP);CHKERRQ(ierr);

  // get last _N values in array
  ierr = VecGetOwnershipRange(sigma_xyMinus,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii>_arrSize - _N - 1) {
      Jj = Ii - (_arrSize - _N);
      ierr = VecGetValues(sigma_xyMinus,1,&Ii,&v);CHKERRQ(ierr);
      ierr = VecSetValues(_tauQSMinus,1,&Jj,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullFault::setTauQS in lithosphere.c\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode FullFault::getResid(const PetscInt ind,const PetscScalar slipVel,PetscScalar *out)
{
  PetscErrorCode ierr = 0;
  PetscScalar    psi,a,sigma_N,zPlus,zMinus,tauQSplus,tauQSminus;
  PetscInt       Istart,Iend;

#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullFault::getResid in fault.cpp\n");CHKERRQ(ierr);
#endif

  ierr = VecGetOwnershipRange(_state,&Istart,&Iend);
  assert( ind>=Istart && ind<Iend );

  //~ierr = VecGetValues(_tempPsi,1,&ind,&psi);CHKERRQ(ierr);
  ierr = VecGetValues(_state,1,&ind,&psi);CHKERRQ(ierr);
  ierr = VecGetValues(_a,1,&ind,&a);CHKERRQ(ierr);
  ierr = VecGetValues(_sigma_N,1,&ind,&sigma_N);CHKERRQ(ierr);

  ierr = VecGetValues(_zP,1,&ind,&zPlus);CHKERRQ(ierr);
  ierr = VecGetValues(_tauQSP,1,&ind,&tauQSplus);CHKERRQ(ierr);

  ierr = VecGetValues(_tauQSMinus,1,&ind,&tauQSminus);CHKERRQ(ierr);
  ierr = VecGetValues(_zM,1,&ind,&zMinus);CHKERRQ(ierr);

  // frictional strength of fault
  PetscScalar strength = (PetscScalar) a*sigma_N*asinh( (double) (slipVel/2/_v0)*exp(psi/a) );

  // stress on fault
  PetscScalar stress = (zMinus/(zPlus+zMinus)*tauQSplus
                       + zPlus/(zPlus+zMinus)*tauQSminus)
                       - zPlus*zMinus/(zPlus+zMinus)*slipVel;

  // from symmetric fault (here for debugging purposes)
  //~stress = tauQSplus - 0.5*zPlus*slipVel; // stress on fault

  *out = strength - stress;

#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"    psi=%g,a=%g,sigma_n=%g,zPlus=%g,tau=%g,vel=%g,out=%g\n",psi,a,sigma_N,zPlus,tauQSplus,vel,out);
#endif
  if (isnan(*out)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isnan(*out) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%g,a=%g,sigma_n=%g,eta=%g,tau=%g,vel=%g\n",psi,a,sigma_N,zPlus,tauQSplus,slipVel);
    CHKERRQ(ierr);
  }
  else if (isinf(*out)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isinf(*out) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%g,a=%g,sigma_n=%g,eta=%g,tau=%g,vel=%g\n",psi,a,sigma_N,zPlus,tauQSplus,slipVel);
    CHKERRQ(ierr);
  }

  assert(!isnan(*out));
  assert(!isinf(*out));

#if VERBOSE > 3
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullFault::getResid in fault.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}







PetscErrorCode FullFault::d_dt(const_it_vec varBegin,const_it_vec varEnd,
                               it_vec dvarBegin,it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 3
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullFault::d_dt in fault.cpp\n");CHKERRQ(ierr);
#endif

  PetscScalar    val,stateVal;
  PetscInt       Ii,Istart,Iend;

  assert(varBegin+1 != varEnd);

  ierr = computeVel();CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(_slipVel,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr = VecGetValues(*varBegin,1,&Ii,&stateVal);
    ierr = agingLaw(Ii,stateVal,val);CHKERRQ(ierr);
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
  ierr = VecView(_zP,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = outputDir + "zMinus";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_zM,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // output normal stress vector
  str =  outputDir + "sigma_N";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_sigma_N,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

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
      ierr = VecView(_uP,_uPlusViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_uPlusViewer);CHKERRQ(ierr);

      PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"velPlus").c_str(),FILE_MODE_WRITE,&_velPlusViewer);
      ierr = VecView(_velPlus,_velPlusViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_velPlusViewer);CHKERRQ(ierr);

      PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"tauQSPlus").c_str(),FILE_MODE_WRITE,&_tauQSPlusViewer);
      ierr = VecView(_tauQSP,_tauQSPlusViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_tauQSPlusViewer);CHKERRQ(ierr);

      PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"psi").c_str(),FILE_MODE_WRITE,&_stateViewer);
      ierr = VecView(_state,_stateViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_stateViewer);CHKERRQ(ierr);

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"uPlus").c_str(),
                                   FILE_MODE_APPEND,&_uPlusViewer);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"velPlus").c_str(),
                                   FILE_MODE_APPEND,&_velPlusViewer);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"tauQSPlus").c_str(),
                                   FILE_MODE_APPEND,&_tauQSPlusViewer);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"psi").c_str(),
                                   FILE_MODE_APPEND,&_stateViewer);CHKERRQ(ierr);

        PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"uMinus").c_str(),FILE_MODE_WRITE,&_uMV);
        ierr = VecView(_uM,_uMV);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&_uMV);CHKERRQ(ierr);

        PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"tauQSMinus").c_str(),FILE_MODE_WRITE,&_tauQSMinusViewer);
        ierr = VecView(_tauQSMinus,_tauQSMinusViewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&_tauQSMinusViewer);CHKERRQ(ierr);

        PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"velMinus").c_str(),FILE_MODE_WRITE,&_velMinusViewer);
        ierr = VecView(_velMinus,_velMinusViewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&_velMinusViewer);CHKERRQ(ierr);

        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"uMinus").c_str(),
                                   FILE_MODE_APPEND,&_uMV);CHKERRQ(ierr);
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"velMinus").c_str(),
                                   FILE_MODE_APPEND,&_velMinusViewer);CHKERRQ(ierr);
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(outputDir+"tauQSMinus").c_str(),
                                   FILE_MODE_APPEND,&_tauQSMinusViewer);CHKERRQ(ierr);
  }
  else {
    ierr = VecView(_uP,_uPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_velPlus,_velPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_tauQSP,_tauQSPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_state,_stateViewer);CHKERRQ(ierr);

      ierr = VecView(_uM,_uMV);CHKERRQ(ierr);
      ierr = VecView(_velMinus,_velMinusViewer);CHKERRQ(ierr);
      ierr = VecView(_tauQSMinus,_tauQSMinusViewer);CHKERRQ(ierr);
  }

#if VERBOSE > 1
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullFault::writeStep in fault.cpp at step %i\n",step);CHKERRQ(ierr);
#endif
  return ierr;
}



