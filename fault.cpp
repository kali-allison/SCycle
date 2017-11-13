#include "fault.hpp"

#define FILENAME "fault.cpp"

using namespace std;


Fault::Fault(Domain&D, HeatEquation& He)
: _file(D._file),_delim(D._delim),_outputDir(D._outputDir),_isMMS(D._isMMS),_bcLTauQS(0),_stateLaw("agingLaw"),
  _N(D._Nz),_L(D._Lz),_h(D._dr),_z(NULL),
  _rootTol(0),_rootIts(0),_maxNumIts(1e8),
  _f0(0.6),_v0(1e-6),_vL(D._vL),
  _fw(0.64),_Vw(0.12),_tau_c(0),_Tw(0),_D(0),_T(NULL),_k(NULL),_rho(NULL),_c(NULL),
  _a(NULL),_b(NULL),_Dc(NULL),_cohesion(NULL),
  _dPsi(NULL),_psi(NULL),_theta(NULL),
  _sigmaN_cap(1e14),_sigmaN_floor(0.),_sNEff(NULL),
  _slip(NULL),_slipVel(NULL),
  _computeVelTime(0),_stateLawTime(0),
  _tauQSP(NULL),_tauP(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "Fault::Fault";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // set a, b, normal stress, and Dc
  loadSettings(_file);
  checkInput();

  // fields that exist on the fault
  VecCreate(PETSC_COMM_WORLD,&_tauQSP);
  VecSetSizes(_tauQSP,PETSC_DECIDE,_N);
  VecSetFromOptions(_tauQSP);      PetscObjectSetName((PetscObject) _tauQSP, "tauQS");  VecSet(_tauQSP,0.0);
  VecDuplicate(_tauQSP,&_tauP);     PetscObjectSetName((PetscObject) _tauP, "tau");  VecSet(_tauP,0.0);
  VecDuplicate(_tauQSP,&_T);     PetscObjectSetName((PetscObject) _T, "faultTemp");  VecSet(_T,293.15);
  VecDuplicate(_tauQSP,&_psi);     PetscObjectSetName((PetscObject) _psi, "psi"); VecSet(_psi,0.0);
  VecDuplicate(_tauQSP,&_theta);   PetscObjectSetName((PetscObject) _theta, "theta"); VecSet(_theta,0.0);
  VecDuplicate(_tauQSP,&_dPsi);    PetscObjectSetName((PetscObject) _dPsi, "dPsi"); VecSet(_dPsi,0.0);
  VecDuplicate(_tauQSP,&_dTheta);  PetscObjectSetName((PetscObject) _dTheta, "dTheta"); VecSet(_dTheta,0.0);
  VecDuplicate(_tauQSP,&_slip);    PetscObjectSetName((PetscObject) _slip, "slip"); VecSet(_slip,0.0);
  VecDuplicate(_tauQSP,&_slipVel); PetscObjectSetName((PetscObject) _slipVel, "slipVel");
  VecSet(_slipVel,0.0);


  // flash heating parameters
  VecDuplicate(_tauQSP,&_rho);   PetscObjectSetName((PetscObject) _rho, "faultRho"); VecSet(_rho,3.0);
  VecDuplicate(_tauQSP,&_k);   PetscObjectSetName((PetscObject) _k, "faultRho"); VecSet(_k,1.89e-9);
  VecDuplicate(_tauQSP,&_c);   PetscObjectSetName((PetscObject) _c, "faultC"); VecSet(_c,900);
  setHeatParams(He._k,He._rho,He._c);


  // initialize _z
  VecDuplicate(_tauQSP,&_z);
  PetscInt    Istart,Iend;
  PetscScalar z = 0;
  VecGetOwnershipRange(D._z,&Istart,&Iend);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    if (Ii < _N) {
      VecGetValues(D._z,1,&Ii,&z);
      VecSetValue(_z,Ii,z,INSERT_VALUES);
    }
  }
  VecAssemblyBegin(_z);
  VecAssemblyEnd(_z);

  setFrictionFields(D);
  VecCopy(_psi,_theta);

  //~ if (D._loadICs==1) { loadFieldsFromFiles(D._inputDir); }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}



// Check that required fields have been set by the input file
PetscErrorCode Fault::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Fault::checkInput";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  assert(_DcVals.size() == _DcDepths.size() );
  assert(_aVals.size() == _aDepths.size() );
  assert(_bVals.size() == _bDepths.size() );
  assert(_sigmaNVals.size() == _sigmaNDepths.size() );
  assert(_cohesionVals.size() == _cohesionDepths.size() );
  assert(_impedanceVals.size() == _impedanceVals.size() );

  assert(_DcVals.size() != 0 );
  assert(_aVals.size() != 0 );
  assert(_bVals.size() != 0 );
  assert(_sigmaNVals.size() != 0 );
  assert(_cohesionVals.size() != 0 );
  assert(_impedanceVals.size() != 0 );

  assert(_rootTol >= 1e-14);

  assert(_stateLaw.compare("agingLaw")==0
    || _stateLaw.compare("slipLaw")==0
    || _stateLaw.compare("flashHeating")==0
    || _stateLaw.compare("stronglyVWLaw")==0);

  assert(_v0 > 0);
  assert(_f0 > 0);

  if (!_stateLaw.compare("flashHeating")) {
    //~ assert(_Vw > 0);
    assert(_fw >= 0);
    assert(_tau_c > 0);
    assert(_Tw > 0);
  }



  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  //~}
  return ierr;
}

// takes in full size rho, c, and k
PetscErrorCode Fault::setHeatParams(const Vec& k,const Vec& rho,const Vec& c)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Fault::setHeatParams";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscInt       Ii,Istart,Iend;
  PetscScalar    v = 0;

  ierr = VecGetOwnershipRange(k,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii<_N) {
      ierr = VecGetValues(k,1,&Ii,&v);CHKERRQ(ierr);
      ierr = VecSetValues(_k,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

      ierr = VecGetValues(rho,1,&Ii,&v);CHKERRQ(ierr);
      ierr = VecSetValues(_rho,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

      ierr = VecGetValues(c,1,&Ii,&v);CHKERRQ(ierr);
      ierr = VecSetValues(_c,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_k);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_rho);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_c);CHKERRQ(ierr);

  ierr = VecAssemblyEnd(_k);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_rho);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_c);CHKERRQ(ierr);


#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
#endif
  return ierr;
}


PetscErrorCode Fault::view(const double totRunTime)
{
  PetscErrorCode ierr = 0;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Fault Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   compute slip vel time (s): %g\n",_computeVelTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   state law time (s): %g\n",_stateLawTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent finding slip vel law: %g\n",_computeVelTime/totRunTime*100.);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent in state law: %g\n",_stateLawTime/totRunTime*100.);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  return ierr;
}


Fault::~Fault()
{
  #if VERBOSE > 1
    std::string funcName = "Fault::~Fault";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // fields that exist on the fault
  VecDestroy(&_tauQSP);
  VecDestroy(&_tauP);
  VecDestroy(&_psi);
  VecDestroy(&_dPsi);
  VecDestroy(&_theta);
  VecDestroy(&_dTheta);
  VecDestroy(&_slip);
  VecDestroy(&_slipVel);
  VecDestroy(&_T);
  VecDestroy(&_z);

  // frictional fields
  VecDestroy(&_Dc);
  VecDestroy(&_zP);
  VecDestroy(&_a);
  VecDestroy(&_b);
  VecDestroy(&_sNEff);
  VecDestroy(&_cohesion);

  // for flash heating
  VecDestroy(&_T);
  VecDestroy(&_k);
  VecDestroy(&_rho);
  VecDestroy(&_c);

  for (map<string,PetscViewer>::iterator it=_viewers.begin(); it!=_viewers.end(); it++ ) {
    PetscViewerDestroy(&_viewers[it->first]);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// Fills vec with the linear interpolation between the pairs of points (vals,depths).
PetscErrorCode Fault::setVecFromVectors(Vec& vec, vector<double>& vals,vector<double>& depths)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    v,z,z0,z1,v0,v1;
  #if VERBOSE > 1
    std::string funcName = "Fault::setVecFromVectors";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecSet(vec,vals[0]);

  // build structure from generalized input
  size_t vecLen = depths.size();
  ierr = VecGetOwnershipRange(vec,&Istart,&Iend);CHKERRQ(ierr);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++)
  {
    VecGetValues(_z,1,&Ii,&z);CHKERRQ(ierr);
    //~ PetscPrintf(PETSC_COMM_SELF,"%i: z = %g\n",Ii,z);
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
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// Fills vec with the linear interpolation between the pairs of points (vals,depths), but always below the specified max value
PetscErrorCode Fault::setVecFromVectors(Vec& vec, vector<double>& vals,vector<double>& depths,
  const PetscScalar maxVal)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    v,z,z0,z1,v0,v1;
  #if VERBOSE > 1
    std::string funcName = "Fault::setVecFromVectors";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // build structure from generalized input
  size_t vecLen = depths.size();
  ierr = VecGetOwnershipRange(vec,&Istart,&Iend);CHKERRQ(ierr);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++)
  {
    VecGetValues(_z,1,&Ii,&z);CHKERRQ(ierr);
    //~ PetscPrintf(PETSC_COMM_SELF,"%i: z = %g\n",Ii,z);
    for (size_t ind = 0; ind < vecLen-1; ind++) {
      z0 = depths[0+ind];
      z1 = depths[0+ind+1];
      v0 = vals[0+ind];
      v1 = vals[0+ind+1];
      if (z>=z0 && z<=z1) { v = (v1 - v0)/(z1-z0) * (z-z0) + v0; }
      v = min(maxVal,v);
      ierr = VecSetValues(vec,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode Fault::setFrictionFields(Domain&D)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Fault::setFrictionFields";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // frictional fields
  VecDuplicate(_tauQSP,&_Dc); PetscObjectSetName((PetscObject) _Dc, "Dc");
  VecDuplicate(_tauQSP,&_sNEff); PetscObjectSetName((PetscObject) _sNEff, "_sNEff");
  VecDuplicate(_tauQSP,&_zP); PetscObjectSetName((PetscObject) _zP, "zP");
  VecDuplicate(_tauQSP,&_a); PetscObjectSetName((PetscObject) _a, "a");
  VecDuplicate(_tauQSP,&_b); PetscObjectSetName((PetscObject) _b, "b");
  VecDuplicate(_tauQSP,&_cohesion); PetscObjectSetName((PetscObject) _cohesion, "_cohesion");
  VecSet(_cohesion,0);

  // set depth-independent fields
  ierr = VecSet(_psi,_f0);CHKERRQ(ierr); // in terms of psi
  ierr = VecSet(_theta,1e9);CHKERRQ(ierr); // correct

  // set a using a vals
  if (_N == 1) {
    VecSet(_b,_bVals[0]);
    VecSet(_a,_aVals[0]);
    VecSet(_sNEff,_sigmaNVals[0]);
    VecSet(_Dc,_DcVals[0]);
    VecSet(_cohesion,_cohesionVals[0]);
  }
  else {
    ierr = setVecFromVectors(_a,_aVals,_aDepths);CHKERRQ(ierr);
    ierr = setVecFromVectors(_b,_bVals,_bDepths);CHKERRQ(ierr);
    ierr = setVecFromVectors(_sNEff,_sigmaNVals,_sigmaNDepths);CHKERRQ(ierr);
    ierr = setVecFromVectors(_Dc,_DcVals,_DcDepths);CHKERRQ(ierr);
    ierr = setVecFromVectors(_cohesion,_cohesionVals,_cohesionDepths);CHKERRQ(ierr);
    ierr = setVecFromVectors(_zP,_impedanceVals,_impedanceDepths);CHKERRQ(ierr);
  }

  // impose floor and ceiling on effective normal stress
  {
    Vec temp; VecDuplicate(_sNEff,&temp);
    VecSet(temp,_sigmaN_cap); VecPointwiseMin(_sNEff,_sNEff,temp);
    VecSet(temp,_sigmaN_floor); VecPointwiseMax(_sNEff,_sNEff,temp);
    VecDestroy(&temp);
  }

  //~ VecWAXPY(_sN,1.0,_p,_sNEff);
  VecDuplicate(_sNEff,&_sN);
  VecCopy(_sNEff,_sN);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode Fault::getTauRS(Vec& tauRS, const PetscScalar vL)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 2
    std::string funcName = "Fault::getTauSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (tauRS == NULL) { VecDuplicate(_slipVel,&tauRS); }

  PetscInt       Istart,Iend;
  PetscScalar   *tauRSV,*sN,*a=0;
  VecGetOwnershipRange(tauRS,&Istart,&Iend);
  VecGetArray(tauRS,&tauRSV);
  VecGetArray(_sNEff,&sN);
  VecGetArray(_a,&a);
  PetscInt Jj = 0;
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    PetscScalar psiSS = _f0;
    tauRSV[Jj] = sN[Jj]*a[Jj]*asinh( (double) 0.5*vL*exp(psiSS/a[Jj])/_v0 );
    Jj++;
  }
  VecRestoreArray(tauRS,&tauRSV);
  VecRestoreArray(_sNEff,&sN);
  VecRestoreArray(_a,&a);

  VecSet(_slipVel,vL);

  #if VERBOSE > 3
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// put tau and slip vel in varSS
PetscErrorCode Fault::initiateVarSS(map<string,Vec>& varSS)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Fault::initiateVarSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  varSS["slipVel"] = _slipVel;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

  // extract slipVel from v, update tauSS from this
PetscErrorCode Fault::updateSS(map<string,Vec>& varSS)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Fault::updateSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // extract slipVel from v
  Vec vel = varSS.find("v")->second;
  PetscInt       Istart,Iend;
  PetscScalar    val = 0;
  ierr = VecGetOwnershipRange(vel,&Istart,&Iend);CHKERRQ(ierr);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    if (Ii<_N) {
      ierr = VecGetValues(vel,1,&Ii,&val);CHKERRQ(ierr);
      val = 2.* val;
      ierr = VecSetValues(_slipVel,1,&Ii,&val,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_slipVel);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_slipVel);CHKERRQ(ierr);

  // update shear stress on fault
  PetscScalar   *tau,*sN,*a,*V=0;
  VecGetOwnershipRange(_sNEff,&Istart,&Iend);
  VecGetArray(varSS["tau"],&tau);
  //~ VecGetArray(varSS["tauExtra"],&tau);
  VecGetArray(_sNEff,&sN);
  VecGetArray(_a,&a);
  VecGetArray(_slipVel,&V);
  PetscInt Jj = 0;
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    PetscScalar psiSS = _f0;
    tau[Jj] = a[Jj] * sN[Jj] * asinh( (double) (0.5*V[Jj]/_v0)  *exp(psiSS/a[Jj]));
    Jj++;
  }
  VecRestoreArray(varSS["tau"],&tau);
  //~ VecRestoreArray(varSS["tauExtra"],&tau);
  VecRestoreArray(_sNEff,&sN);
  VecRestoreArray(_a,&a);
  VecRestoreArray(_slipVel,&V);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// compute slip vel from tau
PetscErrorCode Fault::computeVss(const Vec tau)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Fault::computeVss";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscInt       Istart,Iend;
  PetscScalar   *tauV,*sN,*a,*V=0;
  VecGetOwnershipRange(tau,&Istart,&Iend);
  VecGetArray(tau,&tauV);
  VecGetArray(_sNEff,&sN);
  VecGetArray(_a,&a);
  VecGetArray(_slipVel,&V);
  PetscInt Jj = 0;
  for (PetscInt Ii=Istart;Ii<Iend;Ii++) {
    PetscScalar psiSS = _f0;
    V[Jj] = 2.*_v0*exp(-psiSS/a[Jj]) * sinh( (double) tauV[Jj]/a[Jj]/sN[Jj] );
    Jj++;
  }
  VecRestoreArray(tau,&tauV);
  VecRestoreArray(_sNEff,&sN);
  VecRestoreArray(_a,&a);
  VecRestoreArray(_slipVel,&V);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscScalar Fault::getTauSS(PetscInt& ind)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 3
    std::string funcName = "Fault::getTauSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscInt       Istart,Iend;
  PetscScalar    a,b,sigma_N;

  // throw error if value requested is not stored locally
  ierr = VecGetOwnershipRange(_tauQSP,&Istart,&Iend);CHKERRQ(ierr);
  assert(ind>=Istart && ind<Iend);

  ierr =  VecGetValues(_a,1,&ind,&a);CHKERRQ(ierr);
  ierr =  VecGetValues(_b,1,&ind,&b);CHKERRQ(ierr);
  ierr =  VecGetValues(_sNEff,1,&ind,&sigma_N);CHKERRQ(ierr);

  #if VERBOSE > 3
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return sigma_N*a*asinh( (double) 0.5*_vL*exp(_f0/a)/_v0 );
  //~ return sigma_N* (_f0 + (a-b) * log10(_vL/_v0) );
}

// aging law in terms of theta
PetscErrorCode Fault::agingLaw_theta(const PetscInt ind,const PetscScalar state,PetscScalar &dstate)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    b,slipVel,Dc;


  ierr = VecGetOwnershipRange(_theta,&Istart,&Iend);
  assert( ind>=Istart && ind<Iend);
  ierr = VecGetValues(_Dc,1,&ind,&Dc);CHKERRQ(ierr);
  ierr = VecGetValues(_b,1,&ind,&b);CHKERRQ(ierr);
  ierr = VecGetValues(_slipVel,1,&ind,&slipVel);CHKERRQ(ierr);
  slipVel = abs(slipVel); // state evolution is not sensitive to direction of slip


  dstate = 1 - state*slipVel/Dc;

  assert(!isnan(dstate));
  assert(!isinf(dstate));

  return ierr;
}

// aging law in terms of psi
PetscErrorCode Fault::agingLaw_psi(const PetscInt ind,const PetscScalar state,PetscScalar &dstate)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    b,slipVel,Dc;


  ierr = VecGetOwnershipRange(_psi,&Istart,&Iend);
  assert( ind>=Istart && ind<Iend);
  ierr = VecGetValues(_Dc,1,&ind,&Dc);CHKERRQ(ierr);
  ierr = VecGetValues(_b,1,&ind,&b);CHKERRQ(ierr);
  ierr = VecGetValues(_slipVel,1,&ind,&slipVel);CHKERRQ(ierr);
  slipVel = abs(slipVel); // state evolution is not sensitive to direction of slip



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

  assert(!isnan(dstate));
  assert(!isinf(dstate));

  return ierr;
}


PetscErrorCode Fault::slipLaw_theta(const PetscInt ind,const PetscScalar state,PetscScalar &dstate)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    b,slipVel,Dc;


  ierr = VecGetOwnershipRange(_theta,&Istart,&Iend);
  assert( ind>=Istart && ind<Iend);
  ierr = VecGetValues(_Dc,1,&ind,&Dc);CHKERRQ(ierr);
  ierr = VecGetValues(_b,1,&ind,&b);CHKERRQ(ierr);
  ierr = VecGetValues(_slipVel,1,&ind,&slipVel);CHKERRQ(ierr);
  //~ slipVel = abs(slipVel); // state evolution is not sensitive to direction of slip

    PetscScalar A = state*slipVel/Dc;
    if (A == 0) { dstate = 0; }
    else {dstate = -A*log(A); }

  if (isnan(dstate) || isinf(dstate)) {
    PetscPrintf(PETSC_COMM_WORLD,"state = %e, slipVel=%e,Dc = %e\n",state,slipVel,Dc);
  }
  assert(!isnan(dstate));
  assert(!isinf(dstate));


  return ierr;
}


PetscErrorCode Fault::slipLaw_psi(const PetscInt ind,const PetscScalar state,PetscScalar &dstate)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    a,b,slipVel,Dc,sN;


  ierr = VecGetOwnershipRange(_psi,&Istart,&Iend);
  assert( ind>=Istart && ind<Iend);
  ierr = VecGetValues(_Dc,1,&ind,&Dc);CHKERRQ(ierr);
  ierr = VecGetValues(_a,1,&ind,&a);CHKERRQ(ierr);
  ierr = VecGetValues(_b,1,&ind,&b);CHKERRQ(ierr);
  ierr = VecGetValues(_sNEff,1,&ind,&sN);CHKERRQ(ierr);
  ierr = VecGetValues(_slipVel,1,&ind,&slipVel);CHKERRQ(ierr);
  slipVel = abs(slipVel); // state evolution is not sensitive to direction of slip

  PetscScalar fss = _f0 + (a-b)*log(slipVel/_v0);
  PetscScalar f = state + a*log(slipVel/_v0);
  //~ PetscScalar f = (PetscScalar) a*sN*asinh( (double) (slipVel/2./_v0)*exp(state/a) );
  dstate = -slipVel/Dc *(f - fss);

  if (isnan(dstate)) {
    PetscPrintf(PETSC_COMM_WORLD,"state = %e, slipVel=%e,Dc = %e\n",state,slipVel,Dc);
  }
  assert(!isnan(dstate));
  assert(!isinf(dstate));

  return ierr;
}

PetscErrorCode Fault::flashHeating_psi(const PetscInt ind,const PetscScalar state,PetscScalar &dstate)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    a,b,slipVel,Dc,sN,z;


  ierr = VecGetOwnershipRange(_psi,&Istart,&Iend);
  assert( ind>=Istart && ind<Iend);
  ierr = VecGetValues(_Dc,1,&ind,&Dc);CHKERRQ(ierr);
  ierr = VecGetValues(_a,1,&ind,&a);CHKERRQ(ierr);
  ierr = VecGetValues(_b,1,&ind,&b);CHKERRQ(ierr);
  ierr = VecGetValues(_z,1,&ind,&z);CHKERRQ(ierr);
  ierr = VecGetValues(_sNEff,1,&ind,&sN);CHKERRQ(ierr);
  ierr = VecGetValues(_slipVel,1,&ind,&slipVel);CHKERRQ(ierr);
  slipVel = abs(slipVel); // state evolution is not sensitive to direction of slip

  //~ if (slipVel == 0.0) { PetscPrintf(PETSC_COMM_WORLD,"hi!\n"); }
  if (slipVel == 0.0) { slipVel += 1e-14; }

  // flash heating parameters
  PetscScalar fLV = _f0 + (a-b)*log(slipVel/_v0);
  PetscScalar fss = fLV;


  // if not using constant Vw
  PetscScalar rho,c,k,T;
  ierr = VecGetValues(_T,1,&ind,&T);CHKERRQ(ierr);
  ierr = VecGetValues(_rho,1,&ind,&rho);CHKERRQ(ierr);
  ierr = VecGetValues(_c,1,&ind,&c);CHKERRQ(ierr);
  ierr = VecGetValues(_k,1,&ind,&k);CHKERRQ(ierr);
  PetscScalar rc = rho * c;
  PetscScalar ath = k/rc;

  PetscScalar Vw = (M_PI*ath/_D) * pow((_Tw-T)/(_tau_c/rc),2);
  //~ PetscScalar Vw = _Vw;

  if (abs(slipVel) > Vw) { fss = _fw + (fLV - _fw)*(Vw/slipVel); }
  //~ if (a-b > 0.0) { fss = fLV; }
  PetscScalar f = state + a*log(slipVel/_v0);
  dstate = -slipVel/Dc *(f - fss);

  if (isnan(dstate)) {
    PetscPrintf(PETSC_COMM_WORLD,"state = %e, slipVel=%e,Dc = %e\n",state,slipVel,Dc);
  }
  assert(!isnan(dstate));
  assert(!isinf(dstate));

  return ierr;
}


// state evolution law for strongly velocity-weakening friction
PetscErrorCode Fault::stronglyVWLaw_theta(const PetscInt ind,const PetscScalar state,PetscScalar &dstate)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    b,slipVel,Dc;


  ierr = VecGetOwnershipRange(_theta,&Istart,&Iend);
  assert( ind>=Istart && ind<Iend);
  ierr = VecGetValues(_Dc,1,&ind,&Dc);CHKERRQ(ierr);
  ierr = VecGetValues(_b,1,&ind,&b);CHKERRQ(ierr);
  ierr = VecGetValues(_slipVel,1,&ind,&slipVel);CHKERRQ(ierr);
  slipVel = abs(slipVel); // state evolution is not sensitive to direction of slip

    //~PetscScalar fss = _f0 + log(slipVel/_v0);



    PetscScalar a = 0;
    ierr = VecGetValues(_a,1,&ind,&a);CHKERRQ(ierr);
    PetscScalar fw = 0.2,Vw = 0.1,n = 8.0;
    PetscScalar fLV = _f0 - (b-a)*log(slipVel/_v0);
    PetscScalar fss = fw + (fLV - fw)/pow(1 + pow(slipVel/Vw,n),1.0/n);
    PetscScalar f = (PetscScalar) a * asinh( (double) slipVel/2.0/_v0 * exp(state/a) );
    dstate = -(slipVel/Dc)*(f - fss);

      if (isinf(dstate)) {
    PetscPrintf(PETSC_COMM_WORLD,"slipVel = %.9e, a = %.4e, b = %.4e, f = %.9e\n",slipVel,a,b,f);
    PetscPrintf(PETSC_COMM_WORLD,"fss = %.9e, fLV = %.9e\n",fss,fLV);
    PetscPrintf(PETSC_COMM_WORLD,"state = %.9e\n",fss,fLV);
  }

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
SymmFault::SymmFault(Domain&D, HeatEquation& He)
: Fault(D,He)
{
  #if VERBOSE > 1
    std::string funcName = "SymmFault::SymmFault";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // vectors were allocated in Fault constructor, just need to set values.
  setSplitNodeFields();

  Vec Temp;
  VecDuplicate(He._T0,&Temp);
  He.getTemp(Temp);
  setTemp(Temp);
  VecDestroy(&Temp);

  if (D._loadICs==1) { loadFieldsFromFiles(D._inputDir); }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

SymmFault::~SymmFault()
{
  #if VERBOSE > 1
    std::string funcName = "SymmFault::~SymmFault";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // this is covered by the Fault destructor.

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

PetscErrorCode SymmFault::initiateIntegrand(const PetscScalar time,map<string,Vec>& varEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault::initiateIntegrand";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

 // put variables to be integrated explicitly into varEx
  Vec varPsi; VecDuplicate(_psi,&varPsi); VecCopy(_psi,varPsi);
  varEx["psi"] = varPsi;

  // slip is added by the momentum balance equation
  //~ Vec varSlip; VecDuplicate(_slip,&varSlip); VecCopy(_slip,varSlip);
  //~ varEx["slip"] = varSlip;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode SymmFault::updateFields(const PetscScalar time,const map<string,Vec>& varEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault::updateFields()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  VecCopy(varEx.find("psi")->second,_psi);
  VecCopy(varEx.find("slip")->second,_slip);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// assumes right-lateral fault
PetscErrorCode SymmFault::computeVel()
{
  PetscErrorCode ierr = 0;
  Vec            left,right;
  Vec            tauQS,eta;
  PetscScalar    outVal,leftVal,rightVal,temp;
  PetscInt       Ii,Istart,Iend;

  #if VERBOSE > 1
    std::string funcName = "SymmFault::computeVel";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

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
  ierr = VecAXPY(right,-1.0,_cohesion);CHKERRQ(ierr); // add effect of cohesion!
  ierr = VecPointwiseDivide(right,right,eta);CHKERRQ(ierr);


  ierr = VecDuplicate(right,&left);CHKERRQ(ierr);
  ierr = VecSet(left,0.0);CHKERRQ(ierr);

  PetscScalar *leftA,*rightA,*slipVel=0;
  VecGetArray(left,&leftA);
  VecGetArray(right,&rightA);
  VecGetArray(_slipVel,&slipVel);
  PetscInt Jj = 0; // local index
  ierr = VecGetOwnershipRange(left,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    leftVal = leftA[Jj];
    rightVal = rightA[Jj];
    //~ PetscPrintf(PETSC_COMM_WORLD,"%i: left = %g, right = %g\n",Ii,leftVal,rightVal);

    // check bounds
    if (isnan(leftVal)) {
      PetscPrintf(PETSC_COMM_WORLD,"\n\nError:left evaluated to nan.\n");
      assert(0);
    }
    if (isnan(rightVal)) {
      PetscPrintf(PETSC_COMM_WORLD,"\n\nError:right evaluated to nan.\n");
      assert(0);
    }
    // correct for left-lateral fault motion
    if (leftVal>rightVal) {
      temp = rightVal;
      rightVal = leftVal;
      leftVal = temp;
    }

    if (abs(leftVal-rightVal)<1e-14) { outVal = leftVal; }
    else {
      Bisect rootAlg(_maxNumIts,_rootTol);
      ierr = rootAlg.setBounds(leftVal,rightVal);CHKERRQ(ierr);
      ierr = rootAlg.findRoot(this,Ii,&outVal);CHKERRQ(ierr);
      _rootIts += rootAlg.getNumIts();

      //~ PetscScalar x0;
      //~ ierr = VecGetValues(_slipVel,1,&Ii,&x0);CHKERRQ(ierr);
      //~ BracketedNewton rootAlg(_maxNumIts,_rootTol);
      //~ ierr = rootAlg.setBounds(leftVal,rightVal);CHKERRQ(ierr);
      //~ ierr = rootAlg.findRoot(this,Ii,x0,&outVal);CHKERRQ(ierr);
      _rootIts += rootAlg.getNumIts();
    }
    slipVel[Jj] = outVal;
    //~ PetscPrintf(PETSC_COMM_WORLD,"%i: left = %g, right = %g, slipVel = %g\n",Ii,leftVal,rightVal,outVal);
    Jj++;
  }
  VecRestoreArray(left,&leftA);
  VecRestoreArray(right,&rightA);
  VecRestoreArray(_slipVel,&slipVel);


  ierr = VecDestroy(&tauQS);CHKERRQ(ierr);
  ierr = VecDestroy(&eta);CHKERRQ(ierr);
  ierr = VecDestroy(&left);CHKERRQ(ierr);
  ierr = VecDestroy(&right);CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}



// populate fields on the fault
PetscErrorCode SymmFault::setSplitNodeFields()
{
  PetscErrorCode ierr = 0;
  PetscInt       Ii,Istart,Iend;
  #if VERBOSE > 1
    std::string funcName = "SymmFault::setSplitNodeFields";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  //~ // create properly sized vectors for mu and cs
  //~ Vec muV; VecDuplicate(_a,&muV);
  //~ Vec csV; VecDuplicate(_a,&csV);
  //~ PetscScalar mu,cs;
  //~ ierr = VecGetOwnershipRange(*_muVecP,&Istart,&Iend);CHKERRQ(ierr);
  //~ for (Ii=Istart;Ii<Iend;Ii++) {
    //~ if (Ii < _N) {
    //~ ierr =  VecGetValues(*_muVecP,1,&Ii,&mu);CHKERRQ(ierr);
    //~ ierr =  VecGetValues(*_csVecP,1,&Ii,&cs);CHKERRQ(ierr);

    //~ ierr = VecSetValue(muV,Ii,mu,INSERT_VALUES);CHKERRQ(ierr);
    //~ ierr = VecSetValue(csV,Ii,cs,INSERT_VALUES);CHKERRQ(ierr);
    //~ }
  //~ }
  //~ ierr = VecAssemblyBegin(muV);CHKERRQ(ierr);
  //~ ierr = VecAssemblyBegin(csV);CHKERRQ(ierr);
  //~ ierr = VecAssemblyEnd(muV);CHKERRQ(ierr);
  //~ ierr = VecAssemblyEnd(csV);CHKERRQ(ierr);



  // tau, eta, bcRShift, sigma_N
  //~ PetscScalar a,b,zPlus,tau_inf,sigma_N;
  PetscScalar a,b,tau_inf,sigma_N;
  ierr = VecGetOwnershipRange(_a,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr =  VecGetValues(_a,1,&Ii,&a);CHKERRQ(ierr);
    ierr =  VecGetValues(_b,1,&Ii,&b);CHKERRQ(ierr);
    ierr =  VecGetValues(_sNEff,1,&Ii,&sigma_N);CHKERRQ(ierr);

    //eta = 0.5*sqrt(_rhoArr[Ii]*_muArr[Ii]);
    //~ zPlus = mu/cs;

    tau_inf = sigma_N*a*asinh( (double) 0.5*_vL*exp(_f0/a)/_v0 );

    ierr = VecSetValue(_tauQSP,Ii,tau_inf,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_tauQSP);CHKERRQ(ierr);

  ierr = VecAssemblyEnd(_tauQSP);CHKERRQ(ierr);


  //~ // load impedence from file if desired
  //~ PetscViewer inv; // input viewer
  //~ string vecSourceFile = _inputDir + "imp";
  //~ ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  //~ ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv); CHKERRQ(ierr);
  //~ ierr = PetscViewerPushFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  //~ ierr = VecLoad(_zP,inv);CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// deprecated, no longer useful
PetscErrorCode SymmFault::setFaultDisp(Vec const &bcF, Vec const &bcFminus)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault::setFaultDisp";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // bcF holds displacement at y=0+
  ierr = VecCopy(bcF,_slip);CHKERRQ(ierr);
  ierr = VecScale(_slip,2.0);CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// takes in full size temperature (Ny*Nz)
PetscErrorCode SymmFault::setTemp(const Vec& T)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault::setTemp";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscInt       Ii,Istart,Iend;
  PetscScalar    v = 0;

  //~ PetscScalar a0 = 0.005, b0 = 0.008; // Beeler et al. 2016

  ierr = VecGetOwnershipRange(T,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    if (Ii<_N) {
      ierr = VecGetValues(T,1,&Ii,&v);CHKERRQ(ierr);
      ierr = VecSetValues(_T,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

      // also set a and b
      //~ PetscScalar a = a0*v/298.;
      //~ PetscScalar b = b0*v/298.;
      //~ ierr = VecSetValues(_a,1,&Ii,&a,INSERT_VALUES);CHKERRQ(ierr);
      //~ ierr = VecSetValues(_b,1,&Ii,&b,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_T);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_T);CHKERRQ(ierr);

  //~ ierr = VecAssemblyBegin(_a);CHKERRQ(ierr);
  //~ ierr = VecAssemblyBegin(_b);CHKERRQ(ierr);
  //~ ierr = VecAssemblyEnd(_a);CHKERRQ(ierr);
  //~ ierr = VecAssemblyEnd(_b);CHKERRQ(ierr);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}



PetscErrorCode SymmFault::setTauQS(const Vec&sigma_xyPlus,const Vec& sigma_xyMinus)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault::setTauQS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
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
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// return tau (NOT tauQS)
PetscErrorCode SymmFault::getTau(Vec& tau)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault::getTau";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // tau = tauQS - 0.5*zP*slipVel
  VecPointwiseMult(tau,_zP,_slipVel);
  VecAYPX(tau,-0.5,_tauQSP);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// use pore pressure to compute total normal stress
// sNEff = sN - rho*g*z - dp
PetscErrorCode SymmFault::setSN(const Vec& p)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault::setSN";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif

  ierr = VecWAXPY(_sN,1.,p,_sNEff); CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif
  return ierr;
}

// compute effective normal stress from total and pore pressure:
// sNEff = sN - rho*g*z - dp
PetscErrorCode SymmFault::setSNEff(const Vec& p)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault::setSNEff";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif

  //~ ierr = VecWAXPY(_sNEff,-1.,p,_sN); CHKERRQ(ierr);
    //~ sNEff[Jj] = sN[Jj] - p[Jj];

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME); CHKERRQ(ierr);
  #endif
  return ierr;
}



PetscErrorCode SymmFault::getResid(const PetscInt ind,const PetscScalar slipVel,PetscScalar *out)
{
  PetscErrorCode ierr = 0;
  PetscScalar    psi,a,sigma_N,zPlus,tauQS,Co;
  PetscInt       Istart,Iend;

  #if VERBOSE > 3
    std::string funcName = "SymmFault::getResid";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

    // frictional strength of fault
  ierr = VecGetOwnershipRange(_theta,&Istart,&Iend);
  assert(ind>=Istart && ind<Iend);

  ierr = VecGetValues(_a,1,&ind,&a);CHKERRQ(ierr);
  ierr = VecGetValues(_sNEff,1,&ind,&sigma_N);CHKERRQ(ierr);
  ierr = VecGetValues(_zP,1,&ind,&zPlus);CHKERRQ(ierr);
  ierr = VecGetValues(_tauQSP,1,&ind,&tauQS);CHKERRQ(ierr);
  ierr = VecGetValues(_cohesion,1,&ind,&Co);CHKERRQ(ierr);

  if (!_stateLaw.compare("flashHeating") || !_stateLaw.compare("slipLaw") || !_stateLaw.compare("agingLaw")) {
    // in terms of psi
    ierr = VecGetValues(_psi,1,&ind,&psi);CHKERRQ(ierr);
    //PetscScalar strength = (PetscScalar) a*sigma_N*asinh( (double) (slipVel/2./_v0)*exp(psi/a) );
  }
  else { // if aging law
    //~ // in terms of theta
    PetscScalar state,b,Dc=0;
    ierr = VecGetValues(_theta,1,&ind,&state);CHKERRQ(ierr);
    ierr = VecGetValues(_b,1,&ind,&b);CHKERRQ(ierr);
    ierr = VecGetValues(_Dc,1,&ind,&Dc);CHKERRQ(ierr);
    psi = _f0 + b*log( (double) (state*_v0)/Dc);
    //~ PetscScalar strength = (PetscScalar) a*sigma_N*asinh( (double) (slipVel/2./_v0)*exp(psi/a) );
  }
  PetscScalar strength = (PetscScalar) a*sigma_N*asinh( (double) (slipVel/2./_v0)*exp(psi/a) );



  // effect of cohesion
  strength = strength + Co;

  // stress on fault
  PetscScalar stress = tauQS - 0.5*zPlus*slipVel;

  *out = strength - stress;

#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"    psi=%g,a=%g,sigma_n=%g,z=%g,tau=%g,vel=%g\n",psi,a,sigma_N,zPlus,tauQS,slipVel);
#endif
  if (isnan(*out)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isnan(*out) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%g,a=%g,sigma_n=%g,z=%g,tau=%g,vel=%g\n",psi,a,sigma_N,zPlus,tauQS,slipVel);
    CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(vel/2/_v0)=%.9e\n",slipVel/2/_v0);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"exp(psi/a)=%.9e\n",exp(psi/a));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"z*vel=%.9e\n",zPlus*slipVel);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"strength=%.9e\n",strength);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"stress=%.9e\n",stress);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(slipVel/2./_v0)*exp(psi/a)=%.9e\n",(slipVel/2./_v0)*exp(psi/a));
  }
  else if (isinf(*out)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isinf(*out) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%g,a=%g,sigma_n=%g,z=%g,tau=%g,vel=%g\n",psi,a,sigma_N,zPlus,tauQS,slipVel);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(vel/2/_v0)=%.9e\n",slipVel/2/_v0);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"exp(psi/a)=%.9e\n",exp(psi/a));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"z*vel=%.9e\n",zPlus*slipVel);
    CHKERRQ(ierr);
  }

  assert(!isnan(*out));
  assert(!isinf(*out));

  #if VERBOSE > 3
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// final argument is Jacobian
PetscErrorCode SymmFault::getResid(const PetscInt ind,const PetscScalar slipVel,PetscScalar *out,PetscScalar *J)
{
  PetscErrorCode ierr = 0;
  PetscScalar    psi,a,sigma_N,zPlus,tauQS,Co;
  PetscInt       Istart,Iend;

  #if VERBOSE > 3
    std::string funcName = "SymmFault::getResid";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

    // frictional strength of fault
  ierr = VecGetOwnershipRange(_theta,&Istart,&Iend);
  assert(ind>=Istart && ind<Iend);

  ierr = VecGetValues(_a,1,&ind,&a);CHKERRQ(ierr);
  ierr = VecGetValues(_sNEff,1,&ind,&sigma_N);CHKERRQ(ierr);
  ierr = VecGetValues(_zP,1,&ind,&zPlus);CHKERRQ(ierr);
  ierr = VecGetValues(_tauQSP,1,&ind,&tauQS);CHKERRQ(ierr);
  ierr = VecGetValues(_cohesion,1,&ind,&Co);CHKERRQ(ierr);

  if (!_stateLaw.compare("flashHeating") || !_stateLaw.compare("slipLaw") || !_stateLaw.compare("agingLaw")) {
    // in terms of psi
    ierr = VecGetValues(_psi,1,&ind,&psi);CHKERRQ(ierr);
    //PetscScalar strength = (PetscScalar) a*sigma_N*asinh( (double) (slipVel/2./_v0)*exp(psi/a) );
  }
  else { // if aging law
    //~ // in terms of theta
    PetscScalar state,b,Dc=0;
    ierr = VecGetValues(_theta,1,&ind,&state);CHKERRQ(ierr);
    ierr = VecGetValues(_b,1,&ind,&b);CHKERRQ(ierr);
    ierr = VecGetValues(_Dc,1,&ind,&Dc);CHKERRQ(ierr);
    psi = _f0 + b*log( (double) (state*_v0)/Dc);
    //~ PetscScalar strength = (PetscScalar) a*sigma_N*asinh( (double) (slipVel/2./_v0)*exp(psi/a) );
  }
  PetscScalar A = a*sigma_N;
  PetscScalar B = exp(psi/a)/(2.*_v0);
  //~ PetscScalar strength = (PetscScalar) a*sigma_N*asinh( (double) (slipVel/2./_v0)*exp(psi/a) );
  PetscScalar strength = (PetscScalar) A*asinh( (double) B * slipVel );



  // effect of cohesion
  strength = strength + Co;

  // stress on fault
  PetscScalar stress = tauQS - 0.5*zPlus*slipVel;

  *out = strength - stress;

  *J = A*B/sqrt(B*B*slipVel*slipVel + 1.) + 0.5*zPlus; // derivative with respect to slipVel

#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"    psi=%g,a=%g,sigma_n=%g,z=%g,tau=%g,vel=%g\n",psi,a,sigma_N,zPlus,tauQS,slipVel);
#endif
  if (isnan(*out)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isnan(*out) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%g,a=%g,sigma_n=%g,z=%g,tau=%g,vel=%g\n",psi,a,sigma_N,zPlus,tauQS,slipVel);
    CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(vel/2/_v0)=%.9e\n",slipVel/2/_v0);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"exp(psi/a)=%.9e\n",exp(psi/a));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"z*vel=%.9e\n",zPlus*slipVel);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"strength=%.9e\n",strength);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"stress=%.9e\n",stress);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(slipVel/2./_v0)*exp(psi/a)=%.9e\n",(slipVel/2./_v0)*exp(psi/a));
  }
  else if (isinf(*out)) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"isinf(*out) evaluated to true\n");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"psi=%g,a=%g,sigma_n=%g,z=%g,tau=%g,vel=%g\n",psi,a,sigma_N,zPlus,tauQS,slipVel);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"(vel/2/_v0)=%.9e\n",slipVel/2/_v0);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"exp(psi/a)=%.9e\n",exp(psi/a));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"z*vel=%.9e\n",zPlus*slipVel);
    CHKERRQ(ierr);
  }

  assert(!isnan(*out));
  assert(!isinf(*out));

  #if VERBOSE > 3
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}





PetscErrorCode SymmFault::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault::d_dt";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  double startTime = MPI_Wtime();
  if (_isMMS) {
    ierr = d_dt_mms(time,varEx,dvarEx);CHKERRQ(ierr);
  }
  else if (!_bcLTauQS) {
    ierr = d_dt_eqCycle(time,varEx,dvarEx);CHKERRQ(ierr);
  }
  else {
    VecSet(dvarEx["psi"],0.0);
    VecSet(dvarEx["slip"],0.0);
  }

  _stateLawTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode SymmFault::d_dt_WaveEq(const PetscScalar time,map<string,Vec>& varEx,map<string,Vec>& dvarEx, PetscScalar _deltaT)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault::d_dt_WaveEq";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
//~ PetscScalar    val,psiVal,thetaVal;
// PetscScalar    psi,dpsi;
// PetscInt       Ii,Istart,Iend;

// //~ ierr = VecCopy(varEx.find("psi")->second,_psi);CHKERRQ(ierr);
// //~ ierr = VecCopy(varEx.find("slip")->second,_slip);CHKERRQ(ierr);

// // compute slip velocity
// double startTime = MPI_Wtime();
// ierr = computeVel();CHKERRQ(ierr);
// VecCopy(_slipVel,dvarEx.find("slip")->second);
// _computeVelTime += MPI_Wtime() - startTime;


// // compute rate of state variable
// startTime = MPI_Wtime();
// PetscScalar *psiA,*dpsiA = 0;
// VecGetArray(_psi,&psiA);
// VecGetArray(dvarEx.find("psi")->second,&dpsiA);
// PetscInt Jj = 0; // local array index
// ierr = VecGetOwnershipRange(_psi,&Istart,&Iend); // local portion of global Vec index
// for (Ii=Istart;Ii<Iend;Ii++) {
//   psi = psiA[Jj];

//   if (!_stateLaw.compare("agingLaw")) {
//     //~ ierr = agingLaw_theta(Ii,theta,dtheta);CHKERRQ(ierr);
//     ierr = agingLaw_psi(Ii,psi,dpsi);CHKERRQ(ierr);
//     }
//   else if (!_stateLaw.compare("slipLaw")) {
//     //~ ierr = slipLaw_theta(Ii,theta,dtheta);CHKERRQ(ierr);
//     ierr = slipLaw_psi(Ii,psi,dpsi);CHKERRQ(ierr);
//     }
//   else if (!_stateLaw.compare("flashHeating")) {
//     ierr = flashHeating_psi(Ii,psi,dpsi);CHKERRQ(ierr);
//     }
//   //~ else if (!_stateLaw.compare("stronglyVWLaw")) { ierr = stronglyVWLaw(Ii,stateVal,val);CHKERRQ(ierr); }
//   else { PetscPrintf(PETSC_COMM_WORLD,"_stateLaw not understood!\n"); assert(0); }

//   dpsiA[Jj] = dpsi;
//   Jj++;
// }
// VecRestoreArray(_psi,&psiA);
// VecRestoreArray(dvarEx.find("psi")->second,&dpsiA);
}


PetscErrorCode SymmFault::d_dt_mms(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault::d_dt_mms";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  //~ PetscScalar    val,psiVal,thetaVal;
  //~ PetscScalar    theta,dtheta,psi,dpsi,vel;
  //~ PetscInt       Ii,Istart,Iend;

  VecSet(dvarEx["psi"],0.0);
  VecSet(dvarEx["slip"],0.0);


  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}



PetscErrorCode SymmFault::d_dt_eqCycle(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault::d_dt_eqCycle";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  //~ PetscScalar    val,psiVal,thetaVal;
  PetscScalar    psi,dpsi;
  PetscInt       Ii,Istart,Iend;

  //~ ierr = VecCopy(varEx.find("psi")->second,_psi);CHKERRQ(ierr);
  //~ ierr = VecCopy(varEx.find("slip")->second,_slip);CHKERRQ(ierr);

  // compute slip velocity
double startTime = MPI_Wtime();
  ierr = computeVel();CHKERRQ(ierr);
  VecCopy(_slipVel,dvarEx.find("slip")->second);
_computeVelTime += MPI_Wtime() - startTime;


  // compute rate of state variable
  startTime = MPI_Wtime();
  PetscScalar *psiA,*dpsiA = 0;
  VecGetArray(_psi,&psiA);
  VecGetArray(dvarEx.find("psi")->second,&dpsiA);
  PetscInt Jj = 0; // local array index
  ierr = VecGetOwnershipRange(_psi,&Istart,&Iend); // local portion of global Vec index
  for (Ii=Istart;Ii<Iend;Ii++) {
    psi = psiA[Jj];

    if (!_stateLaw.compare("agingLaw")) {
      //~ ierr = agingLaw_theta(Ii,theta,dtheta);CHKERRQ(ierr);
      ierr = agingLaw_psi(Ii,psi,dpsi);CHKERRQ(ierr);
      }
    else if (!_stateLaw.compare("slipLaw")) {
      //~ ierr = slipLaw_theta(Ii,theta,dtheta);CHKERRQ(ierr);
      ierr = slipLaw_psi(Ii,psi,dpsi);CHKERRQ(ierr);
      }
    else if (!_stateLaw.compare("flashHeating")) {
      ierr = flashHeating_psi(Ii,psi,dpsi);CHKERRQ(ierr);
      }
    //~ else if (!_stateLaw.compare("stronglyVWLaw")) { ierr = stronglyVWLaw(Ii,stateVal,val);CHKERRQ(ierr); }
    else { PetscPrintf(PETSC_COMM_WORLD,"_stateLaw not understood!\n"); assert(0); }

    dpsiA[Jj] = dpsi;
    Jj++;
  }
  VecRestoreArray(_psi,&psiA);
  VecRestoreArray(dvarEx.find("psi")->second,&dpsiA);

  //~ dvarEx["tau"] = _tauQSP;

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}



PetscErrorCode SymmFault::writeContext()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault::writeContext";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  PetscViewer    viewer;

  // write out scalar info
  std::string str = _outputDir + "fault_context.txt";
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());

  ierr = PetscViewerASCIIPrintf(viewer,"rootTol = %.15e\n",_rootTol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"v0 = %.15e\n",_v0);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"v0 = %.15e\n",_v0);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stateEvolutionLaw = %s\n",_stateLaw.c_str());CHKERRQ(ierr);
  if (!_stateLaw.compare("flashHeating")) {
    ierr = PetscViewerASCIIPrintf(viewer,"fw = %.15e\n",_fw);CHKERRQ(ierr);
    //~ ierr = PetscViewerASCIIPrintf(viewer,"Vw = %.15e\n",_Vw);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"tau_c = %.15e # (GPa)\n",_tau_c);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Tw = %.15e # (K)\n",_Tw);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"D = %.15e # (um)\n",_D);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);


  // output vector fields

  str = _outputDir + "a";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_a,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = _outputDir + "b";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_b,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  str = _outputDir + "zPlus";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_zP,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // output normal stress vector
  str =  _outputDir + "sigma_N";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_sNEff,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // output critical distance
  str =  _outputDir + "Dc";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_Dc,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // output cohesion
  str =  _outputDir + "cohesion";
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(_cohesion,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);


  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode SymmFault::writeStep(const PetscInt stepCount, const PetscScalar time)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "SymmFault::writeStep";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif



  if (stepCount == 0) {
    _viewers["slip"] = initiateViewer(_outputDir + "slip");
    _viewers["slipVel"] = initiateViewer(_outputDir + "slipVel");
    _viewers["tauQSP"] = initiateViewer(_outputDir + "tauQSP");
    _viewers["psi"] = initiateViewer(_outputDir + "psi");
    _viewers["fault_T"] = initiateViewer(_outputDir + "fault_T");

    ierr = VecView(_slip,_viewers["slip"]); CHKERRQ(ierr);
    ierr = VecView(_slipVel,_viewers["slipVel"]); CHKERRQ(ierr);
    ierr = VecView(_tauQSP,_viewers["tauQSP"]); CHKERRQ(ierr);
    ierr = VecView(_psi,_viewers["psi"]); CHKERRQ(ierr);
    ierr = VecView(_T,_viewers["fault_T"]); CHKERRQ(ierr);

    ierr = appendViewer(_viewers["slip"],_outputDir + "slip");
    ierr = appendViewer(_viewers["slipVel"],_outputDir + "slipVel");
    ierr = appendViewer(_viewers["tauQSP"],_outputDir + "tauQSP");
    ierr = appendViewer(_viewers["psi"],_outputDir + "psi");
    ierr = appendViewer(_viewers["fault_T"],_outputDir + "fault_T");
  }
  else {
    ierr = VecView(_slip,_viewers["slip"]); CHKERRQ(ierr);
    ierr = VecView(_slipVel,_viewers["slipVel"]); CHKERRQ(ierr);
    ierr = VecView(_tauQSP,_viewers["tauQSP"]); CHKERRQ(ierr);
    ierr = VecView(_psi,_viewers["psi"]); CHKERRQ(ierr);
    ierr = VecView(_T,_viewers["fault_T"]); CHKERRQ(ierr);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}





PetscErrorCode Fault::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting loadData in fault.cpp, loading from file: %s.\n", file);CHKERRQ(ierr);
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

    if (var.compare("bcLTauQS")==0) {
      _bcLTauQS = atoi( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }

    else if (var.compare("DcVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_DcVals);
    }
    else if (var.compare("DcDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_DcDepths);
    }

    else if (var.compare("impedanceVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_impedanceVals);
    }
    else if (var.compare("impedanceDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_impedanceDepths);
    }
    else if (var.compare("sigmaNVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_sigmaNVals);
    }
    else if (var.compare("sigmaNDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_sigmaNDepths);
    }
    else if (var.compare("sigmaN_cap")==0) {
      _sigmaN_cap = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("sigmaN_floor")==0) {
      _sigmaN_floor = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
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

    else if (var.compare("cohesionVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_cohesionVals);
    }
    else if (var.compare("cohesionDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_cohesionDepths);
    }

    else if (var.compare("stateLaw")==0) {
      _stateLaw = line.substr(pos+_delim.length(),line.npos).c_str();
    }

    // tolerance for nonlinear solve
    else if (var.compare("rootTol")==0) {
      _rootTol = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }

    // friction parameters
    else if (var.compare("f0")==0) {
      _f0 = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("v0")==0) {
      _v0 = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }

    // flash heating parameters
    else if (var.compare("fw")==0) {
      _fw = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("Vw")==0) {
      _Vw = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("Tw")==0) {
      _Tw = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("D")==0) {
      _D = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
    else if (var.compare("tau_c")==0) {
      _tau_c = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
    }
  }

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending loadData in fault.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}


// parse input file and load values into data members
PetscErrorCode Fault::loadFieldsFromFiles(std::string inputDir)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting Fault::loadFieldsFromFiles in fault.cpp.\n");CHKERRQ(ierr);
#endif

  // load normal stress: _sNEff
  //~string vecSourceFile = _inputDir + "sigma_N";
  ierr = loadVecFromInputFile(_sNEff,inputDir,"sNEff"); CHKERRQ(ierr);

  // load state: psi
  ierr = loadVecFromInputFile(_psi,inputDir,"psi"); CHKERRQ(ierr);

  // load slip
  ierr = loadVecFromInputFile(_slip,inputDir,"slip"); CHKERRQ(ierr);

  // load quasi-static shear stress
  ierr = loadVecFromInputFile(_tauQSP,inputDir,"tauQS"); CHKERRQ(ierr);

  // load quasi-static shear stress
  ierr = loadVecFromInputFile(_zP,inputDir,"impedence"); CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending Fault::loadFieldsFromFiles in fault.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}






/*
//================= FullFault Functions (both + and - sides) ===========
FullFault::FullFault(Domain&D, HeatEquation& He)
: Fault(D,He),_zM(NULL),
//~ _muArrMinus(D._muArrMinus),_csArrMinus(D._csArrMinus),
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
      //~ // construct fresh each time so the boundaries etc are correct
      //~ Bisect rootAlg(_maxNumIts,_rootTol);
      //~ ierr = rootAlg.setBounds(leftVal,rightVal);CHKERRQ(ierr);
      //~ ierr = rootAlg.findRoot(this,Ii,&outVal);CHKERRQ(ierr);
      //~ _rootIts += rootAlg.getNumIts();

      // construct fresh each time so the boundaries etc are correct
      BracketedNewton rootAlg(_maxNumIts,_rootTol);
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
    ierr =  VecGetValues(_sNEff,1,&Ii,&sigma_N);CHKERRQ(ierr);

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
    ierr = VecSetValue(_sNEff,Ii,sigma_N,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(_tauQSP);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_tauQSMinus);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_zP);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_zM);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_sNEff);CHKERRQ(ierr);

  ierr = VecAssemblyEnd(_tauQSP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_tauQSMinus);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_zP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_zM);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_sNEff);CHKERRQ(ierr);



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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullFault::setTauQS in fault.cpp.\n");CHKERRQ(ierr);
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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullFault::setTauQS in fault.cpp\n");CHKERRQ(ierr);
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

  ierr = VecGetOwnershipRange(_psi,&Istart,&Iend);
  assert( ind>=Istart && ind<Iend );

  //~ierr = VecGetValues(_tempPsi,1,&ind,&psi);CHKERRQ(ierr);
  ierr = VecGetValues(_psi,1,&ind,&psi);CHKERRQ(ierr);
  ierr = VecGetValues(_a,1,&ind,&a);CHKERRQ(ierr);
  ierr = VecGetValues(_sNEff,1,&ind,&sigma_N);CHKERRQ(ierr);

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

PetscErrorCode FullFault::getResid(const PetscInt ind,const PetscScalar slipVel,PetscScalar *out,PetscScalar *J)
{
  PetscErrorCode ierr = 0;
  PetscScalar    psi,a,sigma_N,zPlus,zMinus,tauQSplus,tauQSminus;
  PetscInt       Istart,Iend;

#if VERBOSE > 3
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullFault::getResid in fault.cpp\n");CHKERRQ(ierr);
#endif

  ierr = VecGetOwnershipRange(_psi,&Istart,&Iend);
  assert( ind>=Istart && ind<Iend );

  //~ierr = VecGetValues(_tempPsi,1,&ind,&psi);CHKERRQ(ierr);
  ierr = VecGetValues(_psi,1,&ind,&psi);CHKERRQ(ierr);
  ierr = VecGetValues(_a,1,&ind,&a);CHKERRQ(ierr);
  ierr = VecGetValues(_sNEff,1,&ind,&sigma_N);CHKERRQ(ierr);

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

  *J = 0.; // for now

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
*/





/*
PetscErrorCode FullFault::d_dt(const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 3
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullFault::d_dt in fault.cpp\n");CHKERRQ(ierr);
#endif

  PetscScalar    val,stateVal;
  PetscInt       Ii,Istart,Iend;

  ierr = computeVel();CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(_slipVel,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    ierr = VecGetValues(*varBegin,1,&Ii,&stateVal);
    ierr = agingLaw_psi(Ii,stateVal,val);CHKERRQ(ierr);
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
  ierr = VecView(_sNEff,viewer);CHKERRQ(ierr);
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
      ierr = VecView(_psi,_psiViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_psiViewer);CHKERRQ(ierr);

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
    ierr = VecView(_psi,_psiViewer);CHKERRQ(ierr);

      ierr = VecView(_uM,_uMV);CHKERRQ(ierr);
      ierr = VecView(_velMinus,_velMinusViewer);CHKERRQ(ierr);
      ierr = VecView(_tauQSMinus,_tauQSMinusViewer);CHKERRQ(ierr);
  }

#if VERBOSE > 1
   ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullFault::writeStep in fault.cpp at step %i\n",step);CHKERRQ(ierr);
#endif
  return ierr;
}
*/


