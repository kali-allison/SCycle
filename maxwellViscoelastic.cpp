#include "maxwellViscoelastic.hpp"


SymmMaxwellViscoelastic::SymmMaxwellViscoelastic(Domain& D)
: SymmLinearElastic(D), _file(D._file),_delim(D._delim),_inputDir(D._inputDir),
  _viscDistribution("unspecified"),_visc(NULL),//~_visc(D._visc),
  _gxyP(NULL),_dgxyP(NULL),
  _gxzP(NULL),_dgxzP(NULL),
  _gxyPV(NULL),_dgxyPV(NULL),
  _gxzPV(NULL),_dgxzPV(NULL),
  _gTxyP(NULL),_gTxzP(NULL),
  _gTxyPV(NULL),_gTxzPV(NULL),
  _stressxzP(NULL),_stressxyPV(NULL),_stressxzPV(NULL)
{
  #if VERBOSE > 1
    string funcName = "SymmMaxwellViscoelastic::SymmMaxwellViscoelastic";
    string fileName = "maxwellViscoelastic.cpp";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());
  #endif

  // set viscosity
  loadSettings(_file);
  checkInput();
  setFields(D);


  VecDuplicate(_uP,&_gxyP);
  PetscObjectSetName((PetscObject) _gxyP, "_gxyP");
  VecSet(_gxyP,0.0);
  VecDuplicate(_uP,&_dgxyP);
  PetscObjectSetName((PetscObject) _dgxyP, "_dgxyP");
  VecSet(_dgxyP,0.0);

  VecDuplicate(_uP,&_gxzP);
  PetscObjectSetName((PetscObject) _gxzP, "_gxzP");
  VecSet(_gxzP,0.0);
  VecDuplicate(_uP,&_dgxzP);
  PetscObjectSetName((PetscObject) _dgxzP, "_dgxzP");
  VecSet(_dgxzP,0.0);


  VecDuplicate(_uP,&_gTxyP); VecSet(_gTxyP,0.0);
  VecDuplicate(_uP,&_gTxzP); VecSet(_gTxzP,0.0);
  VecDuplicate(_uP,&_stressxzP); VecSet(_stressxzP,0.0);

  // add viscous strain to integrated variables, stored in _var
  Vec vargxyP; VecDuplicate(_uP,&vargxyP); VecCopy(_gxyP,vargxyP);
  Vec vargxzP; VecDuplicate(_uP,&vargxzP); VecCopy(_gxzP,vargxzP);
  _var.push_back(vargxyP);
  _var.push_back(vargxzP);

  if (_isMMS) { setMMSInitialConditions(); }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
  #endif
}

SymmMaxwellViscoelastic::~SymmMaxwellViscoelastic()
{
  #if VERBOSE > 1
    string funcName = "SymmMaxwellViscoelastic::~SymmMaxwellViscoelastic";
    string fileName = "maxwellViscoelastic.cpp";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());
  #endif

  // from maxwellViscoelastic
  VecDestroy(&_gTxyP);
  VecDestroy(&_gTxzP);
  VecDestroy(&_gxyP);
  VecDestroy(&_gxzP);
  VecDestroy(&_dgxyP);
  VecDestroy(&_dgxzP);

  VecDestroy(&_stressxzP);

  PetscViewerDestroy(&_gTxyPV);
  PetscViewerDestroy(&_gTxzPV);
  PetscViewerDestroy(&_gxyPV);
  PetscViewerDestroy(&_gxzPV);
  PetscViewerDestroy(&_dgxyPV);
  PetscViewerDestroy(&_dgxzPV);
  PetscViewerDestroy(&_stressxyPV);
  PetscViewerDestroy(&_stressxzPV);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
  #endif
}


PetscErrorCode SymmMaxwellViscoelastic::integrate()
{
  PetscErrorCode ierr = 0;
  string funcName = "SymmMaxwellViscoelastic::integrate";
  string fileName = "maxwellViscoelastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif
  double startTime = MPI_Wtime();

  _stepCount++;

  // call odeSolver routine integrate here
  _quadrature->setTolerance(_atol);CHKERRQ(ierr);
  _quadrature->setTimeStepBounds(_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
  ierr = _quadrature->setTimeRange(_initTime,_maxTime);
  ierr = _quadrature->setInitialConds(_var);CHKERRQ(ierr);

  ierr = _quadrature->integrate(this);CHKERRQ(ierr);
  _integrateTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
  CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode SymmMaxwellViscoelastic::d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                 it_vec dvarBegin,it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;
  if (_isMMS) {
    ierr = d_dt_mms(time,varBegin,varEnd,dvarBegin,dvarEnd);CHKERRQ(ierr);
  }
  else {
    ierr = d_dt_eqCycle(time,varBegin,varEnd,dvarBegin,dvarEnd);CHKERRQ(ierr);
  }
  return ierr;
}


PetscErrorCode SymmMaxwellViscoelastic::d_dt_eqCycle(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                 it_vec dvarBegin,it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;
  string funcName = "SymmMaxwellViscoelastic::d_dt_eqCycle";
  string fileName = "maxwellViscoelastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif

  // update boundaries
  ierr = VecCopy(*(varBegin+1),_bcLP);CHKERRQ(ierr);
  ierr = VecScale(_bcLP,0.5);CHKERRQ(ierr);
  ierr = VecSet(_bcRP,_vL*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_bcRP,1.0,_bcRPShift);CHKERRQ(ierr);

  // add source terms to rhs: d/dy( 2*mu*strainV_xy) + d/dz( 2*mu*strainV_xz)
  Vec viscSource;
  ierr = VecDuplicate(_gxyP,&viscSource);CHKERRQ(ierr);
  ierr = setViscStrainSourceTerms(viscSource,varBegin,varEnd);CHKERRQ(ierr);

  // set up rhs vector
  ierr = _sbpP->setRhs(_rhsP,_bcLP,_bcRP,_bcTP,_bcBP);CHKERRQ(ierr); // update rhs from BCs
  ierr = VecAXPY(_rhsP,1.0,viscSource);CHKERRQ(ierr);
  VecDestroy(&viscSource);

  // solve for displacement
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspP,_rhsP,_uP);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  // set shear traction on fault
  ierr = setStresses(time,varBegin,varEnd);CHKERRQ(ierr);
  ierr = _fault.setTauQS(_stressxyP,NULL);CHKERRQ(ierr);

  // set rates
  ierr = _fault.d_dt(varBegin,varEnd, dvarBegin, dvarEnd); // sets rates for slip and state
  ierr = setViscStrainRates(time,varBegin,varEnd,dvarBegin,dvarEnd);CHKERRQ(ierr); // sets viscous strain rates


  // lock the fault to test viscous strain alone
  //~VecSet(*dvarBegin,0.0);
  //~VecSet(*(dvarBegin+1),0.0);
  //~VecSet(*(dvarBegin+2),0.0);
  //~VecSet(*(dvarBegin+3),0.0);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
      CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode SymmMaxwellViscoelastic::d_dt_mms(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                 it_vec dvarBegin,it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;
  string funcName = "SymmMaxwellViscoelastic::d_dt_mms";
  string fileName = "maxwellViscoelastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif

  VecCopy(*(varBegin+2),_gxyP);
  VecCopy(*(varBegin+3),_gxzP);

  // create rhs: set boundary conditions, set rhs, add source terms
  ierr = setMMSBoundaryConditions(time);CHKERRQ(ierr); // modifies _bcLP,_bcRP,_bcTP, and _bcBP
  ierr = _sbpP->setRhs(_rhsP,_bcLP,_bcRP,_bcTP,_bcBP);CHKERRQ(ierr);

  Vec viscSourceMMS,HxviscSourceMMS,viscSource,uSource,HxuSource;
  ierr = VecDuplicate(_uP,&viscSource);CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&viscSourceMMS);CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&HxviscSourceMMS);CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&uSource);CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&HxuSource);CHKERRQ(ierr);

  ierr = setViscStrainSourceTerms(viscSource,_var.begin(),_var.end());CHKERRQ(ierr);
  mapToVec(viscSourceMMS,MMS_gSource,_Nz,_dy,_dz,time);
  ierr = _sbpP->H(viscSourceMMS,HxviscSourceMMS);
  VecDestroy(&viscSourceMMS);
  mapToVec(uSource,MMS_uSource,_Nz,_dy,_dz,time);
  ierr = _sbpP->H(uSource,HxuSource);
  VecDestroy(&uSource);

  ierr = VecAXPY(_rhsP,1.0,viscSource);CHKERRQ(ierr); // add d/dy mu*epsVxy + d/dz mu*epsVxz
  ierr = VecAXPY(_rhsP,1.0,HxviscSourceMMS);CHKERRQ(ierr); // add MMS source for viscous strains
  ierr = VecAXPY(_rhsP,1.0,HxuSource);CHKERRQ(ierr); // add MMS source for u
  VecDestroy(&HxviscSourceMMS);
  VecDestroy(&HxuSource);


  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspP,_rhsP,_uP);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  //~mapToVec(_uP,MMS_uA,_Nz,_dy,_dz,time);

  // update fields on fault
  ierr = setStresses(time,varBegin,varEnd);CHKERRQ(ierr);

  // update rates
  VecSet(*dvarBegin,0.0); // d/dt psi
  VecSet(*(dvarBegin+1),0.0); // d/dt slip
  //~VecSet(*(dvarBegin+2),0.0);
  //~VecSet(*(dvarBegin+3),0.0);

  ierr = setViscStrainRates(time,varBegin,varEnd,dvarBegin,dvarEnd);CHKERRQ(ierr); // sets viscous strain rates

  //~mapToVec(*(dvarBegin+2),MMS_gxy_t,_Nz,_dy,_dz,time);
  //~mapToVec(*(dvarBegin+3),MMS_gxz_t,_Nz,_dy,_dz,time);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode SymmMaxwellViscoelastic::setViscStrainSourceTerms(Vec& out,const_it_vec varBegin,const_it_vec varEnd)
{
  PetscErrorCode ierr = 0;
  string funcName = "SymmMaxwellViscoelastic::setViscStrainSourceTerms";
  string fileName = "maxwellViscoelastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif

  Vec source;
  VecDuplicate(_gxyP,&source);
  VecSet(source,0.0);

  // add source terms to rhs: d/dy( mu*gxy) + d/dz( mu*gxz)
  // + Hz^-1 E0z mu gxz + Hz^-1 ENz mu gxz
  Vec sourcexy_y;
  VecDuplicate(_uP,&sourcexy_y);
  ierr = _sbpP->Dyxmu(*(varBegin+2),sourcexy_y);CHKERRQ(ierr);
  ierr = VecCopy(sourcexy_y,source);CHKERRQ(ierr); // sourcexy_y -> source
  VecDestroy(&sourcexy_y);

  if (_Nz > 1)
  {
    Vec sourcexz_z;
    VecDuplicate(_gxzP,&sourcexz_z);
    ierr = _sbpP->Dzxmu(*(varBegin+3),sourcexz_z);CHKERRQ(ierr);
    ierr = VecAXPY(source,1.0,sourcexz_z);CHKERRQ(ierr); // source += Hxsourcexz_z
    VecDestroy(&sourcexz_z);

    // enforce traction boundary condition
    Vec temp1,bcT,bcB;
    VecDuplicate(_gxzP,&temp1);
    VecDuplicate(_gxzP,&bcT);
    VecDuplicate(_gxzP,&bcB);

    _sbpP->HzinvxE0z(*(varBegin+3),temp1);
    ierr = MatMult(_muP,temp1,bcT); CHKERRQ(ierr);

    _sbpP->HzinvxENz(*(varBegin+3),temp1);
    ierr = MatMult(_muP,temp1,bcB); CHKERRQ(ierr);

    ierr = VecAXPY(source,1.0,bcT);CHKERRQ(ierr);
    ierr = VecAXPY(source,-1.0,bcB);CHKERRQ(ierr);

    VecDestroy(&temp1);
    VecDestroy(&bcT);
    VecDestroy(&bcB);
  }


  ierr = _sbpP->H(source,out); CHKERRQ(ierr);
  VecDestroy(&source);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
      CHKERRQ(ierr);
  #endif
  return ierr = 0;
}



PetscErrorCode SymmMaxwellViscoelastic::setViscousStrainRateSAT(Vec &u, Vec &gL, Vec &gR, Vec &out)
{
    PetscErrorCode ierr = 0;
    string funcName = "SymmMaxwellViscoelastic::viscousStrainRateSAT";
    string fileName = "maxwellViscoelastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif

  VecSet(out,0.0);

  Vec GL, GR,temp1;
  VecDuplicate(u,&GL);
  VecDuplicate(u,&GR);
  VecDuplicate(u,&temp1);

  // left displacement boundary
  ierr = _sbpP->HyinvxE0y(u,temp1);CHKERRQ(ierr);
  ierr = _sbpP->Hyinvxe0y(gL,GL);CHKERRQ(ierr);
  VecAXPY(out,1.0,temp1);
  VecAXPY(out,-1.0,GL);

  // right displacement boundary
  ierr = _sbpP->HyinvxENy(u,temp1);CHKERRQ(ierr);
  ierr = _sbpP->HyinvxeNy(gR,GR);CHKERRQ(ierr);
  VecAXPY(out,-1.0,temp1);
  VecAXPY(out,1.0,GR);

  VecDestroy(&GL);
  VecDestroy(&GR);
  VecDestroy(&temp1);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
      CHKERRQ(ierr);
  #endif
  return ierr = 0;
}



PetscErrorCode SymmMaxwellViscoelastic::setViscStrainRates(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                 it_vec dvarBegin,it_vec dvarEnd)
{
    PetscErrorCode ierr = 0;
    string funcName = "SymmMaxwellViscoelastic::setViscStrainRates";
    string fileName = "maxwellViscoelastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif


  // add SAT terms to strain rate for epsxy
  Vec SAT;
  VecDuplicate(_gTxyP,&SAT);
  ierr = setViscousStrainRateSAT(_uP,_bcLP,_bcRP,SAT);CHKERRQ(ierr);

  PetscScalar deps,visc,epsVisc,sat,sigmaxy,sigmaxz=0;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(*(dvarBegin+2),&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    VecGetValues(_visc,1,&Ii,&visc);
    VecGetValues(_stressxyP,1,&Ii,&sigmaxy);
    VecGetValues(*(varBegin+2),1,&Ii,&epsVisc);
    VecGetValues(SAT,1,&Ii,&sat);

    // d/dt gxy = mu/visc * ( d/dy u - gxy) + SAT
    deps = sigmaxy/visc + _muArrPlus[Ii]/visc * sat;
    VecSetValues(*(dvarBegin+2),1,&Ii,&deps,INSERT_VALUES);

    if (_Nz > 1) {
      VecGetValues(_stressxzP,1,&Ii,&sigmaxz);

      // d/dt gxz = mu/visc * ( *d/dz u - gxz)
      deps = sigmaxz/visc;
      VecSetValues(*(dvarBegin+3),1,&Ii,&deps,INSERT_VALUES);
    }
  }
  VecAssemblyBegin(*(dvarBegin+2));
  VecAssemblyEnd(*(dvarBegin+2));

  VecDestroy(&SAT);


  if (_Nz > 1) {
    VecAssemblyBegin(*(dvarBegin+3));
    VecAssemblyEnd(*(dvarBegin+3));
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
      CHKERRQ(ierr);
  #endif
  return ierr = 0;
}


PetscErrorCode SymmMaxwellViscoelastic::setStresses(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd)
{
    PetscErrorCode ierr = 0;
    string funcName = "SymmMaxwellViscoelastic::setStresses";
    string fileName = "maxwellViscoelastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif

  // compute strains and rates
  _sbpP->Dy(_uP,_gTxyP);
  _sbpP->Dz(_uP,_gTxzP);

  PetscScalar visc,gT,gV,sigmaxy,sigmaxz;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(_gTxyP,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    VecGetValues(_visc,1,&Ii,&visc);
    VecGetValues(_gTxyP,1,&Ii,&gT);
    VecGetValues(*(varBegin+2),1,&Ii,&gV);

    // solve for stressxyP = 2*mu*epsExy (elastic strain)
    //                     = 2*mu*(0.5*d/dy(uhat) - epsVxy)
    //                     = mu*d/dy(uhat) - gxy
    sigmaxy = _muArrPlus[Ii] * (gT - gV);
    VecSetValues(_stressxyP,1,&Ii,&sigmaxy,INSERT_VALUES);

    if (_Nz > 1) {
      VecGetValues(_gTxzP,1,&Ii,&gT);
      VecGetValues(*(varBegin+3),1,&Ii,&gV);

      // solve for stressxzP = 2*mu*epsExy (elastic strain)
      //                     = 2*mu*(0.5*d/dz(uhat) - epsVxz)
      sigmaxz = _muArrPlus[Ii] * (gT - gV);
      VecSetValues(_stressxzP,1,&Ii,&sigmaxz,INSERT_VALUES);
    }
  }
  VecAssemblyBegin(_stressxyP);
  VecAssemblyEnd(_stressxyP);


  if (_Nz > 1) {
    VecAssemblyBegin(_stressxzP);
    VecAssemblyEnd(_stressxzP);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
      CHKERRQ(ierr);
  #endif
  return ierr = 0;
}


PetscErrorCode SymmMaxwellViscoelastic::setMMSInitialConditions()
{
  PetscErrorCode ierr = 0;
  string funcName = "SymmLinearElastic::setMMSInitialConditions";
  string fileName = "lithosphere.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
  #endif

  PetscScalar time = _initTime;
  mapToVec(_visc,MMS_visc,_Nz,_dy,_dz);
  //~mapToVec(_uP,MMS_uA,_Nz,_dy,_dz,time);
  mapToVec(_gxyP,MMS_gxy,_Nz,_dy,_dz,time);
  mapToVec(_gxzP,MMS_gxz,_Nz,_dy,_dz,time);
  VecCopy(_gxyP,*(_var.begin()+2));
  VecCopy(_gxzP,*(_var.begin()+3));

  // create rhs: set boundary conditions, set rhs, add source terms
  ierr = setMMSBoundaryConditions(time);CHKERRQ(ierr); // modifies _bcLP,_bcRP,_bcTP, and _bcBP
  ierr = _sbpP->setRhs(_rhsP,_bcLP,_bcRP,_bcTP,_bcBP);CHKERRQ(ierr);

  Vec viscSourceMMS,HxviscSourceMMS,viscSource,uSource,HxuSource;
  ierr = VecDuplicate(_uP,&viscSource);CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&viscSourceMMS);CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&HxviscSourceMMS);CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&uSource);CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&HxuSource);CHKERRQ(ierr);

  ierr = setViscStrainSourceTerms(viscSource,_var.begin(),_var.end());CHKERRQ(ierr);
  mapToVec(viscSourceMMS,MMS_gSource,_Nz,_dy,_dz,time);
  ierr = _sbpP->H(viscSourceMMS,HxviscSourceMMS);
  VecDestroy(&viscSourceMMS);
  mapToVec(uSource,MMS_uSource,_Nz,_dy,_dz,time);
  ierr = _sbpP->H(uSource,HxuSource);
  VecDestroy(&uSource);

  ierr = VecAXPY(_rhsP,1.0,viscSource);CHKERRQ(ierr); // add d/dy mu*epsVxy + d/dz mu*epsVxz
  ierr = VecAXPY(_rhsP,1.0,HxviscSourceMMS);CHKERRQ(ierr); // add MMS source for viscous strains
  ierr = VecAXPY(_rhsP,1.0,HxuSource);CHKERRQ(ierr); // add MMS source for u
  VecDestroy(&HxviscSourceMMS);
  VecDestroy(&HxuSource);


  // solve for displacement
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspP,_rhsP,_uP);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  // set stresses
  ierr = setStresses(time,_var.begin(),_var.end());CHKERRQ(ierr);

  #if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());
  #endif
  return ierr;
}

PetscErrorCode SymmMaxwellViscoelastic::setMMSBoundaryConditions(const double time)
{
  PetscErrorCode ierr = 0;
  string funcName = "SymmMaxwellViscoelastic::setMMSBoundaryConditions";
  string fileName = "maxwellViscoelastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
  #endif

  // set up boundary conditions: L and R
  PetscScalar y,z,v;
  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(_bcLP,&Istart,&Iend);CHKERRQ(ierr);
  for(Ii=Istart;Ii<Iend;Ii++) {
    z = _dz * Ii;

    y = 0;
    if (!_bcLType.compare("displacement")) { v = MMS_uA(y,z,time); } // uAnal(y=0,z)
    else if (!_bcLType.compare("traction")) { v = MMS_mu(y,z) * (MMS_uA_y(y,z,time) - MMS_gxy(y,z,time)); } // sigma_xy = mu * d/dy u
    ierr = VecSetValues(_bcLP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

    y = _Ly;
    if (!_bcRType.compare("displacement")) { v = MMS_uA(y,z,time); } // uAnal(y=Ly,z)
    else if (!_bcRType.compare("traction")) { v = MMS_mu(y,z) * (MMS_uA_y(y,z,time)- MMS_gxy(y,z,time)); } // sigma_xy = mu * d/dy u
    ierr = VecSetValues(_bcRP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_bcLP);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_bcRP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcLP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcRP);CHKERRQ(ierr);

  // set up boundary conditions: T and B
  ierr = VecGetOwnershipRange(_bcLP,&Istart,&Iend);CHKERRQ(ierr);
  for(Ii=Istart;Ii<Iend;Ii++) {
    y = _dy * Ii;

    z = 0;
    if (!_bcTType.compare("displacement")) { v = MMS_uA(y,z,time); } // uAnal(y,z=0)
    else if (!_bcTType.compare("traction")) { v = MMS_mu(y,z) * (MMS_uA_z(y,z,time) - MMS_gxz(y,z,time)); }
    //~else if (!_bcTType.compare("traction")) { v = MMS_mu(y,z) * (MMS_uA_z(y,z,time)); }
    ierr = VecSetValues(_bcTP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

    z = _Lz;
    if (!_bcBType.compare("displacement")) { v = MMS_uA(y,z,time); } // uAnal(y,z=Lz)
    else if (!_bcBType.compare("traction")) { v = MMS_mu(y,z) * (MMS_uA_z(y,z,time)- MMS_gxz(y,z,time)); }
    else if (!_bcBType.compare("traction")) { v = MMS_mu(y,z) * (MMS_uA_z(y,z,time)); }
    ierr = VecSetValues(_bcBP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_bcTP);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_bcBP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcTP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcBP);CHKERRQ(ierr);

  #if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());
  #endif
  return ierr;
}


PetscErrorCode SymmMaxwellViscoelastic::measureMMSError()
{
  PetscErrorCode ierr = 0;

  // measure error between analytical and numerical solution
  Vec uA,gxyA,gxzA;
  VecDuplicate(_uP,&uA);
  VecDuplicate(_uP,&gxyA);
  VecDuplicate(_uP,&gxzA);
  mapToVec(uA,MMS_uA,_Nz,_dy,_dz,_currTime);
  mapToVec(gxyA,MMS_gxy,_Nz,_dy,_dz,_currTime);
  mapToVec(gxzA,MMS_gxz,_Nz,_dy,_dz,_currTime);

  double err2u = computeNormDiff_2(_uP,uA);
  double err2epsxy = computeNormDiff_2(*(_var.begin()+2),gxyA);
  double err2epsxz = computeNormDiff_2(_gxzP,gxzA);

  PetscPrintf(PETSC_COMM_WORLD,"%3i %3i %.4e %.4e % .15e %.4e % .15e %.4e % .15e\n",
              _order,_Ny,_dy,err2u,log2(err2u),err2epsxy,log2(err2epsxy),err2epsxz,log2(err2epsxz));

  VecDestroy(&uA);
  VecDestroy(&gxyA);
  VecDestroy(&gxzA);
  return ierr;
}


PetscErrorCode SymmMaxwellViscoelastic::timeMonitor(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec varEnd,
                             const_it_vec dvarBegin,const_it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;
  _stepCount++;
  _currTime = time;
  if ( stepCount % _stride1D == 0) {
    //~ierr = PetscViewerHDF5IncrementTimestep(D->viewer);CHKERRQ(ierr);
    ierr = writeStep1D();CHKERRQ(ierr);
  }

  if ( stepCount % _stride2D == 0) {
    //ierr = PetscViewerHDF5IncrementTimestep(D->viewer);CHKERRQ(ierr);
    ierr = writeStep2D();CHKERRQ(ierr);
  }

#if VERBOSE > 0
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %.15e\n",stepCount,_currTime);CHKERRQ(ierr);
#endif
  return ierr;
}

// Outputs data at each time step.
PetscErrorCode SymmMaxwellViscoelastic::debug(const PetscReal time,const PetscInt stepCount,
                     const_it_vec varBegin,const_it_vec varEnd,
                     const_it_vec dvarBegin,const_it_vec dvarEnd,const char *stage)
{
  PetscErrorCode ierr = 0;

#if ODEPRINT > 0
  PetscInt       Istart,Iend;
  PetscScalar    bcRval,uVal,psiVal,velVal,dQVal,tauQS;
  PetscScalar    epsVxy,depsVxy;

  //~PetscScalar k = _muArrPlus[0]/2/_Ly;

  ierr= VecGetOwnershipRange(*varBegin,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(*varBegin,1,&Istart,&psiVal);CHKERRQ(ierr);

  ierr = VecGetValues(*(varBegin+1),1,&Istart,&uVal);CHKERRQ(ierr);

  ierr= VecGetOwnershipRange(*dvarBegin,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(*dvarBegin,1,&Istart,&dQVal);CHKERRQ(ierr);
  ierr = VecGetValues(*(dvarBegin+1),1,&Istart,&velVal);CHKERRQ(ierr);

  ierr = VecGetValues(*(varBegin+2),1,&Istart,&epsVxy);CHKERRQ(ierr);
  ierr = VecGetValues(*(dvarBegin+2),1,&Istart,&depsVxy);CHKERRQ(ierr);

  ierr= VecGetOwnershipRange(_bcRP,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(_bcRP,1,&Istart,&bcRval);CHKERRQ(ierr);

  ierr = VecGetValues(_fault._tauQSP,1,&Istart,&tauQS);CHKERRQ(ierr);

  if (stepCount == 0) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%-4s %-6s  | %-15s %-15s %-15s | %-15s %-15s %-16s | %-15s\n",
                       "Step","Stage", "bcR","D","eVxy", "tauQS","V","deVxy","time");
    CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%4i %-6s ",stepCount,stage);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," | %.9e %.9e %.9e ",bcRval,uVal,epsVxy);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," | %.9e %.9e %.9e ",tauQS,velVal,depsVxy);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," | %.9e\n",time);CHKERRQ(ierr);


  //~VecView(_fault._tauQSP,PETSC_VIEWER_STDOUT_WORLD);
#endif
  return ierr;
}


PetscErrorCode SymmMaxwellViscoelastic::writeStep1D()
{
  PetscErrorCode ierr = 0;
  string funcName = "SymmMaxwellViscoelastic::writeStep1D";
  string fileName = "maxwellViscoelastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at step %i\n",funcName.c_str(),fileName.c_str(),_stepCount);
    CHKERRQ(ierr);
  #endif

  double startTime = MPI_Wtime();

  if (_stepCount==0) {
    // write contextual fields
    //~ierr = _sbpP->writeOps(_outputDir);CHKERRQ(ierr);
    ierr = _fault.writeContext(_outputDir);CHKERRQ(ierr);

    // output viscosity vector
    string str =  _outputDir + "visc";
    PetscViewer viewer;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(_visc,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(_outputDir+"time.txt").c_str(),&_timeV1D);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_timeV1D, "%.15e\n",_currTime);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDispPlus").c_str(),
                                 FILE_MODE_WRITE,&_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_surfDispPlus,_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDispPlus").c_str(),
                                   FILE_MODE_APPEND,&_surfDispPlusViewer);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcR").c_str(),
              FILE_MODE_WRITE,&_bcRPlusV);CHKERRQ(ierr);
    ierr = VecView(_bcRP,_bcRPlusV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_bcRPlusV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcR").c_str(),
                                   FILE_MODE_APPEND,&_bcRPlusV);CHKERRQ(ierr);

    ierr = _fault.writeStep(_outputDir,_stepCount);CHKERRQ(ierr);
  }
  else {
    ierr = PetscViewerASCIIPrintf(_timeV1D, "%.15e\n",_currTime);CHKERRQ(ierr);
    ierr = _fault.writeStep(_outputDir,_stepCount);CHKERRQ(ierr);

    ierr = VecView(_surfDispPlus,_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_bcRP,_bcRPlusV);CHKERRQ(ierr);
  }

  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s at step %i\n",funcName.c_str(),fileName.c_str(),_stepCount);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode SymmMaxwellViscoelastic::writeStep2D()
{
  PetscErrorCode ierr = 0;
  string funcName = "SymmMaxwellViscoelastic::writeStep2D";
  string fileName = "maxwellViscoelastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at step %i\n",funcName.c_str(),fileName.c_str(),_stepCount);
    CHKERRQ(ierr);
  #endif

  double startTime = MPI_Wtime();

  if (_stepCount==0) {
    // write contextual fields
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(_outputDir+"time2D.txt").c_str(),&_timeV2D);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_timeV2D, "%.15e\n",_currTime);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"uBodyP").c_str(),
              FILE_MODE_WRITE,&_uPV);CHKERRQ(ierr);
    ierr = VecView(_uP,_uPV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_uPV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"uBodyP").c_str(),
                                   FILE_MODE_APPEND,&_uPV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gTxyP").c_str(),
              FILE_MODE_WRITE,&_gTxyPV);CHKERRQ(ierr);
    ierr = VecView(_gTxyP,_gTxyPV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_gTxyPV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gTxyP").c_str(),
                                   FILE_MODE_APPEND,&_gTxyPV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"stressxyP").c_str(),
              FILE_MODE_WRITE,&_stressxyPV);CHKERRQ(ierr);
    ierr = VecView(_stressxyP,_stressxyPV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_stressxyPV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"stressxyP").c_str(),
                                   FILE_MODE_APPEND,&_stressxyPV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gxyP").c_str(),
              FILE_MODE_WRITE,&_gxyPV);CHKERRQ(ierr);
    ierr = VecView(_gxyP,_gxyPV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_gxyPV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gxyP").c_str(),
                                   FILE_MODE_APPEND,&_gxyPV);CHKERRQ(ierr);
    //~if (_isMMS) {
      //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"uAnal").c_str(),
                //~FILE_MODE_WRITE,&_uAnalV);CHKERRQ(ierr);
      //~ierr = VecView(_uAnal,_uAnalV);CHKERRQ(ierr);
      //~ierr = PetscViewerDestroy(&_uAnalV);CHKERRQ(ierr);
      //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"uAnal").c_str(),
                                     //~FILE_MODE_APPEND,&_uAnalV);CHKERRQ(ierr);
    //~}
    if (_Nz>1)
    {
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gxzP").c_str(),
              FILE_MODE_WRITE,&_gTxzPV);CHKERRQ(ierr);
      ierr = VecView(_gTxzP,_gTxzPV);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_gTxzPV);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gxzP").c_str(),
                                     FILE_MODE_APPEND,&_gTxzPV);CHKERRQ(ierr);

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"stressxzP").c_str(),
               FILE_MODE_WRITE,&_stressxzPV);CHKERRQ(ierr);
      ierr = VecView(_stressxzP,_stressxzPV);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_stressxzPV);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"stressxzP").c_str(),
                                     FILE_MODE_APPEND,&_stressxzPV);CHKERRQ(ierr);

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gxzP").c_str(),
               FILE_MODE_WRITE,&_gxzPV);CHKERRQ(ierr);
      ierr = VecView(_gxzP,_gxzPV);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_gxzPV);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gxzP").c_str(),
                                   FILE_MODE_APPEND,&_gxzPV);CHKERRQ(ierr);
    }
  }
  else {
    ierr = PetscViewerASCIIPrintf(_timeV2D, "%.15e\n",_currTime);CHKERRQ(ierr);

    ierr = VecView(_uP,_uPV);CHKERRQ(ierr);
    ierr = VecView(_gTxyP,_gTxyPV);CHKERRQ(ierr);
    ierr = VecView(_stressxyP,_stressxyPV);CHKERRQ(ierr);
    ierr = VecView(_gxyP,_gxyPV);CHKERRQ(ierr);
    //~if (_isMMS) {ierr = VecView(_uAnal,_uAnalV);CHKERRQ(ierr);}
    if (_Nz>1)
    {
      ierr = VecView(_gTxzP,_gTxzPV);CHKERRQ(ierr);
      ierr = VecView(_stressxzP,_stressxzPV);CHKERRQ(ierr);
      ierr = VecView(_gxzP,_gxzPV);CHKERRQ(ierr);
    }
  }

  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s at step %i\n",funcName.c_str(),fileName.c_str(),_stepCount);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode SymmMaxwellViscoelastic::view()
{
  PetscErrorCode ierr = 0;
  ierr = _quadrature->view();
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in integration (s): %g\n",_integrateTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of times linear system was solved: %i\n",_linSolveCount);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent solving linear system (s): %g\n",_linSolveTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  return ierr;
}



// loads settings from the input text file
PetscErrorCode SymmMaxwellViscoelastic::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting loadData in maxwellViscoelastic.cpp, loading from file: %s.\n", file);CHKERRQ(ierr);
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

    // viscosity for asthenosphere
    if (var.compare("viscDistribution")==0) {
      _viscDistribution = line.substr(pos+_delim.length(),line.npos).c_str();
    }

    // if explicity setting viscosity distribution as layered
    else if (var.compare("viscVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_viscVals);
    }
    else if (var.compare("viscDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_viscDepths);
    }

    // if using effective viscosity from power law
    else if (var.compare("AVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_AVals);
    }
    else if (var.compare("ADepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_ADepths);
    }
    else if (var.compare("BVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_BVals);
    }
    else if (var.compare("BDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_BDepths);
    }
    else if (var.compare("nVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_nVals);
    }
    else if (var.compare("nDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_nDepths);
    }
    else if (var.compare("sigmadevVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_sigmadevVals);
    }
    else if (var.compare("sigmadevDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_sigmadevDepths);
    }

  }

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending loadData in maxwellViscoelastic.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}

// set viscosity
PetscErrorCode SymmMaxwellViscoelastic::setFields(Domain& D)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting setFields in maxwellViscoelastic.cpp.\n");CHKERRQ(ierr);
#endif

    ierr = VecCreate(PETSC_COMM_WORLD,&_visc);CHKERRQ(ierr);
    ierr = VecSetSizes(_visc,PETSC_DECIDE,_Ny*_Nz);CHKERRQ(ierr);
    ierr = VecSetFromOptions(_visc);CHKERRQ(ierr);

  if (_viscDistribution.compare("effectiveVisc")==0) {
    std::string _thermalCoupling;
    HeatEquation _he(D);
    Vec         _A,_n,_B;
    ierr = VecDuplicate(_uP,&_A);CHKERRQ(ierr);
    ierr = VecDuplicate(_uP,&_B);CHKERRQ(ierr);
    ierr = VecDuplicate(_uP,&_n);CHKERRQ(ierr);
    Vec sigmadev;
    ierr = VecDuplicate(_uP,&sigmadev);CHKERRQ(ierr);
    if (_Nz == 1) {
      VecSet(_A,_AVals[0]);
      VecSet(_B,_BVals[0]);
      VecSet(_n,_nVals[0]);
      VecSet(sigmadev,_sigmadevVals[0]);
    }
    else {
      ierr = setVecFromVectors(_A,_AVals,_ADepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_B,_BVals,_BDepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_n,_nVals,_nDepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(sigmadev,_sigmadevVals,_sigmadevDepths);CHKERRQ(ierr);
    }

    // compute effective viscosity using heat equation's computed temperature
    PetscScalar s,A,B,n,T,effVisc,invVisc=0;
    PetscInt Ii,Istart,Iend;
    VecGetOwnershipRange(_A,&Istart,&Iend);
    for (Ii=Istart;Ii<Iend;Ii++) {
      VecGetValues(sigmadev,1,&Ii,&s);
      VecGetValues(_A,1,&Ii,&A);
      VecGetValues(_B,1,&Ii,&B);
      VecGetValues(_n,1,&Ii,&n);
      VecGetValues(_he._T,1,&Ii,&T);
      invVisc = A*pow(s,n-1.0)*exp(-B/T) * 1e-3; // *1e-3 to get resulting eff visc in GPa s
      effVisc = 1.0/invVisc;
      VecSetValues(_visc,1,&Ii,&effVisc,INSERT_VALUES);
      assert(!isnan(invVisc));
    }
    VecAssemblyBegin(_visc);
    VecAssemblyEnd(_visc);
    VecDestroy(&sigmadev);
    VecDestroy(&_A);
    VecDestroy(&_B);
    VecDestroy(&_n);
  }
  else if (_viscDistribution.compare("layered")==0) {
    PetscInt       Ii;
    PetscScalar    v=0,z=0;
    PetscScalar z0,z1,v0,v1;
    PetscInt Istart,Iend;
    ierr = VecGetOwnershipRange(_visc,&Istart,&Iend);CHKERRQ(ierr);

    // build viscosity structure from generalized input
    size_t vecLen = _viscDepths.size();
    for (Ii=Istart;Ii<Iend;Ii++)
    {
      z = _dz*(Ii-_Nz*(Ii/_Nz));
      //~PetscPrintf(PETSC_COMM_WORLD,"1: Ii = %i, z = %g\n",Ii,z);
      for (size_t ind = 0; ind < vecLen-1; ind++) {
          z0 = _viscDepths[0+ind];
          z1 = _viscDepths[0+ind+1];
          v0 = log10(_viscVals[0+ind]);
          v1 = log10(_viscVals[0+ind+1]);

          if (z>=z0 && z<=z1) {
            v = (v1 - v0)/(z1-z0) * (z-z0) + v0;
            v = pow(10,v);
            }
          ierr = VecSetValues(_visc,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = VecAssemblyBegin(_visc);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(_visc);CHKERRQ(ierr);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: viscDistribution type not understood\n");
    assert(0); // automatically fail
  }

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending setFields in maxwellViscoelastic.cpp.\n");CHKERRQ(ierr);
#endif
return ierr;
}

 //parse input file and load values into data members
PetscErrorCode SymmMaxwellViscoelastic::loadFieldsFromFiles()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting loadFieldsFromFiles in maxwellViscoelastic.cpp.\n");CHKERRQ(ierr);
#endif

  // load viscosity from input file
  ierr = VecCreate(PETSC_COMM_WORLD,&_visc);CHKERRQ(ierr);
  ierr = VecSetSizes(_visc,PETSC_DECIDE,_Ny*_Nz);CHKERRQ(ierr);
  ierr = VecSetFromOptions(_visc);
  PetscObjectSetName((PetscObject) _visc, "_visc");
  ierr = loadVecFromInputFile(_visc,_inputDir, "visc");CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending loadFieldsFromFiles in maxwellViscoelastic.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}


// Check that required fields have been set by the input file
PetscErrorCode SymmMaxwellViscoelastic::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting Domain::checkInputPlus in maxwellViscoelastic.cpp.\n");CHKERRQ(ierr);
  #endif

  assert(_viscDistribution.compare("layered")==0 ||
      _viscDistribution.compare("mms")==0 ||
      _viscDistribution.compare("loadFromFile")==0 ||
      _viscDistribution.compare("effectiveVisc")==0 );

  if (_viscDistribution.compare("effectiveVisc")==0) {
    assert(_AVals.size() == _ADepths.size() );
    assert(_BVals.size() == _BDepths.size() );
    assert(_nVals.size() == _nDepths.size() );
    assert(_sigmadevVals.size() == _sigmadevDepths.size() );
    assert(_AVals.size() > 0);
    assert(_BVals.size() > 0);
    assert(_nVals.size() > 0);
    assert(_sigmadevVals.size() > 0);
  }

  assert(_viscVals.size() == _viscDepths.size() );
  if (_viscDistribution.compare("layered")==0) {
    assert(_viscVals.size() > 0);
  }

#if VERBOSE > 1
ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending Domain::checkInputPlus in maxwellViscoelastic.cpp.\n");CHKERRQ(ierr);
#endif
  //~}
  return ierr;
}


// Fills vec with the linear interpolation between the pairs of points (vals,depths)
PetscErrorCode SymmMaxwellViscoelastic::setVecFromVectors(Vec& vec, vector<double>& vals,vector<double>& depths)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    v,z,z0,z1,v0,v1;
  #if VERBOSE > 1
    std::string funcName = "SymmMaxwellViscoelastic::setVecFromVectors";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // build structure from generalized input
  size_t vecLen = depths.size();
  ierr = VecGetOwnershipRange(vec,&Istart,&Iend);CHKERRQ(ierr);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++)
  {
    z = _dz*(Ii-_Nz*(Ii/_Nz));
    //~PetscPrintf(PETSC_COMM_WORLD,"1: Ii = %i, z = %g\n",Ii,z);
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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}
