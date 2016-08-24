#include "maxwellViscoelastic.hpp"


SymmMaxwellViscoelastic::SymmMaxwellViscoelastic(Domain& D)
: SymmLinearElastic(D), _file(D._file),_delim(D._delim),_inputDir(D._inputDir),
  _viscDistribution("unspecified"),_visc(NULL),
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

  if (D._loadICs==1) { loadFieldsFromFiles(); }
  //~ _fault.computeVel();

// if also solving heat equation
  if (_thermalCoupling.compare("coupled")==0 || _thermalCoupling.compare("uncoupled")==0) {
    Vec T;
    VecDuplicate(_uP,&T);
    VecCopy(_he._T,T);
    _varIm.push_back(T);
  }

  #if CALCULATE_ENERGY == 1
    // remove E that was added by SymmLinearElastic
    Vec  vec = * (_var.end() - 1);
    VecDestroy(&vec);
    _var.pop_back();
  #endif

  // add viscous strain to integrated variables, stored in _var
  Vec vargxyP; VecDuplicate(_uP,&vargxyP); VecCopy(_gxyP,vargxyP);
  Vec vargxzP; VecDuplicate(_uP,&vargxzP); VecCopy(_gxzP,vargxzP);
  _var.push_back(vargxyP);
  _var.push_back(vargxzP);

  if (_isMMS) { setMMSInitialConditions(); }

  #if CALCULATE_ENERGY == 1
    Vec E;
    VecDuplicate(_E,&E);
    VecCopy(_E,E);
    _var.push_back(E);

    VecDuplicate(_uP,&_uPPrev);
    VecCopy(_uP,_uPPrev);
  #endif

    //~ writeVec(_bcRP,"test/bcR");
    //~ writeVec(_fault._tauQSP,"test/tauQS");
    //~ writeVec(_fault._slipVel,"test/slipVel");
    //~ writeVec(_uP,"test/u");
    //~ writeVec(_stressxyP,"test/stressxyP");
    //~ writeVec(_stressxzP,"test/stressxzP");
    //~ writeVec(_gxyP,"test/gxy");
    //~ writeVec(_gxzP,"test/gxz");

    //~ writeVec(*(_var.begin()+0),"test/state");
    //~ writeVec(*(_var.begin()+1),"test/slip");
    //~ writeVec(*(_var.begin()+2),"test/gxy");
    //~ writeVec(*(_var.begin()+3),"test/gxz");
    //~ assert(0);

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

  VecDestroy(&_bcRPShift);
  for(std::vector<Vec>::size_type i = 0; i != _var.size(); i++) {
    VecDestroy(&_var[i]);
  }


  // from maxwellViscoelastic
  VecDestroy(&_visc);

  VecDestroy(&_gTxyP);
  VecDestroy(&_gTxzP);
  VecDestroy(&_gxyP);
  VecDestroy(&_gxzP);
  VecDestroy(&_dgxyP);
  VecDestroy(&_dgxzP);

  VecDestroy(&_stressxyP);
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
  if (_timeIntegrator.compare("IMEX")==0) {
    _quadImex->setTolerance(_atol);CHKERRQ(ierr);
    _quadImex->setTimeStepBounds(_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
    ierr = _quadImex->setTimeRange(_initTime,_maxTime);
    ierr = _quadImex->setInitialConds(_var,_varIm);CHKERRQ(ierr);

    // control which fields are used to select step size
    int arrInds[] = {1}; // state: 0, slip: 1
    std::vector<int> errInds(arrInds,arrInds+1); // !! UPDATE THIS LINE TOO
    ierr = _quadImex->setErrInds(errInds);

    ierr = _quadImex->integrate(this);CHKERRQ(ierr);
  }
  else {
    // call odeSolver routine integrate here
    _quadEx->setTolerance(_atol);CHKERRQ(ierr);
    _quadEx->setTimeStepBounds(_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
    ierr = _quadEx->setTimeRange(_initTime,_maxTime);
    ierr = _quadEx->setInitialConds(_var);CHKERRQ(ierr);

    // control which fields are used to select step size
    if (_isMMS) {
      int arrInds[] = {2}; // state: 0, slip: 1
      std::vector<int> errInds(arrInds,arrInds+1); // !! UPDATE THIS LINE TOO
      ierr = _quadEx->setErrInds(errInds);
    }
    else  {
        int arrInds[] = {1}; // state: 0, slip: 1
        std::vector<int> errInds(arrInds,arrInds+1); // !! UPDATE THIS LINE TOO
      ierr = _quadEx->setErrInds(errInds);
    }
    ierr = _quadEx->integrate(this);CHKERRQ(ierr);
  }

  _integrateTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
  CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode SymmMaxwellViscoelastic::d_dt(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin)
{
  PetscErrorCode ierr = 0;
  if (_isMMS) {
    ierr = d_dt_mms(time,varBegin,dvarBegin);CHKERRQ(ierr);
  }
  else {
    ierr = d_dt_eqCycle(time,varBegin,dvarBegin);CHKERRQ(ierr);
  }
  return ierr;
}


// implicit/explicit time stepping
PetscErrorCode SymmMaxwellViscoelastic::d_dt(const PetscScalar time,
  const_it_vec varBegin,it_vec dvarBegin,it_vec varBeginIm,const_it_vec varBeginImo,
  const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmMaxwellViscoelastic::d_dt IMEX in maxwellViscoelastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
#endif

  ierr = d_dt_eqCycle(time,varBegin,dvarBegin);CHKERRQ(ierr);

  Vec stressxzP;
  VecDuplicate(_uP,&stressxzP);
  ierr = _sbpP->muxDz(_uP,stressxzP); CHKERRQ(ierr);
  ierr = _he.be(time,*(dvarBegin+1),_fault._tauQSP,_stressxyP,stressxzP,NULL,
    NULL,*varBeginIm,*varBeginImo,dt);CHKERRQ(ierr);
  VecDestroy(&stressxzP);
  // arguments:
  // time, slipVel, txy, sigmaxy, sigmaxz, dgxy, dgxz, T, dTdt


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmMaxwellViscoelastic::d_dt IMEX in maxwellViscoelastic.cpp: time=%.15e\n",time);CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode SymmMaxwellViscoelastic::d_dt_eqCycle(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin)
{
  PetscErrorCode ierr = 0;
  string funcName = "SymmMaxwellViscoelastic::d_dt_eqCycle";
  string fileName = "maxwellViscoelastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif

  VecCopy(*(varBegin+2),_gxyP);
  VecCopy(*(varBegin+3),_gxzP);

  // update boundaries
  ierr = VecCopy(*(varBegin+1),_bcLP);CHKERRQ(ierr);
  ierr = VecScale(_bcLP,0.5);CHKERRQ(ierr);
  ierr = VecSet(_bcRP,_vL*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_bcRP,1.0,_bcRPShift);CHKERRQ(ierr);

  //~ writeVec(*(_var.begin()+0),"test/state");
  //~ writeVec(*(_var.begin()+1),"test/slip");
  //~ writeVec(_gxyP,"test/gxy");
  //~ writeVec(_gxzP,"test/gxz");
  //~ writeVec(_bcLP,"test/bcL");
  //~ writeVec(_bcRP,"test/bcR");
  //~ PetscPrintf(PETSC_COMM_WORLD,"time = %.9e\n",time);
  //~ assert(0);

  //~ #if LOCK_FAULT == 1
    //~ PetscInt Ii,Istart,Iend;
    //~ VecGetOwnershipRange(_bcLP,&Istart,&Iend);
    //~ for (Ii=Istart;Ii<Iend;Ii++) {
      //~ PetscScalar z = _dz*(Ii-_Nz*(Ii/_Nz));
      //~ PetscScalar v = 3e-4 * (tanh((z-14.8)*10.0) + 1.0) * 0.5;
      //~ VecSetValues(_bcLP,1,&Ii,&v,INSERT_VALUES);
    //~ }
    //~ VecAssemblyBegin(_bcLP);
    //~ VecAssemblyEnd(_bcLP);
  //~ #endif

  // add source terms to rhs: d/dy(mu * gxy) + d/dz(mu * gxz)
  Vec viscSource;
  ierr = VecDuplicate(_gxyP,&viscSource);CHKERRQ(ierr);
  ierr = setViscStrainSourceTerms(viscSource,varBegin);CHKERRQ(ierr);

  // set up rhs vector
  VecSet(_rhsP,0.0);
  ierr = _sbpP->setRhs(_rhsP,_bcLP,_bcRP,_bcTP,_bcBP);CHKERRQ(ierr); // update rhs from BCs
  ierr = VecAXPY(_rhsP,1.0,viscSource);CHKERRQ(ierr);
  VecDestroy(&viscSource);

  // solve for displacement
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspP,_rhsP,_uP);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  //~ writeVec(_uP,"test/u");

  // set shear traction on fault
  ierr = setStresses(time,varBegin);CHKERRQ(ierr);
  ierr = _fault.setTauQS(_stressxyP,NULL);CHKERRQ(ierr);

  //~ writeVec(_stressxyP,"test/sxy");
  //~ writeVec(_stressxzP,"test/sxz");
  //~ assert(0);

  // set rates
  ierr = _fault.d_dt(varBegin,dvarBegin); // sets rates for slip and state
  ierr = setViscStrainRates(time,varBegin,dvarBegin); CHKERRQ(ierr); // sets viscous strain rates

  // lock the fault to test viscous strain alone
  //~ #if LOCK_FAULT == 1
    //~ VecSet(*dvarBegin,0.0); // dstate
    //~ VecSet(*(dvarBegin+1),0.0); // slip vel
  //~ #endif
  //~ VecSet(*(dvarBegin+2),0.0); // dgxy
  //~ VecSet(*(dvarBegin+3),0.0); // dgxz
  //~ VecSet(*(dvarBegin+4),0.0); // dtemp

  #if CALCULATE_ENERGY == 1
    computeEnergyRate(time,varBegin,dvarBegin);
  #endif

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
      CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode SymmMaxwellViscoelastic::d_dt_mms(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin)
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

  ierr = setViscStrainSourceTerms(viscSource,_var.begin());CHKERRQ(ierr);
  if (_Nz == 1) { mapToVec(viscSourceMMS,MMS_gSource1D,*_y,time); }
  else { mapToVec(viscSourceMMS,MMS_gSource,*_y,*_z,time); }
  //~ mapToVec(viscSourceMMS,MMS_gSource,_Nz,_dy,_dz,time);
  ierr = _sbpP->H(viscSourceMMS,HxviscSourceMMS);
  VecDestroy(&viscSourceMMS);
  if (_Nz == 1) { mapToVec(uSource,MMS_uSource1D,*_y,time); }
  else { mapToVec(uSource,MMS_uSource,*_y,*_z,time); }
  //~ mapToVec(uSource,MMS_uSource,_Nz,_dy,_dz,time);
  ierr = _sbpP->H(uSource,HxuSource);
  VecDestroy(&uSource);

  ierr = VecAXPY(_rhsP,1.0,viscSource);CHKERRQ(ierr); // add d/dy mu*epsVxy + d/dz mu*epsVxz
  ierr = VecAXPY(_rhsP,1.0,HxviscSourceMMS);CHKERRQ(ierr); // add MMS source for viscous strains
  ierr = VecAXPY(_rhsP,1.0,HxuSource);CHKERRQ(ierr); // add MMS source for u
  VecDestroy(&viscSource);
  VecDestroy(&HxviscSourceMMS);
  VecDestroy(&HxuSource);


  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspP,_rhsP,_uP);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  //~ if (_Nz == 1) { mapToVec(_uP,MMS_uA1D,*_y,time); }
  //~ else { mapToVec(_uP,MMS_uA,*_y,*_z,time); }

  // update fields on fault
  ierr = setStresses(time,varBegin);CHKERRQ(ierr);

  // update rates
  VecSet(*dvarBegin,0.0); // d/dt psi
  VecSet(*(dvarBegin+1),0.0); // d/dt slip

  // update rates
  ierr = setViscStrainRates(time,varBegin,dvarBegin);CHKERRQ(ierr); // set viscous strain rates
  Vec source;
  VecDuplicate(_uP,&source);
  if (_Nz == 1) { mapToVec(source,MMS_pl_gxy_t_source1D,*_y,_currTime); }
  else { mapToVec(source,MMS_max_gxy_t_source,*_y,*_z,_currTime); }
  VecAXPY(*(dvarBegin+2),1.0,source);
  if (_Nz == 1) { mapToVec(source,MMS_pl_gxz_t_source1D,*_y,_currTime); }
  else { mapToVec(source,MMS_max_gxz_t_source,*_y,*_z,_currTime); }
  VecAXPY(*(dvarBegin+3),1.0,source);
  VecDestroy(&source);

  //~ if (_Nz == 1) { mapToVec(*(dvarBegin+2),MMS_gxy_t1D,*_y,time); }
  //~ else { mapToVec(*(dvarBegin+2),MMS_gxy_t,*_y,*_z,time); }
  //~ if (_Nz == 1) { VecSet(*(dvarBegin+3),0.0);}
  //~ else { mapToVec(*(dvarBegin+3),MMS_gxz_t,*_y,*_z,time); }


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode SymmMaxwellViscoelastic::computeEnergy(const PetscScalar time,Vec& out)
{
  PetscErrorCode ierr = 0;
  string funcName = "SymmMaxwellViscoelastic::computeEnergy";
  string fileName = "maxwellViscoelastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif

  PetscScalar E = 0, alphaDy = 0;//, alphaDz = 0;

  // get relevant matrices
  Mat muqy,murz,H,Ry,Rz,E0y_Iz,ENy_Iz,Iy_E0z,Iy_ENz,By_Iz,Iy_Bz,Hy_Iz,Iy_Hz;
  ierr =  _sbpP->getMus(muqy,murz); CHKERRQ(ierr);
  ierr =  _sbpP->getR(Ry,Rz); CHKERRQ(ierr);
  ierr =  _sbpP->getEs(E0y_Iz,ENy_Iz,Iy_E0z,Iy_ENz); CHKERRQ(ierr);
  ierr =  _sbpP->getBs(By_Iz,Iy_Bz); CHKERRQ(ierr);
  ierr =  _sbpP->getHs(Hy_Iz,Iy_Hz); CHKERRQ(ierr);
  ierr =  _sbpP->getH(H); CHKERRQ(ierr);

  // compute elastic strains
  Vec gExy=NULL,gExz=NULL;
  ierr = VecDuplicate(_uP,&gExy); CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&gExz); CHKERRQ(ierr);
  if (_sbpType.compare("mfc")==0) {
    ierr = _sbpP->Dy(_uP,gExy); CHKERRQ(ierr);
    ierr = VecAXPY(gExy,-1.0,_gxyP); CHKERRQ(ierr);

    if (_order==2) { alphaDy = -4.0/_dy; }
    if (_order==4) { alphaDy = -48.0/17.0 /_dy; }

    // compute energy
    E = multVecMatsVec(gExy,H,muqy,gExy);
    E += multVecMatsVec(_uP,Iy_Hz,Ry,_uP);

    E -= multVecMatsVec(_uP,Iy_Hz,By_Iz,muqy,gExy);
    E -= multVecMatsVec(gExy,Iy_Hz,By_Iz,muqy,_uP);
    E -= alphaDy * multVecMatsVec(_uP,Iy_Hz,muqy,E0y_Iz,_uP);
    E -= alphaDy * multVecMatsVec(_uP,Iy_Hz,muqy,ENy_Iz,_uP);

    if (_Nz > 1) {
      ierr = _sbpP->Dz(_uP,gExz); CHKERRQ(ierr);
      ierr = VecAXPY(gExz,-1.0,_gxzP); CHKERRQ(ierr);
      E += multVecMatsVec(gExz,H,murz,gExz);
      E += multVecMatsVec(_uP,Hy_Iz,Rz,_uP);
    }
  }

  else { // if mfc_coordTrans
    PetscScalar dq = 1.0/(_Ny-1); //, dr = 1.0/(_Nz-1);
    if (_order==2) { alphaDy = -4.0/dq; }
    if (_order==4) { alphaDy = -48.0/17.0 /dq; }

    Mat qy,rz,yq,zr,yqxHy_Iz,Iy_Hzxzr;
    ierr = _sbpP->getCoordTrans(qy,rz,yq,zr); CHKERRQ(ierr);
    Vec temp;
    ierr = VecDuplicate(_uP,&temp); CHKERRQ(ierr);

    // compute elastic strain
    ierr = _sbpP->Dy(_uP,temp); CHKERRQ(ierr);
    ierr = VecAXPY(temp,-1.0,_gxyP); CHKERRQ(ierr);
    ierr = MatMult(yq,temp,gExy);
    VecDestroy(&temp);

    ierr = MatMatMult(zr,Iy_Hz,MAT_INITIAL_MATRIX,1.0,&Iy_Hzxzr);
    ierr = MatMatMult(yq,Iy_Hz,MAT_INITIAL_MATRIX,1.0,&yqxHy_Iz);

    // compute energy
    E = multVecMatsVec(gExy,H,muqy,gExy);
    E += multVecMatsVec(_uP,Iy_Hzxzr,Ry,_uP);

    E -= multVecMatsVec(_uP,Iy_Hzxzr,By_Iz,muqy,gExy);
    E -= multVecMatsVec(gExy,Iy_Hzxzr,By_Iz,muqy,_uP);
    E -= alphaDy * multVecMatsVec(_uP,Iy_Hzxzr,muqy,E0y_Iz,_uP);
    E -= alphaDy * multVecMatsVec(_uP,Iy_Hzxzr,muqy,ENy_Iz,_uP);

    if (_Nz > 1) {
      // compute elastic strain
      Vec temp; VecDuplicate(_gxzP,&temp);
      ierr = _sbpP->Dz(_uP,temp); CHKERRQ(ierr);
      ierr = VecAXPY(temp,-1.0,_gxzP); CHKERRQ(ierr);
      ierr = MatMult(rz,temp,gExz);
      VecDestroy(&temp);

      E += multVecMatsVec(gExz,rz,yq,H,murz,gExz);
      E += multVecMatsVec(_uP,yqxHy_Iz,Rz,_uP);
    }

    MatDestroy(&Iy_Hzxzr);
    MatDestroy(&yqxHy_Iz);
  }

  E = E * 0.5;
  VecSet(out,E);

  VecDestroy(&gExy);
  VecDestroy(&gExz);


  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
      CHKERRQ(ierr);
  #endif
  return ierr = 0;
}

PetscErrorCode SymmMaxwellViscoelastic::computeEnergyRate(const PetscScalar time,const_it_vec varBegin,it_vec dvarBegin)
{
  PetscErrorCode ierr = 0;
  string funcName = "SymmMaxwellViscoelastic::computeEnergy";
  string fileName = "maxwellViscoelastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif

  computeEnergy(time,_E);

  PetscScalar dE = 0, alphaDy = 0; //, alphaDz = 0;

  // get relevant matrices
  Mat muqy,murz,By_Iz,Iy_Bz;
  Mat H,Hy_Iz,Iy_Hz,Hyinv_Iz,Iy_Hzinv;
  Mat e0y_Iz,eNy_Iz,Iy_e0z,Iy_eNz,E0y_Iz,ENy_Iz,Iy_E0z,Iy_ENz;
  ierr =  _sbpP->getMus(muqy,murz); CHKERRQ(ierr);
  ierr =  _sbpP->getes(e0y_Iz,eNy_Iz,Iy_e0z,Iy_eNz); CHKERRQ(ierr);
  ierr =  _sbpP->getEs(E0y_Iz,ENy_Iz,Iy_E0z,Iy_ENz); CHKERRQ(ierr);
  ierr =  _sbpP->getBs(By_Iz,Iy_Bz); CHKERRQ(ierr);
  ierr =  _sbpP->getHs(Hy_Iz,Iy_Hz); CHKERRQ(ierr);
  ierr =  _sbpP->getHinvs(Hyinv_Iz,Iy_Hzinv); CHKERRQ(ierr);
  ierr =  _sbpP->getH(H); CHKERRQ(ierr);

  Vec ut;
  VecDuplicate(_uP,&ut);
  VecSet(ut,0.0);
  if (abs(time - _currTime) > 1e-14) {
    VecWAXPY(ut,-1.0,_uPPrev,_uP);
    VecScale(ut,1.0/(time - _currTime));
  }

  Vec ones;
  VecDuplicate(_visc,&ones);
  VecSet(ones,1.0);
  Vec invViscV;
  ierr = VecDuplicate(_visc,&invViscV); CHKERRQ(ierr);
  ierr = VecPointwiseDivide(invViscV,ones,_visc); CHKERRQ(ierr);
  Mat invVisc;
  MatDuplicate(muqy,MAT_DO_NOT_COPY_VALUES,&invVisc);
  ierr = MatDiagonalSet(invVisc,invViscV,INSERT_VALUES); CHKERRQ(ierr);
  VecDestroy(&invViscV);
  VecDestroy(&ones);


  Vec ut_y;
  VecDuplicate(ut,&ut_y);
  VecSet(ut_y,0.0);

  Vec gExy=NULL,gExz=NULL;
  ierr = VecDuplicate(_uP,&gExy); CHKERRQ(ierr);
  ierr = VecDuplicate(_uP,&gExz); CHKERRQ(ierr);

  if (_sbpType.compare("mfc")==0) {
    ierr = _sbpP->Dy(ut,ut_y); CHKERRQ(ierr);

    ierr = _sbpP->Dy(_uP,gExy); CHKERRQ(ierr);
    ierr = VecAXPY(gExy,-1.0,_gxyP); CHKERRQ(ierr);

    Mat coeff;
    ierr = MatMatMatMult(muqy,muqy,invVisc,MAT_INITIAL_MATRIX,1.0,&coeff);

    if (_order==2) { alphaDy = -4.0/_dy; }
    if (_order==4) { alphaDy = -48.0/17.0 /_dy; }

    // energy rate
    dE -= multVecMatsVec(gExy,H,coeff,gExy);

    dE -= 2.0 * multVecMatsVec(_uP,Iy_Hz,coeff,E0y_Iz,gExy);
    dE += 2.0 * multVecMatsVec(_uP,Iy_Hz,coeff,ENy_Iz,gExy);
    dE -= multVecMatsVec(_uP,Iy_Hz,coeff,Hyinv_Iz,E0y_Iz,_uP);
    dE -= multVecMatsVec(_uP,Iy_Hz,coeff,Hyinv_Iz,ENy_Iz,_uP);

    dE -= alphaDy * multVecMatsVec(ut,Iy_Hz,muqy,e0y_Iz,_bcLP);
    dE -= alphaDy * multVecMatsVec(ut,Iy_Hz,muqy,eNy_Iz,_bcRP);
    dE += multVecMatsVec(ut_y,Iy_Hz,muqy,e0y_Iz,_bcLP);
    dE -= multVecMatsVec(ut_y,Iy_Hz,muqy,eNy_Iz,_bcRP);

    dE += multVecMatsVec(gExy,coeff,Iy_Hz,e0y_Iz,_bcLP);
    dE -= multVecMatsVec(gExy,coeff,Iy_Hz,eNy_Iz,_bcRP);
    dE += multVecMatsVec(_uP,Hyinv_Iz,Iy_Hz,coeff,e0y_Iz,_bcLP);
    dE += multVecMatsVec(_uP,Hyinv_Iz,Iy_Hz,coeff,eNy_Iz,_bcRP);

    if (_Nz > 1) {
      ierr = _sbpP->Dz(_uP,gExz); CHKERRQ(ierr);
      ierr = VecAXPY(gExz,-1.0,_gxzP); CHKERRQ(ierr);

      dE -= multVecMatsVec(gExz,H,coeff,gExz);
      dE -= multVecMatsVec(ut,Hy_Iz,Iy_e0z,_bcTP);
      dE += multVecMatsVec(ut,Hy_Iz,Iy_eNz,_bcBP);
    }

    MatDestroy(&coeff);
  }

  else { // if mfc_coordTrans
    PetscScalar dq = 1.0/(_Ny-1); //, dr = 1.0/(_Nz-1);
    if (_order==2) { alphaDy = -4.0/dq; }
    if (_order==4) { alphaDy = -48.0/17.0 /dq; }

    Mat qy,rz,yq,zr,Iy_Hzxzr,yqxHy_Iz;
    ierr = _sbpP->getCoordTrans(qy,rz,yq,zr); CHKERRQ(ierr);
    Vec temp;
    ierr = VecDuplicate(ut,&temp); CHKERRQ(ierr);
    ierr = _sbpP->Dy(ut,temp); CHKERRQ(ierr);
    ierr = MatMult(yq,temp,ut_y);

    // compute elastic strain
    ierr = _sbpP->Dy(_uP,temp); CHKERRQ(ierr);
    ierr = VecAXPY(temp,-1.0,_gxyP); CHKERRQ(ierr);
    ierr = MatMult(yq,temp,gExy);
    VecDestroy(&temp);

    ierr = MatMatMult(zr,Iy_Hz,MAT_INITIAL_MATRIX,1.0,&Iy_Hzxzr);
    ierr = MatMatMult(yq,Iy_Hz,MAT_INITIAL_MATRIX,1.0,&yqxHy_Iz);


    Mat coeff,yqcoeff;
    ierr = MatMatMatMult(muqy,muqy,invVisc,MAT_INITIAL_MATRIX,1.0,&coeff);
    ierr = MatMatMult(yq,coeff,MAT_INITIAL_MATRIX,1.0,&yqcoeff);

    // energy rate
    dE -= multVecMatsVec(gExy,H,zr,yq,coeff,gExy);

    dE -= 2.0 * multVecMatsVec(_uP,Iy_Hzxzr,yqcoeff,E0y_Iz,gExy);
    dE += 2.0 * multVecMatsVec(_uP,Iy_Hzxzr,yqcoeff,ENy_Iz,gExy);
    dE -= multVecMatsVec(_uP,Iy_Hzxzr,yqcoeff,Hyinv_Iz,E0y_Iz,_uP);
    dE -= multVecMatsVec(_uP,Iy_Hzxzr,yqcoeff,Hyinv_Iz,ENy_Iz,_uP);

    dE -= alphaDy * multVecMatsVec(ut,Iy_Hzxzr,muqy,e0y_Iz,_bcLP);
    dE -= alphaDy * multVecMatsVec(ut,Iy_Hzxzr,muqy,eNy_Iz,_bcRP);
    dE += multVecMatsVec(ut_y,Iy_Hzxzr,muqy,e0y_Iz,_bcLP);
    dE -= multVecMatsVec(ut_y,Iy_Hzxzr,muqy,eNy_Iz,_bcRP);

    dE += multVecMatsVec(gExy,yqcoeff,Iy_Hzxzr,e0y_Iz,_bcLP);
    dE -= multVecMatsVec(gExy,yqcoeff,Iy_Hzxzr,eNy_Iz,_bcRP);
    dE += multVecMatsVec(_uP,Hyinv_Iz,Iy_Hzxzr,yqcoeff,e0y_Iz,_bcLP);
    dE += multVecMatsVec(_uP,Hyinv_Iz,Iy_Hzxzr,yqcoeff,eNy_Iz,_bcRP);

    MatDestroy(&coeff);
    MatDestroy(&yqcoeff);

    if (_Nz > 1) {
      Mat coeff;
      ierr = MatMatMatMult(murz,murz,invVisc,MAT_INITIAL_MATRIX,1.0,&coeff);

      // compute elastic strain
      Vec temp; VecDuplicate(_gxzP,&temp);
      ierr = _sbpP->Dz(_uP,temp); CHKERRQ(ierr);
      ierr = VecAXPY(temp,-1.0,_gxzP); CHKERRQ(ierr);
      ierr = MatMult(rz,temp,gExz);
      VecDestroy(&temp);

      dE -= multVecMatsVec(gExz,H,rz,yq,coeff,gExz);
      dE -= multVecMatsVec(ut,yqxHy_Iz,Iy_e0z,_bcTP);
      dE += multVecMatsVec(ut,yqxHy_Iz,Iy_eNz,_bcBP);
    }
    MatDestroy(&Iy_Hzxzr);
    MatDestroy(&yqxHy_Iz);
    MatDestroy(&coeff);
  }

  assert(!isnan(dE));

  VecSet(*(dvarBegin+4),dE);

  VecDestroy(&ut);
  VecDestroy(&ut_y);
  VecDestroy(&gExy);
  VecDestroy(&gExz);
  MatDestroy(&invVisc);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
      CHKERRQ(ierr);
  #endif
  return ierr = 0;
}


PetscErrorCode SymmMaxwellViscoelastic::setViscStrainSourceTerms(Vec& out,const_it_vec varBegin)
{
  PetscErrorCode ierr = 0;
  string funcName = "SymmMaxwellViscoelastic::setViscStrainSourceTerms";
  string fileName = "maxwellViscoelastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());
    CHKERRQ(ierr);
  #endif

  Vec source;
  VecDuplicate(_gxyP,&source);
  VecSet(source,0.0);

  // add source terms to rhs: d/dy( mu*gxy) + d/dz( mu*gxz)
  // + Hz^-1 E0z mu gxz + Hz^-1 ENz mu gxz
  Vec sourcexy_y;
  VecDuplicate(_uP,&sourcexy_y);
  VecSet(sourcexy_y,0.0);
  ierr = _sbpP->Dyxmu(_gxyP,sourcexy_y);CHKERRQ(ierr);
  if (_sbpType.compare("mfc_coordTrans")==0) {
    Mat qy,rz,yq,zr;
    Vec temp1,temp2;
    VecDuplicate(_gxyP,&temp1);
    VecDuplicate(_gxyP,&temp2);
    ierr = _sbpP->getCoordTrans(qy,rz,yq,zr); CHKERRQ(ierr);
    MatMult(yq,sourcexy_y,temp1);
    MatMult(zr,temp1,temp2);
    VecCopy(temp2,sourcexy_y);
    VecDestroy(&temp1);
    VecDestroy(&temp2);
  }
  ierr = VecCopy(sourcexy_y,source);CHKERRQ(ierr); // sourcexy_y -> source
  VecDestroy(&sourcexy_y);

  if (_Nz > 1)
  {
    Vec sourcexz_z;
    VecDuplicate(_gxzP,&sourcexz_z);
    ierr = _sbpP->Dzxmu(_gxzP,sourcexz_z);CHKERRQ(ierr);
    if (_sbpType.compare("mfc_coordTrans")==0) {
      Mat qy,rz,yq,zr;
      Vec temp1,temp2;
      VecDuplicate(_gxzP,&temp1);
      VecDuplicate(_gxzP,&temp2);
      ierr = _sbpP->getCoordTrans(qy,rz,yq,zr); CHKERRQ(ierr);
      MatMult(zr,sourcexz_z,temp1);
      MatMult(yq,temp1,temp2);
      VecCopy(temp2,sourcexz_z);
      VecDestroy(&temp1);
      VecDestroy(&temp2);
    }
    ierr = VecAXPY(source,1.0,sourcexz_z);CHKERRQ(ierr); // source += Hxsourcexz_z
    VecDestroy(&sourcexz_z);

    // enforce traction boundary condition
    Vec temp1,bcT,bcB;
    VecDuplicate(_gxzP,&temp1); VecSet(temp1,0.0);
    VecDuplicate(_gxzP,&bcT);
    VecDuplicate(_gxzP,&bcB);

    _sbpP->HzinvxE0z(_gxzP,temp1);
    ierr = VecPointwiseMult(bcT,_muVecP,temp1); CHKERRQ(ierr);

    _sbpP->HzinvxENz(_gxzP,temp1);
    ierr = VecPointwiseMult(bcB,_muVecP,temp1); CHKERRQ(ierr);

    if (_sbpType.compare("mfc_coordTrans")==0) {
      Mat qy,rz,yq,zr;
      Vec temp2;
      VecDuplicate(_gxzP,&temp2);
      ierr = _sbpP->getCoordTrans(qy,rz,yq,zr); CHKERRQ(ierr);
      MatMult(yq,bcB,temp2);
      VecCopy(temp2,bcB);
      MatMult(yq,bcT,temp2);
      VecCopy(temp2,bcT);
      VecDestroy(&temp2);
    }

    ierr = VecAXPY(source,1.0,bcT);CHKERRQ(ierr);
    ierr = VecAXPY(source,-1.0,bcB);CHKERRQ(ierr);

    VecDestroy(&temp1);
    VecDestroy(&bcT);
    VecDestroy(&bcB);
  }


  ierr = _sbpP->H(source,out); CHKERRQ(ierr);
  VecDestroy(&source);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %se\n",funcName.c_str(),fileName.c_str());
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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
      CHKERRQ(ierr);
  #endif
  return ierr = 0;
}



PetscErrorCode SymmMaxwellViscoelastic::setViscStrainRates(const PetscScalar time,const_it_vec varBegin,
                 it_vec dvarBegin)
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

  // d/dt gxy = sxy/visc + qy*mu/visc*SAT
  VecPointwiseMult(*(dvarBegin+2),_muVecP,SAT);
  if (_sbpType.compare("mfc_coordTrans")==0) {
    Mat qy,rz,yq,zr;
    Vec temp1;
    VecDuplicate(_gxyP,&temp1);
    ierr = _sbpP->getCoordTrans(qy,rz,yq,zr); CHKERRQ(ierr);
    MatMult(qy,*(dvarBegin+2),temp1);
    VecCopy(temp1,*(dvarBegin+2));
    VecDestroy(&temp1);
  }
  VecSet(*(dvarBegin+2),0.0);
  VecAXPY(*(dvarBegin+2),1.0,_stressxyP);
  VecPointwiseDivide(*(dvarBegin+2),*(dvarBegin+2),_visc);

  if (_Nz > 1) {
    VecCopy(_stressxzP,*(dvarBegin+3));
    VecPointwiseDivide(*(dvarBegin+3),*(dvarBegin+3),_visc);
  }

  VecDestroy(&SAT);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
      CHKERRQ(ierr);
  #endif
  return ierr = 0;
}


PetscErrorCode SymmMaxwellViscoelastic::setStresses(const PetscScalar time,const_it_vec varBegin)
{
    PetscErrorCode ierr = 0;
    string funcName = "SymmMaxwellViscoelastic::setStresses";
    string fileName = "maxwellViscoelastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif

  // compute strains
  _sbpP->Dy(_uP,_gTxyP);
  VecCopy(_gTxyP,_stressxyP);
  VecAXPY(_stressxyP,-1.0,_gxyP);
  VecPointwiseMult(_stressxyP,_stressxyP,_muVecP);

  if (_Nz > 1) {
    _sbpP->Dz(_uP,_gTxzP);
    VecCopy(_gTxzP,_stressxzP);
    VecAXPY(_stressxzP,-1.0,_gxzP);
    VecPointwiseMult(_stressxzP,_stressxzP,_muVecP);
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
  //~ mapToVec2D(_visc,MMS_visc,_Nz,_dy,_dz);
  //~mapToVec(_uP,MMS_uA,_Nz,_dy,_dz,time);
  //~ mapToVec(_gxyP,MMS_gxy,_Nz,_dy,_dz,time);
  //~ mapToVec(_gxzP,MMS_gxz,_Nz,_dy,_dz,time);

  if (_Nz == 1) { mapToVec(_visc,MMS_visc1D,*_y); }
  else { mapToVec(_visc,MMS_visc,*_y,*_z); }
  if (_Nz == 1) { mapToVec(_gxyP,MMS_gxy1D,*_y,time); }
  else { mapToVec(_gxyP,MMS_gxy,*_y,*_z,time); }
  if (_Nz == 1) { VecSet(_gxzP,0.0); }
  else { mapToVec(_gxzP,MMS_gxz,*_y,*_z,time); }

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

  ierr = setViscStrainSourceTerms(viscSource,_var.begin());CHKERRQ(ierr);
  if (_Nz == 1) { mapToVec(viscSourceMMS,MMS_gSource1D,*_y,time); }
  else { mapToVec(viscSourceMMS,MMS_gSource,*_y,*_z,time); }
  //~ mapToVec(viscSourceMMS,MMS_gSource,_Nz,_dy,_dz,time);
  ierr = _sbpP->H(viscSourceMMS,HxviscSourceMMS);
  VecDestroy(&viscSourceMMS);
  if (_Nz == 1) { mapToVec(uSource,MMS_uSource1D,*_y,time); }
  else { mapToVec(uSource,MMS_uSource,*_y,*_z,time); }
  //~ mapToVec(uSource,MMS_uSource,_Nz,_dy,_dz,time);
  ierr = _sbpP->H(uSource,HxuSource);
  VecDestroy(&uSource);

  ierr = VecAXPY(_rhsP,1.0,viscSource);CHKERRQ(ierr); // add d/dy mu*epsVxy + d/dz mu*epsVxz
  ierr = VecAXPY(_rhsP,1.0,HxviscSourceMMS);CHKERRQ(ierr); // add MMS source for viscous strains
  ierr = VecAXPY(_rhsP,1.0,HxuSource);CHKERRQ(ierr); // add MMS source for u
  VecDestroy(&HxviscSourceMMS);
  VecDestroy(&HxuSource);
  VecDestroy(&viscSource);


  // solve for displacement
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspP,_rhsP,_uP);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  // set stresses
  ierr = setStresses(time,_var.begin());CHKERRQ(ierr);

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
  if (_Nz == 1) {
    Ii = Istart;
    y = 0;
    if (!_bcLType.compare("Dirichlet")) { v = MMS_uA1D(y,time); } // uAnal(y=0,z)
    else if (!_bcLType.compare("Neumann")) { v = MMS_mu1D(y) * (MMS_uA_y1D(y,time)); } // sigma_xy = mu * d/dy u
    ierr = VecSetValues(_bcLP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

    y = _Ly;
    if (!_bcRType.compare("Dirichlet")) { v = MMS_uA1D(y,time); } // uAnal(y=Ly,z)
    else if (!_bcRType.compare("Neumann")) { v = MMS_mu1D(y) * (MMS_uA_y1D(y,time)); } // sigma_xy = mu * d/dy u
    ierr = VecSetValues(_bcRP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  else {
    for(Ii=Istart;Ii<Iend;Ii++) {
      z = _dz * Ii;

      y = 0;
      if (!_bcLType.compare("Dirichlet")) { v = MMS_uA(y,z,time); } // uAnal(y=0,z)
      else if (!_bcLType.compare("Neumann")) { v = MMS_mu(y,z) * (MMS_uA_y(y,z,time)); } // sigma_xy = mu * d/dy u
      ierr = VecSetValues(_bcLP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

      y = _Ly;
      if (!_bcRType.compare("Dirichlet")) { v = MMS_uA(y,z,time); } // uAnal(y=Ly,z)
      else if (!_bcRType.compare("Neumann")) { v = MMS_mu(y,z) * (MMS_uA_y(y,z,time)); } // sigma_xy = mu * d/dy u
      ierr = VecSetValues(_bcRP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_bcLP);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_bcRP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcLP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcRP);CHKERRQ(ierr);

  // set up boundary conditions: T and B
  ierr = VecGetOwnershipRange(_bcTP,&Istart,&Iend);CHKERRQ(ierr);
  for(Ii=Istart;Ii<Iend;Ii++) {
    y = _dy * Ii;

    z = 0;
    if (!_bcTType.compare("Dirichlet")) { v = MMS_uA(y,z,time); } // uAnal(y,z=0)
    else if (!_bcTType.compare("Neumann")) { v = MMS_mu(y,z) * (MMS_uA_z(y,z,time) - MMS_gxz(y,z,time)); }
    //~ else if (!_bcTType.compare("Neumann")) { v = MMS_mu(y,z) * (MMS_uA_z(y,z,time)); }
    ierr = VecSetValues(_bcTP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

    z = _Lz;
    if (!_bcBType.compare("Dirichlet")) { v = MMS_uA(y,z,time); } // uAnal(y,z=Lz)
    else if (!_bcBType.compare("Neumann")) { v = MMS_mu(y,z) * (MMS_uA_z(y,z,time)- MMS_gxz(y,z,time)); }
    //~ else if (!_bcBType.compare("Neumann")) { v = MMS_mu(y,z) * (MMS_uA_z(y,z,time)); }
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
  //~ mapToVec(uA,MMS_uA,_Nz,_dy,_dz,_currTime);
  //~ mapToVec(gxyA,MMS_gxy,_Nz,_dy,_dz,_currTime);
  //~ mapToVec(gxzA,MMS_gxz,_Nz,_dy,_dz,_currTime);

  if (_Nz == 1) { mapToVec(uA,MMS_uA1D,*_y,_currTime); }
  else { mapToVec(uA,MMS_uA,*_y,*_z,_currTime); }
    if (_Nz == 1) { mapToVec(gxyA,MMS_gxy1D,*_y,_currTime); }
  else { mapToVec(gxyA,MMS_gxy,*_y,*_z,_currTime); }
  if (_Nz == 1) { VecSet(gxzA,0.0); }
  else { mapToVec(gxzA,MMS_gxz,*_y,*_z,_currTime); }


  double err2u = computeNormDiff_2(_uP,uA);
  double err2epsxy = computeNormDiff_2(*(_var.begin()+2),gxyA);
  double err2epsxz = computeNormDiff_2(_gxzP,gxzA);

  PetscPrintf(PETSC_COMM_WORLD,"%i  %3i %.4e %.4e % .15e %.4e % .15e %.4e % .15e\n",
              _order,_Ny,_dy,err2u,log2(err2u),err2epsxy,log2(err2epsxy),err2epsxz,log2(err2epsxz));

  VecDestroy(&uA);
  VecDestroy(&gxyA);
  VecDestroy(&gxzA);
  return ierr;
}


PetscErrorCode SymmMaxwellViscoelastic::timeMonitor(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec dvarBegin)
{
  PetscErrorCode ierr = 0;
  _stepCount = stepCount;
  _currTime = time;
  #if CALCULATE_ENERGY == 1
    VecCopy(_uP,_uPPrev);
  #endif
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
                     const_it_vec varBegin,const_it_vec dvarBegin,const char *stage)
{
  PetscErrorCode ierr = 0;

#if ODEPRINT > 0
  PetscInt       Istart,Iend;
  PetscScalar    bcRval,uVal,psiVal,velVal,dQVal,tauQS;
  PetscScalar    epsVxy,depsVxy;

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
    ierr = _sbpP->writeOps(_outputDir);CHKERRQ(ierr);
    _he.writeContext(_outputDir);
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

    // write out boundary conditions for testing purposes
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcR").c_str(),
              FILE_MODE_WRITE,&_bcRPlusV);CHKERRQ(ierr);
    ierr = VecView(_bcRP,_bcRPlusV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_bcRPlusV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcR").c_str(),
                                   FILE_MODE_APPEND,&_bcRPlusV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcL").c_str(),
              FILE_MODE_WRITE,&_bcLPlusV);CHKERRQ(ierr);
    ierr = VecView(_bcLP,_bcLPlusV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_bcLPlusV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcL").c_str(),
                                   FILE_MODE_APPEND,&_bcLPlusV);CHKERRQ(ierr);

    ierr = _fault.writeStep(_outputDir,_stepCount);CHKERRQ(ierr);

    #if CALCULATE_ENERGY == 1
      // write out calculated energy
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"E").c_str(),FILE_MODE_WRITE,
        &_eV);CHKERRQ(ierr);
      ierr = VecView(_E,_eV);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_eV);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"E").c_str(),
        FILE_MODE_APPEND,&_eV);CHKERRQ(ierr);

      // write out integrated energy
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"intE").c_str(),FILE_MODE_WRITE,
        &_intEV);CHKERRQ(ierr);
      ierr = VecView(*(_var.end()-1),_intEV);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_intEV);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"intE").c_str(),
        FILE_MODE_APPEND,&_intEV);CHKERRQ(ierr);
    #endif
  }
  else {
    ierr = PetscViewerASCIIPrintf(_timeV1D, "%.15e\n",_currTime);CHKERRQ(ierr);
    ierr = _fault.writeStep(_outputDir,_stepCount);CHKERRQ(ierr);

    ierr = VecView(_surfDispPlus,_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_bcRP,_bcRPlusV);CHKERRQ(ierr);
    ierr = VecView(_bcLP,_bcLPlusV);CHKERRQ(ierr);

    #if CALCULATE_ENERGY == 1
      ierr = VecView(_E,_eV);CHKERRQ(ierr);
      ierr = VecView(*(_var.end()-1),_intEV);CHKERRQ(ierr);
    #endif
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
    _he.writeStep2D(_stepCount);

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(_outputDir+"time2D.txt").c_str(),&_timeV2D);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_timeV2D, "%.15e\n",_currTime);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"u").c_str(),
              FILE_MODE_WRITE,&_uPV);CHKERRQ(ierr);
    ierr = VecView(_uP,_uPV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_uPV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"u").c_str(),
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
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gTxzP").c_str(),
              FILE_MODE_WRITE,&_gTxzPV);CHKERRQ(ierr);
      ierr = VecView(_gTxzP,_gTxzPV);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_gTxzPV);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"gTxzP").c_str(),
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
    _he.writeStep2D(_stepCount);

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
  //~ ierr = _quadEx->view();
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
    else if (var.compare("strainRate")==0) {
      _strainRate = atof( (line.substr(pos+_delim.length(),line.npos)).c_str() );
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
    Vec         _A,_n,_B;
    ierr = VecDuplicate(_uP,&_A);CHKERRQ(ierr);
    ierr = VecDuplicate(_uP,&_B);CHKERRQ(ierr);
    ierr = VecDuplicate(_uP,&_n);CHKERRQ(ierr);
    //~ Vec sigmadev;
    //~ ierr = VecDuplicate(_uP,&sigmadev);CHKERRQ(ierr);
    if (_Nz == 1) {
      VecSet(_A,_AVals[0]);
      VecSet(_B,_BVals[0]);
      VecSet(_n,_nVals[0]);
      //~ VecSet(sigmadev,_sigmadevVals[0]);
    }
    else {
      ierr = setVecFromVectors(_A,_AVals,_ADepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_B,_BVals,_BDepths);CHKERRQ(ierr);
      ierr = setVecFromVectors(_n,_nVals,_nDepths);CHKERRQ(ierr);
      //~ ierr = setVecFromVectors(sigmadev,_sigmadevVals,_sigmadevDepths);CHKERRQ(ierr);
    }

    // compute effective viscosity using heat equation's computed temperature
    PetscScalar s,A,B,n,T,effVisc,invVisc=0;
    PetscInt Ii,Istart,Iend;
    VecGetOwnershipRange(_A,&Istart,&Iend);
    for (Ii=Istart;Ii<Iend;Ii++) {
      VecGetValues(_A,1,&Ii,&A);
      VecGetValues(_B,1,&Ii,&B);
      VecGetValues(_n,1,&Ii,&n);
      VecGetValues(_he._T,1,&Ii,&T);
      s = pow(_strainRate/(A*exp(-B/T)),1.0/n);
      effVisc =  s/_strainRate* 1e-3; // (GPa s)  in terms of strain rate
      invVisc = 1.0/effVisc;

      PetscScalar z;
      VecGetValues(*_z,1,&Ii,&z); // !!
      //~ if (z <= 15) { effVisc = 7.693e11; }
      if (z <= 25) { effVisc = 1e14; }

      VecSetValues(_visc,1,&Ii,&effVisc,INSERT_VALUES);
      assert(!isnan(invVisc));
      assert(!isnan(effVisc));
    }
    VecAssemblyBegin(_visc);
    VecAssemblyEnd(_visc);
    //~ VecDestroy(&sigmadev);
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
  //~ ierr = VecCreate(PETSC_COMM_WORLD,&_visc);CHKERRQ(ierr);
  //~ ierr = VecSetSizes(_visc,PETSC_DECIDE,_Ny*_Nz);CHKERRQ(ierr);
  //~ ierr = VecSetFromOptions(_visc);
  //~ PetscObjectSetName((PetscObject) _visc, "_visc");
  //~ ierr = loadVecFromInputFile(_visc,_inputDir, "visc");CHKERRQ(ierr);

  PetscViewer inv; // in viewer

  // load bcL
  //~ string vecSourceFile = _inputDir + "bcL";
  //~ ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  //~ ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  //~ ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  //~ ierr = VecLoad(_bcLP,inv);CHKERRQ(ierr);

  //~ // load bcR
  //~ vecSourceFile = _inputDir + "bcR";
  //~ ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  //~ ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  //~ ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  //~ ierr = VecLoad(_bcRPShift,inv);CHKERRQ(ierr);


  // load gxy
  string vecSourceFile = _inputDir + "Gxy";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_gxyP,inv);CHKERRQ(ierr);
  //~ ierr = PetscViewerDestroy(&inv);

  // load gxz
  vecSourceFile = _inputDir + "Gxz";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&inv);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecSourceFile.c_str(),FILE_MODE_READ,&inv);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(inv,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = VecLoad(_gxzP,inv);CHKERRQ(ierr);
  //~ ierr = PetscViewerDestroy(&inv);

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
    assert(_AVals.size() > 0);
    assert(_BVals.size() > 0);
    assert(_nVals.size() > 0);
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
    std::string FILENAME = "SymmMaxwellViscoelastic";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME.c_str());
    CHKERRQ(ierr);
  #endif

  // build structure from generalized input
  size_t vecLen = depths.size();
  ierr = VecGetOwnershipRange(vec,&Istart,&Iend);CHKERRQ(ierr);
  for (PetscInt Ii=Istart;Ii<Iend;Ii++)
  {
    //~ z = _dz*(Ii-_Nz*(Ii/_Nz));
    VecGetValues(*_z,1,&Ii,&z);CHKERRQ(ierr);
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
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME.c_str());
    CHKERRQ(ierr);
  #endif
  return ierr;
}
