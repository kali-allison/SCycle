#include "iceSheet.hpp"


IceSheet::IceSheet(Domain& D)
: SymmMaxwellViscoelastic(D),_inputDir(D._inputDir),
  _epsVxyP(NULL),_depsVxyP(NULL),
  _epsVxzP(NULL),_depsVxzP(NULL),
  _epsVxyPV(NULL),_depsVxyPV(NULL),
  _epsVxzPV(NULL),_depsVxzPV(NULL),
  _epsTotxyP(NULL),_epsTotxzP(NULL),
  _epsTotxyPV(NULL),_epsTotxzPV(NULL),
  _stressxzP(NULL),_stressxyPV(NULL),_stressxzPV(NULL)
{
  #if VERBOSE > 1
    string funcName = "IceSheet::IceSheet";
    string fileName = "iceSheet.cpp";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());
  #endif

  // load material properties A,n,temp for power law
  VecDuplicate(_uP,&_A);
  //~loadVecFromInputFile(_A,_inputDir,"powerLaw_A");
  VecSet(_A,11);

  VecDuplicate(_uP,&_temp);
  //~loadVecFromInputFile(_temp,_inputDir,"powerLaw_temp");
  VecSet(_temp,500);

  VecDuplicate(_uP,&_n);
  //~loadVecFromInputFile(_n,_inputDir,"powerLaw_n");
  VecSet(_n,3.0);


  VecDuplicate(_uP,&_epsVxyP);
  PetscObjectSetName((PetscObject) _epsVxyP, "_epsVxyP");
  VecSet(_epsVxyP,0.0);
  VecDuplicate(_uP,&_depsVxyP);
  PetscObjectSetName((PetscObject) _depsVxyP, "_depsVxyP");
  VecSet(_depsVxyP,0.0);

  VecDuplicate(_uP,&_epsVxzP);
  PetscObjectSetName((PetscObject) _epsVxzP, "_epsVxzP");
  VecSet(_epsVxzP,0.0);
  VecDuplicate(_uP,&_depsVxzP);
  PetscObjectSetName((PetscObject) _depsVxzP, "_depsVxzP");
  VecSet(_depsVxzP,0.0);


  VecDuplicate(_uP,&_epsTotxyP); VecSet(_epsTotxyP,0.0);
  VecDuplicate(_uP,&_epsTotxzP); VecSet(_epsTotxzP,0.0);
  VecDuplicate(_uP,&_stressxzP); VecSet(_stressxzP,0.0);


  // add viscous strain to integrated variables, stored in _fault._var
  _fault._var.push_back(_epsVxyP);
  _fault._var.push_back(_epsVxzP);

  if (_isMMS) {
    setMMSInitialConditions();
  }


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
  #endif
}

IceSheet::~IceSheet()
{
  #if VERBOSE > 1
    string funcName = "IceSheet::~IceSheet";
    string fileName = "iceSheet.cpp";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());
  #endif

  // from maxwellViscoelastic
  VecDestroy(&_epsTotxyP);
  VecDestroy(&_epsTotxzP);
  VecDestroy(&_epsVxyP);
  VecDestroy(&_epsVxzP);
  VecDestroy(&_depsVxyP);
  VecDestroy(&_depsVxzP);
  VecDestroy(&_stressxzP);
  PetscViewerDestroy(&_epsTotxyPV);
  PetscViewerDestroy(&_epsTotxzPV);
  PetscViewerDestroy(&_epsVxyPV);
  PetscViewerDestroy(&_epsVxzPV);
  PetscViewerDestroy(&_depsVxyPV);
  PetscViewerDestroy(&_depsVxzPV);
  PetscViewerDestroy(&_stressxzPV);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),fileName.c_str());
  #endif
}


PetscErrorCode IceSheet::d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
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


PetscErrorCode IceSheet::d_dt_eqCycle(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                 it_vec dvarBegin,it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;
  string funcName = "IceSheet::d_dt_eqCycle";
  string fileName = "iceSheet.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif

  // update boundaries
  ierr = VecCopy(*(varBegin+1),_bcLP);CHKERRQ(ierr);
  ierr = VecScale(_bcLP,0.5);CHKERRQ(ierr); // var holds slip velocity, bcL is displacement at y=0+
  ierr = VecSet(_bcRP,_vL*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_bcRP,1.0,_bcRPShift);CHKERRQ(ierr);

  // add source terms to rhs: d/dy( 2*mu*strainV_xy) + d/dz( 2*mu*strainV_xz)
  Vec viscSource;
  ierr = setViscStrainSourceTerms(viscSource);CHKERRQ(ierr);

  // set up rhs vector
  ierr = _sbpP.setRhs(_rhsP,_bcLP,_bcRP,_bcTP,_bcBP);CHKERRQ(ierr); // update rhs from BCs
  ierr = VecAXPY(_rhsP,1.0,viscSource);CHKERRQ(ierr);

  // solve for displacement
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspP,_rhsP,_uP);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  // set shear traction on fault
  ierr = MatMult(_sbpP._muxDy_Iz,_uP,_stressxyP);CHKERRQ(ierr);
  ierr = _fault.setTauQS(_stressxyP,NULL);CHKERRQ(ierr);

  // set rates for slip and state
  ierr = _fault.d_dt(varBegin,varEnd, dvarBegin, dvarEnd);

  // compute viscous strains and strain rates
  ierr = setViscStrainsAndRates(time,varBegin,varEnd,dvarBegin,dvarEnd);CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
      CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode IceSheet::d_dt_mms(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                 it_vec dvarBegin,it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;
  string funcName = "IceSheet::d_dt_mms";
  string fileName = "iceSheet.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif

  PetscInt Ii,Istart,Iend; // d/dt u
  PetscScalar y,z,v;

  MMS_uA(_uAnal,time);

  // set viscous source terms: d/dy( 2*mu*strainV_xy) + d/dz( 2*mu*strainV_xz)
  Vec viscSource;
  ierr = setViscStrainSourceTerms(viscSource);CHKERRQ(ierr);
  //~ierr = setMMSuSourceTerms(viscSource,time);CHKERRQ(ierr);

  // set up rhs vector
  setMMSBoundaryConditions(time);
  ierr = _sbpP.setRhs(_rhsP,_bcLP,_bcRP,_bcTP,_bcBP);CHKERRQ(ierr); // update rhs from BCs

  // add source terms
  ierr = VecAXPY(_rhsP,1.0,viscSource);CHKERRQ(ierr); // rhs = rhs + source


  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspP,_rhsP,_uP);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  // update fields on fault
  ierr = MatMult(_sbpP._muxDy_Iz,_uP,_stressxyP);CHKERRQ(ierr);
  ierr = _fault.setTauQS(_stressxyP,NULL);CHKERRQ(ierr);

  // update rates
  VecSet(*dvarBegin,0.0); // d/dt psi

  ierr = VecGetOwnershipRange(*(dvarBegin+1),&Istart,&Iend);CHKERRQ(ierr);
  for(Ii=Istart;Ii<Iend;Ii++) {
    y = 0;
    z = _dz * Ii;
    // set slip velocity on the fault
    v = MMS_uA_t(y,z,time);
    ierr = VecSetValues(*(dvarBegin+1),1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(*(dvarBegin+1));CHKERRQ(ierr);
  ierr = VecAssemblyEnd(*(dvarBegin+1));CHKERRQ(ierr);

  // d/dt viscous strains
  VecSet(*(dvarBegin+2),0.0);
  VecSet(*(dvarBegin+3),0.0);
  ierr = setViscStrainsAndRates(time,varBegin,varEnd,dvarBegin,dvarEnd);CHKERRQ(ierr);
  ierr = addMMSViscStrainsAndRates(time,varBegin,varEnd,dvarBegin,dvarEnd);CHKERRQ(ierr);

  VecDestroy(&viscSource);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode IceSheet::setViscStrainsAndRates(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                 it_vec dvarBegin,it_vec dvarEnd)
{
    PetscErrorCode ierr = 0;
    string funcName = "IceSheet::setViscStrainsAndRates";
    string fileName = "iceSheet.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif

  // compute strains and rates
  MatMult(_sbpP._Dy_Iz,_uP,_epsTotxyP);
  VecScale(_epsTotxyP,0.5);

  MatMult(_sbpP._Iy_Dz,_uP,_epsTotxzP);
  VecScale(_epsTotxzP,0.5);

  PetscScalar R = 8.314; // gas constant (J/K/mol)
  PetscScalar deps,epsTotxy,epsTotxz,epsVxy,epsVxz,sigmaxy,sigmaxz,sigmaE = 0;
  PetscScalar A,n,temp;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(*(dvarBegin+2),&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    VecGetValues(_A,1,&Ii,&A);
    VecGetValues(_n,1,&Ii,&n);
    VecGetValues(_temp,1,&Ii,&temp);
    VecGetValues(_epsTotxyP,1,&Ii,&epsTotxy);
    VecGetValues(*(varBegin+2),1,&Ii,&epsVxy);

    // solve for stressxyP = 2*mu*epsExy (elastic strain)
    //                     = 2*mu*(0.5*d/dy(uhat) - epsVxy)
    sigmaxy = 2.0 * _muArrPlus[Ii] * (epsTotxy - epsVxy);
    VecSetValues(_stressxyP,1,&Ii,&sigmaxy,INSERT_VALUES);

    if (_Nz > 1) {
      VecGetValues(_epsTotxzP,1,&Ii,&epsTotxz);
      VecGetValues(*(varBegin+3),1,&Ii,&epsVxz);

      // solve for stressxzP = 2*mu*epsExy (elastic strain)
      //                     = 2*mu*(0.5*d/dz(uhat) - epsVxz)
      sigmaxz = 2.0 * _muArrPlus[Ii] * (epsTotxz - epsVxz);
      VecSetValues(_stressxzP,1,&Ii,&sigmaxz,INSERT_VALUES);
    }

    sigmaE = sqrt(0.5 * (pow(sigmaxy,2.0) + pow(sigmaxz,2.0)));

    // d/dt epsVxy
    deps = A * pow(sigmaE,n-1) * sigmaxy * exp(-1/R/temp);
    VecSetValues(*(dvarBegin+2),1,&Ii,&deps,INSERT_VALUES);

    if (_Nz > 1) {
      // d/dt epsVxz
      deps = A * pow(sigmaE,n-1) * sigmaxz * exp(-1/R/temp);
      VecSetValues(*(dvarBegin+3),1,&Ii,&deps,INSERT_VALUES);
    }
  }
  VecAssemblyBegin(_stressxyP);
  VecAssemblyBegin(*(dvarBegin+2));

  VecAssemblyEnd(_stressxyP);
  VecAssemblyEnd(*(dvarBegin+2));

  if (_Nz > 1) {
    VecAssemblyBegin(_stressxzP);
    VecAssemblyBegin(*(dvarBegin+3));

    VecAssemblyEnd(_stressxzP);
    VecAssemblyEnd(*(dvarBegin+3));
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
      CHKERRQ(ierr);
  #endif
  return ierr = 0;
}
