#include "maxwellViscoelastic.hpp"


SymmMaxwellViscoelastic::SymmMaxwellViscoelastic(Domain& D)
: SymmLinearElastic(D), _file(D._file),_delim(D._delim),_inputDir(D._inputDir),
  _visc(NULL),
  //~_visc(D._visc),
  _epsVxyP(NULL),_depsVxyP(NULL),
  _epsVxzP(NULL),_depsVxzP(NULL),
  _epsVxyPV(NULL),_depsVxyPV(NULL),
  _epsVxzPV(NULL),_depsVxzPV(NULL),
  _epsTotxyP(NULL),_epsTotxzP(NULL),
  _epsTotxyPV(NULL),_epsTotxzPV(NULL),
  _stressxzP(NULL),_stressxyPV(NULL),_stressxzPV(NULL)
{
  #if VERBOSE > 1
    string funcName = "SymmMaxwellViscoelastic::SymmMaxwellViscoelastic";
    string fileName = "maxwellViscoelastic.cpp";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());
  #endif

  // set viscosity
  loadSettings(_file);
  //~if (_viscDistribution.compare("loadFromFile")==0) {

  setVisc();
  checkInput();

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

SymmMaxwellViscoelastic::~SymmMaxwellViscoelastic()
{
  #if VERBOSE > 1
    string funcName = "SymmMaxwellViscoelastic::~SymmMaxwellViscoelastic";
    string fileName = "maxwellViscoelastic.cpp";
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

  // call odeSolver routine integrate here
  _quadrature->setTolerance(_atol);CHKERRQ(ierr);
  _quadrature->setTimeStepBounds(_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
  ierr = _quadrature->setTimeRange(_initTime,_maxTime);
  ierr = _quadrature->setInitialConds(_fault._var);CHKERRQ(ierr);

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
  ierr = VecDuplicate(_epsVxyP,&viscSource);CHKERRQ(ierr);
  ierr = VecCopy(*(varBegin+2),_epsVxyP);CHKERRQ(ierr);
  ierr = VecCopy(*(varBegin+3),_epsVxzP);CHKERRQ(ierr);
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
  ierr = _sbpP.muxDy(_uP,_stressxyP); CHKERRQ(ierr);
  ierr = _fault.setTauQS(_stressxyP,NULL);CHKERRQ(ierr);

  // set rates for slip and state
  ierr = _fault.d_dt(varBegin,varEnd, dvarBegin, dvarEnd);

  // compute viscous strains and strain rates
  ierr = setViscStrainRates(time,varBegin,varEnd,dvarBegin,dvarEnd);CHKERRQ(ierr);

  VecDestroy(&viscSource);

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

  PetscInt Ii,Istart,Iend; // d/dt u
  PetscScalar y,z,v;

  MMS_uA(_uAnal,time);
  //~MMS_epsVxy(_epsVxyP,time);
  //~MMS_epsVxz(_epsVxzP,time);

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
  //~ierr = MatMult(_sbpP._muxDy_Iz,_uP,_stressxyP);CHKERRQ(ierr);
  ierr = _sbpP.muxDy(_uP,_stressxyP); CHKERRQ(ierr);
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
  ierr = setViscStrainRates(time,varBegin,varEnd,dvarBegin,dvarEnd);CHKERRQ(ierr);
  ierr = addMMSViscStrainsAndRates(time,varBegin,varEnd,dvarBegin,dvarEnd);CHKERRQ(ierr);

  VecDestroy(&viscSource);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode SymmMaxwellViscoelastic::setViscStrainSourceTerms(Vec& out)
{
  PetscErrorCode ierr = 0;
  string funcName = "SymmMaxwellViscoelastic::setViscStrainSourceTerms";
  string fileName = "maxwellViscoelastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif

  Vec source;
  VecDuplicate(_epsVxyP,&source);


  // add source terms to rhs: d/dy( 2*mu*epsV_xy) + d/dz( 2*mu*epsV_xz)
  // + Hz^-1 E0z mu epsV_xz + Hz^-1 ENz mu epsV_xz
  Vec sourcexy_y;
  VecDuplicate(_epsVxyP,&sourcexy_y);
  ierr = _sbpP.Dyxmu(_epsVxyP,sourcexy_y);CHKERRQ(ierr);
  ierr = VecScale(sourcexy_y,2.0);CHKERRQ(ierr);
  ierr = VecCopy(sourcexy_y,source);CHKERRQ(ierr); // sourcexy_y -> source
  VecDestroy(&sourcexy_y);

  if (_Nz > 1)
  {
    Vec sourcexz_z;
    VecDuplicate(_epsVxzP,&sourcexz_z);
    ierr = _sbpP.Dzxmu(_epsVxzP,sourcexz_z);CHKERRQ(ierr);
    ierr = VecScale(sourcexz_z,2.0);CHKERRQ(ierr);

    ierr = VecAXPY(source,1.0,sourcexz_z);CHKERRQ(ierr); // source += Hxsourcexz_z
    VecDestroy(&sourcexz_z);

    Vec temp1,bcT,bcB;
    VecDuplicate(_epsVxzP,&temp1);
    VecDuplicate(_epsVxzP,&bcT);
    VecDuplicate(_epsVxzP,&bcB);

    _sbpP.HzinvxE0z(_epsVxzP,temp1);
    ierr = MatMult(_muP,temp1,bcT); CHKERRQ(ierr);

    _sbpP.HzinvxENz(_epsVxzP,temp1);
    ierr = MatMult(_muP,temp1,bcB); CHKERRQ(ierr);

    ierr = VecAXPY(source,2.0,bcT);CHKERRQ(ierr);
    ierr = VecAXPY(source,-2.0,bcB);CHKERRQ(ierr);

    VecDestroy(&temp1);
    VecDestroy(&bcT);
    VecDestroy(&bcB);
  }

  ierr = _sbpP.H(source,out); CHKERRQ(ierr);
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

  ierr = _sbpP.HyinvxE0y(u,temp1);CHKERRQ(ierr);
  ierr = _sbpP.Hyinvxe0y(gL,GL);CHKERRQ(ierr);
  VecAXPY(out,-_sbpP._alphaDy,temp1);
  VecAXPY(out,_sbpP._alphaDy,GL);

  ierr = _sbpP.HyinvxENy(u,temp1);CHKERRQ(ierr);
  ierr = _sbpP.HyinvxeNy(gR,GR);CHKERRQ(ierr);
  VecAXPY(out,_sbpP._alphaDy,temp1);
  VecAXPY(out,-_sbpP._alphaDy,GL);

  VecDestroy(&GL);
  VecDestroy(&GR);
  VecDestroy(&temp1);


  //~VecSet(out,0.0);


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

  // compute strains and rates
  _sbpP.Dy(_uP,_epsTotxyP);
  VecScale(_epsTotxyP,0.5);

  _sbpP.Dz(_uP,_epsTotxzP);
  VecScale(_epsTotxzP,0.5);

  // add SAT terms to strain rate for epsxy
  Vec SAT;
  VecDuplicate(_epsTotxzP,&SAT);
  ierr = setViscousStrainRateSAT(_uP,_bcLP,_bcRP,SAT);CHKERRQ(ierr);

  PetscScalar deps,visc,epsTot,epsVisc,sigmaxy,sigmaxz,sat;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(*(dvarBegin+2),&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    VecGetValues(_visc,1,&Ii,&visc);
    VecGetValues(_epsTotxyP,1,&Ii,&epsTot);
    VecGetValues(*(varBegin+2),1,&Ii,&epsVisc);
    VecGetValues(SAT,1,&Ii,&sat);

    // solve for stressxyP = 2*mu*epsExy (elastic strain)
    //                     = 2*mu*(0.5*d/dy(uhat) - epsVxy)
    sigmaxy = 2.0 * _muArrPlus[Ii] * (epsTot - epsVisc);
    VecSetValues(_stressxyP,1,&Ii,&sigmaxy,INSERT_VALUES);

    // d/dt epsVxy = mu/visc * ( 0.5*d/dy u - epsxy) - SAT
    deps = _muArrPlus[Ii]/visc * (epsTot - epsVisc) + _muArrPlus[Ii]/visc * sat;
    VecSetValues(*(dvarBegin+2),1,&Ii,&deps,INSERT_VALUES);

    if (_Nz > 1) {
      VecGetValues(_epsTotxzP,1,&Ii,&epsTot);
      VecGetValues(*(varBegin+3),1,&Ii,&epsVisc);

      // solve for stressxzP = 2*mu*epsExy (elastic strain)
      //                     = 2*mu*(0.5*d/dz(uhat) - epsVxz)
      sigmaxz = 2.0 * _muArrPlus[Ii] * (epsTot - epsVisc);
      VecSetValues(_stressxzP,1,&Ii,&sigmaxz,INSERT_VALUES);

      // d/dt epsVxz = mu/visc * ( 0.5*d/dz u - epsxz)
      deps = _muArrPlus[Ii]/visc * (epsTot - epsVisc);
      VecSetValues(*(dvarBegin+3),1,&Ii,&deps,INSERT_VALUES);
    }
  }
  VecAssemblyBegin(_stressxyP);
  VecAssemblyBegin(*(dvarBegin+2));

  VecAssemblyEnd(_stressxyP);
  VecAssemblyEnd(*(dvarBegin+2));

  VecDestroy(&SAT);



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


PetscErrorCode SymmMaxwellViscoelastic::setMMSInitialConditions()
{
  PetscErrorCode ierr = 0;
  string funcName = "SymmLinearElastic::setMMSInitialConditions";
  string fileName = "lithosphere.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),fileName.c_str());CHKERRQ(ierr);
  #endif

  PetscScalar time = _initTime;

  PetscInt Ii,Istart,Iend;
  PetscScalar y,z,v;

  MMS_uA(_uAnal,time);
  MMS_epsVxy(_epsVxyP,time);
  MMS_epsVxz(_epsVxzP,time);


  // set up boundary conditions and add source term
  ierr = setMMSBoundaryConditions(time);CHKERRQ(ierr);
  Vec viscSource;
  ierr = setViscStrainSourceTerms(viscSource);CHKERRQ(ierr);
  //~ierr = setMMSuSourceTerms(viscSource,time);CHKERRQ(ierr);

  ierr = VecAXPY(_rhsP,1.0,viscSource);CHKERRQ(ierr); // rhs = rhs + source


  // solve for displacement
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspP,_rhsP,_uP);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  // solve for shear stress
  //~ierr = MatMult(_sbpP._muxDy_Iz,_uP,_stressxyP);CHKERRQ(ierr);
  ierr = _sbpP.muxDy(_uP,_stressxyP); CHKERRQ(ierr);

  // update fields on fault
  ierr = _fault.setTauQS(_stressxyP,NULL);CHKERRQ(ierr);

  // update rates
  ierr = VecSet(*(_fault._var.begin()),0.0);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(*(_fault._var.begin()+1),&Istart,&Iend);CHKERRQ(ierr);
  for(Ii=Istart;Ii<Iend;Ii++) {
    y = 0;
    z = _dz * Ii;
    v = MMS_uA_t(y,z,time);
    ierr = VecSetValues(*(_fault._var.begin()+1),1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(*(_fault._var.begin()+1));CHKERRQ(ierr);
  ierr = VecAssemblyEnd(*(_fault._var.begin()+1));CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(_epsVxyP,&Istart,&Iend);CHKERRQ(ierr);
  PetscScalar visc;
  for(Ii=Istart;Ii<Iend;Ii++) {
    y = _dy*(Ii/_Nz);
    z = _dz*(Ii-_Nz*(Ii/_Nz));
    VecGetValues(_visc,1,&Ii,&visc);

    v = MMS_epsVxy_t_source(y,z,time);
    ierr = VecSetValues(_depsVxyP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);

    v = MMS_epsVxz_t_source(y,z,time);
    ierr = VecSetValues(_depsVxzP,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_depsVxyP);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(_depsVxzP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_depsVxyP);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_depsVxzP);CHKERRQ(ierr);

  writeVec(_depsVxyP,"depsVxyP");
  writeVec(_depsVxzP,"depsVxzP");

  VecDestroy(&viscSource);


  #if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s.\n",funcName.c_str(),fileName.c_str());
  #endif
  return ierr;
}

// MMS distribution for viscosity
double SymmMaxwellViscoelastic::MMS_visc(const double y,const double z)
{
  return cos(y)*cos(z) + 2.0;
}

// MMS analytical distribution for: viscous strain xy epsVxy
double SymmMaxwellViscoelastic::MMS_epsVxy(const double y,const double z,const double t)
{
  return 0.5 * MMS_uA_y(y,z,t);
}

// Vec form of MMS analytical distribution for: viscous strain xy epsVxy
PetscErrorCode SymmMaxwellViscoelastic::MMS_epsVxy(Vec& vec,const double time)
{
  PetscErrorCode ierr = 0;
  PetscScalar y,z,v;
  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(vec,&Istart,&Iend);
  for (Ii=Istart; Ii<Iend; Ii++) {
    y = _dy*(Ii/_Nz);
    z = _dz*(Ii-_Nz*(Ii/_Nz));
    v = MMS_epsVxy(y,z,time);
    ierr = VecSetValues(vec,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);
  return ierr;
}

// MMS analytical distribution for: d/dy viscous strain xy epsVxy
double SymmMaxwellViscoelastic::MMS_epsVxy_y(const double y,const double z,const double t)
{
  return 0.5 * MMS_uA_yy(y,z,t);
}

// MMS analytical distribution for: d/dt viscous strain xy epsVxy
double SymmMaxwellViscoelastic::MMS_epsVxy_t_source(const double y,const double z,const double t)
{
  return -1.0 * MMS_epsVxy(y,z,t);
}

// MMS analytical distribution for: viscous strain xz epsVxz
double SymmMaxwellViscoelastic::MMS_epsVxz(const double y,const double z,const double t)
{
  return 0.5 * MMS_uA_z(y,z,t);
}

// Vec form of MMS analytical distribution for: viscous strain xz epsVxz
PetscErrorCode SymmMaxwellViscoelastic::MMS_epsVxz(Vec& vec,const double time)
{
  PetscErrorCode ierr = 0;
  PetscScalar y,z,v;
  PetscInt Ii,Istart,Iend;
  ierr = VecGetOwnershipRange(vec,&Istart,&Iend);
  for (Ii=Istart; Ii<Iend; Ii++) {
    y = _dy*(Ii/_Nz);
    z = _dz*(Ii-_Nz*(Ii/_Nz));
    v = MMS_epsVxz(y,z,time);
    ierr = VecSetValues(vec,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);
  return ierr;
}

// MMS analytical distribution for: d/dz viscous strain xz epsVxz
double SymmMaxwellViscoelastic::MMS_epsVxz_z(const double y,const double z,const double t)
{
  return 0.5 * MMS_uA_zz(y,z,t);
}

// MMS analytical distribution for: d/dt viscous strain xz epsVxz
double SymmMaxwellViscoelastic::MMS_epsVxz_t_source(const double y,const double z,const double t)
{
  return -1.0 * MMS_epsVxz(y,z,t);
}



PetscErrorCode SymmMaxwellViscoelastic::addMMSViscStrainsAndRates(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                 it_vec dvarBegin,it_vec dvarEnd)
{
    PetscErrorCode ierr = 0;
    string funcName = "SymmMaxwellViscoelastic::setMMSViscStrainsAndRates";
    string fileName = "maxwellViscoelastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s: time=%.15e\n",funcName.c_str(),fileName.c_str(),time);
    CHKERRQ(ierr);
  #endif

  PetscScalar v,y,z;
  PetscInt Ii,Istart,Iend;
  VecGetOwnershipRange(*(dvarBegin+2),&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    y = _dy*(Ii/_Nz);
    z = _dz*(Ii-_Nz*(Ii/_Nz));

    v = MMS_epsVxy_t_source(y,z,time);
    VecSetValues(*(dvarBegin+2),1,&Ii,&v,ADD_VALUES);

    if (_Nz > 1) {
      v = MMS_epsVxz_t_source(y,z,time);
      VecSetValues(*(dvarBegin+3),1,&Ii,&v,ADD_VALUES);
    }
  }
  VecAssemblyBegin(*(dvarBegin+2));
  VecAssemblyEnd(*(dvarBegin+2));

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


PetscErrorCode SymmMaxwellViscoelastic::measureMMSError()
{
  PetscErrorCode ierr = 0;

  // measure error between uAnal and _uP (the numerical solution)
  //~double errH = computeNormDiff_Mat(_sbpP._H,_uP,_uAnal);
  double errH = 111111;
  double err2 = computeNormDiff_2(_uP,_uAnal);

  PetscPrintf(PETSC_COMM_WORLD,"%3i %.4e %.4e % .15e %.4e % .15e\n",
              _Ny,_dy,err2,log2(err2),errH,log2(errH));

  return ierr;
}


PetscErrorCode SymmMaxwellViscoelastic::timeMonitor(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec varEnd,
                             const_it_vec dvarBegin,const_it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;

  if ( stepCount % _strideLength == 0) {
    //~_stepCount++;
    _currTime = time;
    _stepCount = stepCount;
    //~ierr = PetscViewerHDF5IncrementTimestep(D->viewer);CHKERRQ(ierr);

    ierr = writeStep();CHKERRQ(ierr);

  }

#if VERBOSE > 0
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %.15e\n",stepCount,_currTime);CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode SymmMaxwellViscoelastic::writeStep()
{
  PetscErrorCode ierr = 0;
  string funcName = "SymmMaxwellViscoelastic::writeStep";
  string fileName = "maxwellViscoelastic.cpp";
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s at step %i\n",funcName.c_str(),fileName.c_str(),_stepCount);
    CHKERRQ(ierr);
  #endif

  double startTime = MPI_Wtime();

  if (_stepCount==0) {
    // write contextual fields
    ierr = _sbpP.writeOps(_outputDir);CHKERRQ(ierr);
    ierr = _fault.writeContext(_outputDir);CHKERRQ(ierr);

    // output viscosity vector
    string str =  _outputDir + "visc";
    PetscViewer viewer;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,str.c_str(),FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(_visc,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(_outputDir+"time.txt").c_str(),&_timeViewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_timeViewer, "%.15e\n",_currTime);CHKERRQ(ierr);

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


   // output body fields
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"uBodyP").c_str(),
              FILE_MODE_WRITE,&_uPV);CHKERRQ(ierr);
    ierr = VecView(_uP,_uPV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_uPV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"uBodyP").c_str(),
                                   FILE_MODE_APPEND,&_uPV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"totStrainxyP").c_str(),
              FILE_MODE_WRITE,&_epsTotxyPV);CHKERRQ(ierr);
    ierr = VecView(_epsTotxyP,_epsTotxyPV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_epsTotxyPV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"totStrainxyP").c_str(),
                                   FILE_MODE_APPEND,&_epsTotxyPV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"stressxyP").c_str(),
              FILE_MODE_WRITE,&_stressxyPV);CHKERRQ(ierr);
    ierr = VecView(_stressxyP,_stressxyPV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_stressxyPV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"stressxyP").c_str(),
                                   FILE_MODE_APPEND,&_stressxyPV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"epsVxyP").c_str(),
              FILE_MODE_WRITE,&_epsVxyPV);CHKERRQ(ierr);
    ierr = VecView(_epsVxyP,_epsVxyPV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_epsVxyPV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"epsVxyP").c_str(),
                                   FILE_MODE_APPEND,&_epsVxyPV);CHKERRQ(ierr);
    if (_isMMS) {
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"uAnal").c_str(),
                FILE_MODE_WRITE,&_uAnalV);CHKERRQ(ierr);
      ierr = VecView(_uAnal,_uAnalV);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_uAnalV);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"uAnal").c_str(),
                                     FILE_MODE_APPEND,&_uAnalV);CHKERRQ(ierr);
    }
    if (_Nz>1)
    {
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"totStrainxzP").c_str(),
              FILE_MODE_WRITE,&_epsTotxzPV);CHKERRQ(ierr);
      ierr = VecView(_epsTotxzP,_epsTotxzPV);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_epsTotxzPV);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"totStrainxzP").c_str(),
                                     FILE_MODE_APPEND,&_epsTotxzPV);CHKERRQ(ierr);

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"stressxzP").c_str(),
               FILE_MODE_WRITE,&_stressxzPV);CHKERRQ(ierr);
      ierr = VecView(_stressxzP,_stressxzPV);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_stressxzPV);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"stressxzP").c_str(),
                                     FILE_MODE_APPEND,&_stressxzPV);CHKERRQ(ierr);

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"epsVxzP").c_str(),
               FILE_MODE_WRITE,&_epsVxzPV);CHKERRQ(ierr);
      ierr = VecView(_epsVxzP,_epsVxzPV);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_epsVxzPV);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"epsVxzP").c_str(),
                                   FILE_MODE_APPEND,&_epsVxzPV);CHKERRQ(ierr);
    }

    ierr = _fault.writeStep(_outputDir,_stepCount);CHKERRQ(ierr);
    _stepCount++;
  }
  else {
    ierr = PetscViewerASCIIPrintf(_timeViewer, "%.15e\n",_currTime);CHKERRQ(ierr);
    ierr = _fault.writeStep(_outputDir,_stepCount);CHKERRQ(ierr);

    ierr = VecView(_surfDispPlus,_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_bcRP,_bcRPlusV);CHKERRQ(ierr);
    ierr = VecView(_uP,_uPV);CHKERRQ(ierr);
    ierr = VecView(_epsTotxyP,_epsTotxyPV);CHKERRQ(ierr);
    ierr = VecView(_stressxyP,_stressxyPV);CHKERRQ(ierr);
    ierr = VecView(_epsVxyP,_epsVxyPV);CHKERRQ(ierr);
    if (_isMMS) {ierr = VecView(_uAnal,_uAnalV);CHKERRQ(ierr);}
    if (_Nz>1)
    {
      ierr = VecView(_epsTotxzP,_epsTotxzPV);CHKERRQ(ierr);
      ierr = VecView(_stressxzP,_stressxzPV);CHKERRQ(ierr);
      ierr = VecView(_epsVxzP,_epsVxzPV);CHKERRQ(ierr);
    }
  }

  //~ierr = VecView(_epsVxyP,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //~ierr = VecView(_epsVxyP,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


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

    // viscosity for asthenosphere
    if (var.compare("viscDistribution")==0) {
      _viscDistribution = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("viscVals")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_viscVals);
    }
    else if (var.compare("viscDepths")==0) {
      string str = line.substr(pos+_delim.length(),line.npos);
      loadVectorFromInputFile(str,_viscDepths);
    }

  }

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending loadData in domain.cpp.\n");CHKERRQ(ierr);
#endif
  return ierr;
}

// set viscosity
PetscErrorCode SymmMaxwellViscoelastic::setVisc()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting setFieldsPlus in maxwellVisc.cpp.\n");CHKERRQ(ierr);
#endif

  PetscInt       Ii;
  PetscScalar    v,z;
  PetscScalar z0,z1,v0,v1;


  ierr = VecCreate(PETSC_COMM_WORLD,&_visc);CHKERRQ(ierr);
  ierr = VecSetSizes(_visc,PETSC_DECIDE,_Ny*_Nz);CHKERRQ(ierr);
  ierr = VecSetFromOptions(_visc);CHKERRQ(ierr);


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
        //~PetscPrintf(PETSC_COMM_WORLD,"  ind=%i: z0 = %g | z1 = %g | v0 = %g  | v1 = %g\n",ind,z0,z1,v0,v1);
        if (z>=z0 && z<=z1) {
          v = (v1 - v0)/(z1-z0) * (z-z0) + v0;
          v = pow(10,v);
          }
        ierr = VecSetValues(_visc,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_visc);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_visc);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending setFieldsPlus in domain.cpp.\n");CHKERRQ(ierr);
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

  assert(_viscVals.size() == _viscDepths.size() );

  assert(_viscDistribution.compare("layered")==0 ||
      _viscDistribution.compare("mms")==0 ||
      _viscDistribution.compare("loadFromFile")==0 );

#if VERBOSE > 1
ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending Domain::checkInputPlus in maxwellViscoelastic.cpp.\n");CHKERRQ(ierr);
#endif
  //~}
  return ierr;
}
