#include "lithosphere.hpp"


//================= constructor and destructor ========================

Lithosphere::Lithosphere(Domain&D)
: _order(D._order),_Ny(D._Ny),_Nz(D._Nz),
  _Ly(D._Ly),_Lz(D._Lz),_dy(_Ly/(_Ny-1.)),_dz(_Lz/(_Nz-1.)),
  _outputDir(D._outputDir),
  _v0(D._v0),_vp(D._vp),
  _rhoIn(D._rhoIn),_rhoOut(D._rhoOut),_muIn(D._muIn),_muOut(D._muOut),_muArr(D._muArr),_mu(D._mu),
  _depth(D._depth),_width(D._width),
  _sbp(D),_fault(D),
  _strideLength(D._strideLength),_maxStepCount(D._maxStepCount),
  _initTime(D._initTime),_currTime(_initTime),_maxTime(D._maxTime),_minDeltaT(D._minDeltaT),_maxDeltaT(D._maxDeltaT),
  _stepCount(0),_atol(D._atol),_initDeltaT(D._initDeltaT),
  _integrateTime(0),_writeTime(0),_linSolveTime(0),_factorTime(0),_linSolveCount(0)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting constructor in lithosphere.cpp.\n");
#endif

  // boundary conditions
  VecCreate(PETSC_COMM_WORLD,&_bcF);
  VecSetSizes(_bcF,PETSC_DECIDE,_Nz);
  VecSetFromOptions(_bcF);     PetscObjectSetName((PetscObject) _bcF, "_bcF");
  VecSet(_bcF,0.0);

  VecDuplicate(_bcF,&_bcRShift); PetscObjectSetName((PetscObject) _bcRShift, "_bcRShift");
  _bcRShift = _fault.getBcRShift();
  //~VecSet(_bcRShift,0.0); // !!!!!!!!
  VecDuplicate(_bcF,&_bcR); PetscObjectSetName((PetscObject) _bcR, "_bcR");
  VecSet(_bcR,_vp*_initTime/2.0);
  VecAXPY(_bcR,1.0,_bcRShift);

  VecCreate(PETSC_COMM_WORLD,&_bcS);
  VecSetSizes(_bcS,PETSC_DECIDE,_Ny);
  VecSetFromOptions(_bcS);     PetscObjectSetName((PetscObject) _bcS, "_bcS");
  VecSet(_bcS,0.0);

  VecDuplicate(_bcS,&_bcD); PetscObjectSetName((PetscObject) _bcD, "_bcD");
  VecSet(_bcD,0.0);

  KSPCreate(PETSC_COMM_WORLD,&_ksp);
  setupKSP();

  VecCreate(PETSC_COMM_WORLD,&_rhs);
  VecSetSizes(_rhs,PETSC_DECIDE,_Ny*_Nz);
  VecSetFromOptions(_rhs);
  _sbp.setRhs(_rhs,_bcF,_bcR,_bcS,_bcD);

  VecDuplicate(_rhs,&_uhat);
  KSPSolve(_ksp,_rhs,_uhat);

  VecDuplicate(_rhs,&_sigma_xy);
  MatMult(_sbp._Dy_Iz,_uhat,_sigma_xy);
  _fault.setTau(_sigma_xy);
  _fault.setFaultDisp(_bcF);

  VecDuplicate(_bcS,&_surfDisp); PetscObjectSetName((PetscObject) _surfDisp, "_surfDisp");
  setSurfDisp();

  //~_quadrature = new FEuler(_maxStepCount,_maxTime,_initDeltaT);
  _quadrature = new RK32(_maxStepCount,_maxTime,_initDeltaT);
  _quadrature->setTolerance(D._atol);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending constructor in lithosphere.cpp.\n");
#endif
}

Lithosphere::~Lithosphere()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting destructor in lithosphere.cpp.\n");
#endif

  // boundary conditions
  VecDestroy(&_bcF);
  VecDestroy(&_bcR);
  VecDestroy(&_bcS);
  VecDestroy(&_bcD);

  // body fields
  VecDestroy(&_rhs);
  //~VecDestroy(&_uhat);
  //~VecDestroy(&_sigma_xy);
//~
  //~KSPDestroy(&_ksp);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending destructor in lithosphere.cpp.\n");
#endif
}

PetscErrorCode Lithosphere::view()
{
  PetscErrorCode ierr = 0;
  ierr = _quadrature->view();
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Timing Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in integration (s): %g\n",_integrateTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of times linear system was solved: %i\n",_linSolveCount);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent solving linear system (s): %g\n",_linSolveTime);CHKERRQ(ierr);
  return ierr;
}

//===================== private member functions =======================
/*
PetscErrorCode Lithosphere::setShearModulus()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting setShearModulus in lithosphere.cpp.\n");CHKERRQ(ierr);
#endif

  PetscInt       Ii;
  PetscScalar    v,y,z;

  Vec muVec;
  PetscInt *muInds;
  ierr = PetscMalloc(_Ny*_Nz*sizeof(PetscInt),&muInds);CHKERRQ(ierr);
  ierr = PetscMalloc(_Ny*_Nz*sizeof(PetscInt),&_muArr);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&muVec);CHKERRQ(ierr);
  ierr = VecSetSizes(muVec,PETSC_DECIDE,_Ny*_Nz);CHKERRQ(ierr);
  ierr = VecSetFromOptions(muVec);CHKERRQ(ierr);

  PetscScalar r = 0;
  PetscScalar rbar = 0.25*_width*_width;
  PetscScalar rw = 1+0.5*_width/_depth;
  for (Ii=0;Ii<_Ny*_Nz;Ii++) {
    z = _dz*(Ii-_Nz*(Ii/_Nz));
    y = _dy*(Ii/_Nz);
    r=y*y+(0.25*_width*_width/_depth/_depth)*z*z;
    v = 0.5*(_muOut-_muIn)*(tanh((double)(r-rbar)/rw)+1) + _muIn;
    _muArr[Ii] = v;
    muInds[Ii] = Ii;
  }
  ierr = VecSetValues(muVec,_Ny*_Nz,muInds,_muArr,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(muVec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(muVec);CHKERRQ(ierr);

  ierr = MatSetSizes(_mu,PETSC_DECIDE,PETSC_DECIDE,_Ny*_Nz,_Ny*_Nz);CHKERRQ(ierr);
  ierr = MatSetFromOptions(_mu);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(_mu,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(_mu,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(_mu);CHKERRQ(ierr);
  ierr = MatDiagonalSet(_mu,muVec,INSERT_VALUES);CHKERRQ(ierr);

  VecDestroy(&muVec);
  PetscFree(muInds);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting setShearModulus in lithosphere.cpp.\n");CHKERRQ(ierr);
#endif
return ierr;
}
*/


PetscErrorCode Lithosphere::computeShearStress()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting computeShearStress in lithosphere.cpp.\n");CHKERRQ(ierr);
#endif

  ierr = MatMult(_sbp._Dy_Iz,_uhat,_sigma_xy);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending computeShearStress in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode Lithosphere::setupKSP()
{
  PetscErrorCode ierr = 0;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting setupKSP in lithosphere.cpp\n");CHKERRQ(ierr);
#endif

  //~ierr = KSPSetType(D.ksp,KSPGMRES);CHKERRQ(ierr);
  //~ierr = KSPSetOperators(D.ksp,D.A,D.A,SAME_PRECONDITIONER);CHKERRQ(ierr);
  //~ierr = KSPGetPC(D.ksp,&D.pc);CHKERRQ(ierr);

  ierr = KSPSetType(_ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPSetOperators(_ksp,_sbp._A,_sbp._A,SAME_PRECONDITIONER);CHKERRQ(ierr);
  ierr = KSPGetPC(_ksp,&_pc);CHKERRQ(ierr);

  // use PETSc's direct LU - only available on 1 processor!!!
  //~ierr = PCSetType(D.pc,PCLU);CHKERRQ(ierr);

  // use HYPRE
  //~ierr = PCSetType(D.pc,PCHYPRE);CHKERRQ(ierr);
  //~ierr = PCHYPRESetType(D.pc,"boomeramg");CHKERRQ(ierr);
  //~ierr = KSPSetTolerances(D.ksp,D.kspTol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  //~ierr = PCFactorSetLevels(D.pc,4);CHKERRQ(ierr);

  // use direct LU from MUMPS
  PCSetType(_pc,PCLU);
  PCFactorSetMatSolverPackage(_pc,MATSOLVERMUMPS);
  PCFactorSetUpMatSolverPackage(_pc);

  ierr = KSPSetUp(_ksp);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(_ksp);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending setupKSP in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode Lithosphere::setSurfDisp()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting setSurfDisp in lithosphere.cpp\n");CHKERRQ(ierr);
#endif

  PetscInt    Ii,Istart,Iend;
  PetscScalar u,y,z;
  ierr = VecGetOwnershipRange(_uhat,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    z = Ii-_Nz*(Ii/_Nz);
    y = Ii/_Nz;
    if (z == 0) {
      ierr = VecGetValues(_uhat,1,&Ii,&u);CHKERRQ(ierr);
      ierr = VecSetValue(_surfDisp,y,u,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_surfDisp);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_surfDisp);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending setSurfDisp in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode Lithosphere::d_dt(PetscScalar const time,Vec const*var,Vec*dvar)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting d_dt in lithosphere.cpp\n");CHKERRQ(ierr);
#endif

  // update boundaries
  ierr = VecCopy(var[0],_bcF);CHKERRQ(ierr);
  ierr = VecScale(_bcF,0.5);CHKERRQ(ierr);
  ierr = VecSet(_bcR,_vp*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_bcR,1.0,_bcRShift);CHKERRQ(ierr);

  // solve for displacement
  ierr = _sbp.setRhs(_rhs,_bcF,_bcR,_bcS,_bcD);CHKERRQ(ierr);
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_ksp,_rhs,_uhat);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  ierr = MatMult(_sbp._Dy_Iz,_uhat,_sigma_xy);CHKERRQ(ierr);
  ierr = _fault.setTau(_sigma_xy);CHKERRQ(ierr);

  ierr = _fault.d_dt(var,dvar);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending d_dt in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode Lithosphere::timeMonitor(const PetscReal time, const PetscInt stepCount,const Vec* var,const Vec*dvar)
{
  PetscErrorCode ierr = 0;
  //~UserContext*    D = (UserContext*) userContext;

  if ( stepCount % _strideLength == 0) {
    _stepCount++;
    _currTime = time;
    //~ierr = PetscViewerHDF5IncrementTimestep(D->viewer);CHKERRQ(ierr);
    ierr = writeStep();CHKERRQ(ierr);
  }

#if VERBOSE >0
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %.15e\n",stepCount,_currTime);CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode Lithosphere::integrate()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting integrate in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  double startTime = MPI_Wtime();

  // call odeSolver routine integrate here
  _quadrature->setTolerance(_atol);CHKERRQ(ierr);
  _quadrature->setTimeStepBounds(_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
  ierr = _quadrature->setTimeRange(_initTime,_maxTime);
  ierr = _quadrature->setInitialConds(_fault._var, 2);CHKERRQ(ierr);

  ierr = _quadrature->integrate(this);CHKERRQ(ierr);
  _integrateTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending integrate in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode Lithosphere::writeStep()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting writeStep in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  double startTime = MPI_Wtime();

  if (_stepCount==0) {
    _sbp.writeOps(_outputDir);
    _fault.writeContext(_outputDir);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(_outputDir+"time.txt").c_str(),&_timeViewer);CHKERRQ(ierr);
    PetscViewerASCIIPrintf(_timeViewer, "%.15e\n",_currTime);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDisp").c_str(),FILE_MODE_WRITE,&_surfDispViewer);CHKERRQ(ierr);
    ierr = VecView(_surfDisp,_surfDispViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_surfDispViewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDisp").c_str(),
                                   FILE_MODE_APPEND,&_surfDispViewer);CHKERRQ(ierr);
  }
  _fault.writeStep(_outputDir,_stepCount);
  PetscViewerASCIIPrintf(_timeViewer, "%.15e\n",_currTime);
  ierr = VecView(_surfDisp,_surfDispViewer);CHKERRQ(ierr);

  _writeTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending writeStep in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}



// Outputs data at each time step.
PetscErrorCode Lithosphere::debug(const PetscReal time,const PetscInt steps,const Vec *var,const Vec *dvar,const char *stage)
{
  PetscErrorCode ierr = 0;
  PetscInt       Istart,Iend;
  PetscScalar    gRval,uVal,psiVal,velVal,dQVal;

  PetscScalar k = _muOut/2/_Ly;

  ierr= VecGetOwnershipRange(var[0],&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(var[0],1,&Istart,&uVal);CHKERRQ(ierr);
  ierr = VecGetValues(var[1],1,&Istart,&psiVal);CHKERRQ(ierr);

  ierr= VecGetOwnershipRange(dvar[0],&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(dvar[0],1,&Istart,&velVal);CHKERRQ(ierr);
  ierr = VecGetValues(dvar[1],1,&Istart,&dQVal);CHKERRQ(ierr);

  ierr= VecGetOwnershipRange(_bcR,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(_bcR,1,&Istart,&gRval);CHKERRQ(ierr);

  //~PetscScalar tauVal;
  //~ierr = VecGetValues(_fault._tau,1,&Istart,&tauVal);CHKERRQ(ierr);
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"tau = %e\n",tauVal);CHKERRQ(ierr);

  if (steps == 0) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%-4s %-6s | %-15s %-15s %-15s | %-15s %-15s %-15s | %-9s\n",
                       "Step","Stage","gR","D","Q","VL","V","dQ","time");
    CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%4i %-6s ",steps,stage);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%| %.9e %.9e %.9e ",2*gRval*k,uVal,psiVal);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%| %.9e %.9e %.9e ",_vp/2.,velVal,dQVal);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%| %.9e\n",time);CHKERRQ(ierr);



  return ierr;
}
