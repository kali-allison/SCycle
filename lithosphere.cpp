#include "lithosphere.hpp"

// allow this to work when coupled or not
//~enum CONTROL { controlP,controlPI,controlPID };
//~const CONTROL controlType = controlPID;

using namespace std;

//================= constructor and destructor ========================

Lithosphere::Lithosphere(Domain&D)
: _order(D._order),_Ny(D._Ny),_Nz(D._Nz),
  _Ly(D._Ly),_Lz(D._Lz),_dy(_Ly/(_Ny-1.)),_dz(_Lz/(_Nz-1.)),
  _outputDir(D._outputDir),
  _v0(D._v0),_vp(D._vp),
  _rhoIn(D._rhoIn),_rhoOut(D._rhoOut),_muIn(D._muIn),_muOut(D._muOut),_muArr(D._muArr),_mu(D._mu),
  _depth(D._depth),_width(D._width),
  _linSolver(D._linSolver),_kspTol(D._kspTol),
  _sbp(D),
  _timeIntegrator(D._timeIntegrator),
  _strideLength(D._strideLength),_maxStepCount(D._maxStepCount),
  _initTime(D._initTime),_currTime(_initTime),_maxTime(D._maxTime),_minDeltaT(D._minDeltaT),_maxDeltaT(D._maxDeltaT),
  _stepCount(0),_atol(D._atol),_initDeltaT(D._initDeltaT),
  _timeViewer(NULL),_surfDispViewer(NULL),
  _integrateTime(0),_writeTime(0),_linSolveTime(0),_factorTime(0),_linSolveCount(0),
  _fault(D)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting constructor in lithosphere.cpp.\n");
#endif

  // boundary conditions
  VecCreate(PETSC_COMM_WORLD,&_bcF);
  VecSetSizes(_bcF,PETSC_DECIDE,_Nz);
  VecSetFromOptions(_bcF);     PetscObjectSetName((PetscObject) _bcF, "_bcF");
  VecSet(_bcF,0.0);
  //~VecSet(_bcF,_vp*_initTime/2.0);

  _bcRShift = _fault.getBcRShift();
  VecDuplicate(_bcF,&_bcR); PetscObjectSetName((PetscObject) _bcR, "_bcR");
  VecSet(_bcR,_vp*_initTime/2.0);
  VecAXPY(_bcR,1.0,_bcRShift);
  VecSet(_bcRShift,5.0);
  VecSet(_bcR,5.0);

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
  _fault.computeVel();

  VecDuplicate(_bcS,&_surfDisp); PetscObjectSetName((PetscObject) _surfDisp, "_surfDisp");
  setSurfDisp();

  if (_timeIntegrator.compare("FEuler")==0) {
    _quadrature = new FEuler(_maxStepCount,_maxTime,_initDeltaT,D._timeControlType);
  }
  else if (_timeIntegrator.compare("RK32")==0) {
    _quadrature = new RK32(_maxStepCount,_maxTime,_initDeltaT,D._timeControlType);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: timeIntegrator type type not understood\n");
    assert(0>1); // automatically fail, because I can't figure out how to use exit commands properly
  }

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
  VecDestroy(&_uhat);
  VecDestroy(&_sigma_xy);
  VecDestroy(&_surfDisp);

  KSPDestroy(&_ksp);

  PetscViewerDestroy(&_timeViewer);
  PetscViewerDestroy(&_surfDispViewer);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending destructor in lithosphere.cpp.\n");
#endif
}

//===================== private member functions =======================


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

  //~ierr = KSPSetType(_ksp,KSPGMRES);CHKERRQ(ierr);
  //~ierr = KSPSetOperators(_ksp,_A,_A,SAME_PRECONDITIONER);CHKERRQ(ierr);
  //~ierr = KSPGetPC(_ksp,&_pc);CHKERRQ(ierr);



  // use PETSc's direct LU - only available on 1 processor!!!
  //~ierr = PCSetType(D.pc,PCLU);CHKERRQ(ierr);

  if (_linSolver.compare("AMG")==0) {
    // use HYPRE
    ierr = KSPSetType(_ksp,KSPRICHARDSON);CHKERRQ(ierr);
    ierr = KSPSetOperators(_ksp,_sbp._A,_sbp._A,SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = KSPGetPC(_ksp,&_pc);CHKERRQ(ierr);
    ierr = PCSetType(_pc,PCHYPRE);CHKERRQ(ierr);
    ierr = PCHYPRESetType(_pc,"boomeramg");CHKERRQ(ierr);
    ierr = KSPSetTolerances(_ksp,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    ierr = PCFactorSetLevels(_pc,4);CHKERRQ(ierr);
    //~ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n!!ksp type: HYPRE boomeramg\n\n");CHKERRQ(ierr);
  }

  else if (_linSolver.compare("MUMPSLU")==0) {
    // use direct LU from MUMPS
    ierr = KSPSetType(_ksp,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPSetOperators(_ksp,_sbp._A,_sbp._A,SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = KSPGetPC(_ksp,&_pc);CHKERRQ(ierr);
    PCSetType(_pc,PCLU);
    PCFactorSetMatSolverPackage(_pc,MATSOLVERMUMPS);
    PCFactorSetUpMatSolverPackage(_pc);
    //~ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n!!ksp type: MUMPS direct LU\n\n");CHKERRQ(ierr);
  }
  else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"ERROR: linSolver type not understood\n");
    assert(0>1);
  }

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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting writeStep in lithosphere.cpp at step %i\n",_stepCount);CHKERRQ(ierr);
#endif
  double startTime = MPI_Wtime();

  if (_stepCount==0) {
    ierr = _sbp.writeOps(_outputDir);CHKERRQ(ierr);
    ierr = _fault.writeContext(_outputDir);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(_outputDir+"time.txt").c_str(),&_timeViewer);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDisp").c_str(),FILE_MODE_WRITE,&_surfDispViewer);CHKERRQ(ierr);
    ierr = VecView(_surfDisp,_surfDispViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_surfDispViewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDisp").c_str(),
                                   FILE_MODE_APPEND,&_surfDispViewer);CHKERRQ(ierr);
  }
  else {
    ierr = VecView(_surfDisp,_surfDispViewer);CHKERRQ(ierr);
  }
  ierr = _fault.writeStep(_outputDir,_stepCount);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(_timeViewer, "%.15e\n",_currTime);CHKERRQ(ierr);


  _writeTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending writeStep in lithosphere.cpp at step %i\n",_stepCount);CHKERRQ(ierr);
#endif
  return ierr;
}



// Outputs data at each time step.
PetscErrorCode Lithosphere::debug(const PetscReal time,const PetscInt steps,
                 const vector<Vec>& var,const vector<Vec>& dvar,const char *stage)
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

// =====================================================================
//               OnlyLithosphere functions
// =====================================================================
OnlyLithosphere::OnlyLithosphere(Domain& D)
: Lithosphere(D)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting OnlyLithosphere::OnlyLithosphere in lithosphere.cpp\n");
  #endif

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending OnlyLithosphere::OnlyLithosphere in lithosphere.cpp\n");
  #endif
}


PetscErrorCode OnlyLithosphere::d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                 it_vec dvarBegin,it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting OnlyLithosphere::d_dt in lithosphere.cpp\n");CHKERRQ(ierr);
#endif

  // update boundaries
  ierr = VecCopy(*varBegin,_bcF);CHKERRQ(ierr);
  ierr = VecScale(_bcF,0.5);CHKERRQ(ierr);
  ierr = VecSet(_bcR,_vp*time/2.0);CHKERRQ(ierr); // for if I'm only integrating 1 spring-slider
  ierr = VecAXPY(_bcR,1.0,_bcRShift);CHKERRQ(ierr);

  // solve for displacement
  ierr = _sbp.setRhs(_rhs,_bcF,_bcR,_bcS,_bcD);CHKERRQ(ierr);
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_ksp,_rhs,_uhat);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  // solve for shear stress
  ierr = MatMult(_sbp._Dy_Iz,_uhat,_sigma_xy);CHKERRQ(ierr);
  ierr = _fault.setTau(_sigma_xy);CHKERRQ(ierr);

  //~ierr = _fault.d_dt(var,dvar);
  ierr = _fault.d_dt(varBegin,varEnd, dvarBegin, dvarEnd);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending OnlyLithosphere::d_dt in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode OnlyLithosphere::timeMonitor(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec varEnd,
                             const_it_vec dvarBegin,const_it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;

  if ( stepCount % _strideLength == 0) {
    _stepCount++;
    _currTime = time;
    //~ierr = PetscViewerHDF5IncrementTimestep(D->viewer);CHKERRQ(ierr);
    ierr = writeStep();CHKERRQ(ierr);
  }

#if VERBOSE > 0
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %.15e\n",stepCount,_currTime);CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode OnlyLithosphere::view()
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




// =====================================================================
//               CoupledLithosphere functions
// =====================================================================
CoupledLithosphere::CoupledLithosphere(Domain& D)
: Lithosphere(D)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting CoupledLithosphere::CoupledLithosphere in lithosphere.cpp\n");
  #endif

  // boundary conditions
  VecSet(_bcF,0.0);
  VecSet(_bcR,0.0);
  VecSet(_bcS,0.0);
  VecSet(_bcD,0.0);

  _sbp.setRhs(_rhs,_bcF,_bcR,_bcS,_bcD);

  KSPSolve(_ksp,_rhs,_uhat);

  MatMult(_sbp._Dy_Iz,_uhat,_sigma_xy);
  _fault.setTau(_sigma_xy);
  _fault.setFaultDisp(_bcF);
  _fault.computeVel();

  setSurfDisp();

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending CoupledLithosphere::CoupledLithosphere in lithosphere.cpp\n");
  #endif
}

// update initial conds after BCs have been set by exterior function
PetscErrorCode CoupledLithosphere::resetInitialConds()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting CoupledLithosphere::resetInitialConds in lithosphere.cpp\n");CHKERRQ(ierr);
  #endif

  ierr = _sbp.setRhs(_rhs,_bcF,_bcR,_bcS,_bcD);CHKERRQ(ierr);

  ierr = KSPSolve(_ksp,_rhs,_uhat);CHKERRQ(ierr);

  ierr = MatMult(_sbp._Dy_Iz,_uhat,_sigma_xy);CHKERRQ(ierr);
  ierr = _fault.setTau(_sigma_xy);CHKERRQ(ierr);
  ierr = _fault.setFaultDisp(_bcF);CHKERRQ(ierr);
  ierr = _fault.computeVel();CHKERRQ(ierr);

  ierr = setSurfDisp();CHKERRQ(ierr);

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting CoupledLithosphere::resetInitialConds in lithosphere.cpp\n");CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode CoupledLithosphere::d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                 it_vec dvarBegin,it_vec dvarEnd,Vec& tauMod)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting CoupledLithosphere::d_dt in lithosphere.cpp\n");CHKERRQ(ierr);
#endif

  // boundaries updates happen in an exterior function

  // solve for displacement
  ierr = _sbp.setRhs(_rhs,_bcF,_bcR,_bcS,_bcD);CHKERRQ(ierr);
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_ksp,_rhs,_uhat);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  // solve for shear stress
  ierr = MatMult(_sbp._Dy_Iz,_uhat,_sigma_xy);CHKERRQ(ierr);
  ierr = _fault.setTau(_sigma_xy);CHKERRQ(ierr);

  ierr = VecAXPY(_fault._tau,1.0,tauMod);CHKERRQ(ierr); // if it's attached to another spring slider

  ierr = _fault.d_dt(varBegin,varEnd, dvarBegin, dvarEnd);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending CoupledLithosphere::d_dt in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode CoupledLithosphere::d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                 it_vec dvarBegin,it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting CoupledLithosphere::d_dt in lithosphere.cpp\n");CHKERRQ(ierr);
#endif

  // boundaries updates happen in an exterior function

  // solve for displacement
  ierr = _sbp.setRhs(_rhs,_bcF,_bcR,_bcS,_bcD);CHKERRQ(ierr);
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_ksp,_rhs,_uhat);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  // solve for shear stress
  ierr = MatMult(_sbp._Dy_Iz,_uhat,_sigma_xy);CHKERRQ(ierr);
  ierr = _fault.setTau(_sigma_xy);CHKERRQ(ierr);

  ierr = _fault.d_dt(varBegin,varEnd, dvarBegin, dvarEnd);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending CoupledLithosphere::d_dt in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode CoupledLithosphere::timeMonitor(const PetscReal time,const PetscInt stepCount,
                             const_it_vec varBegin,const_it_vec varEnd,
                             const_it_vec dvarBegin,const_it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;

  if ( stepCount % _strideLength == 0) {
    _stepCount++;
    _currTime = time;
    //~ierr = PetscViewerHDF5IncrementTimestep(D->viewer);CHKERRQ(ierr);
    ierr = writeStep();CHKERRQ(ierr);
  }

  return ierr;
}

PetscErrorCode CoupledLithosphere::view()
{
  PetscErrorCode ierr = 0;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in integration (s): %g\n",_integrateTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of times linear system was solved: %i\n",_linSolveCount);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent solving linear system (s): %g\n",_linSolveTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  return ierr;
}

