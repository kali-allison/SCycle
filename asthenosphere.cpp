#include "asthenosphere.hpp"

OnlyAsthenosphere::OnlyAsthenosphere(Domain& D)
: Lithosphere(D), _visc(D._visc),
  _strainDamper(NULL),_strainDamperRate(NULL),_rhsCorrection(NULL),_strainDamperViewer(NULL),_strainDamperRateViewer(NULL)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting OnlyAsthenosphere::OnlyAsthenosphere in asthenosphere.cpp\n");
  #endif

  VecDuplicate(_uhat,&_strainDamper); PetscObjectSetName((PetscObject) _strainDamper, "_strainDamper");
  VecSet(_strainDamper,0.0);

  VecDuplicate(_uhat,&_strainDamperRate); PetscObjectSetName((PetscObject) _strainDamperRate, "_strainDamperRate");
  VecSet(_strainDamperRate,0.0);

  VecDuplicate(_uhat,&_rhsCorrection); PetscObjectSetName((PetscObject) _rhsCorrection, "_rhsCorrection");
  VecSet(_rhsCorrection,0.0);


  // set up initial conditions for integration (shallow copy)
  _var.push_back(_fault._var[0]);
  _var.push_back(_fault._var[1]);
  _var.push_back(_strainDamper);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending OnlyAsthenosphere::OnlyAsthenosphere in asthenosphere.cpp\n");
  #endif
}

OnlyAsthenosphere::~OnlyAsthenosphere()
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting OnlyAsthenosphere::~OnlyAsthenosphere in asthenosphere.cpp\n");
  #endif
  // from Lithosphere
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

  // from OnlyAsthenosphere
  VecDestroy(&_strainDamper);
  VecDestroy(&_strainDamperRate);
  VecDestroy(&_rhsCorrection);
  PetscViewerDestroy(&_strainDamperViewer);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending OnlyAsthenosphere::~OnlyAsthenosphere in asthenosphere.cpp\n");
  #endif
}


// update initial conds after BCs have been set by exterior function
PetscErrorCode OnlyAsthenosphere::resetInitialConds()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting OnlyAsthenosphere::resetInitialConds in asthenosphere.cpp\n");CHKERRQ(ierr);
  #endif

  ierr = _sbp.setRhs(_rhs,_bcF,_bcR,_bcS,_bcD);CHKERRQ(ierr);

  ierr = KSPSolve(_ksp,_rhs,_uhat);CHKERRQ(ierr);

  ierr = MatMult(_sbp._Dy_Iz,_uhat,_sigma_xy);CHKERRQ(ierr);
  ierr = _fault.setTau(_sigma_xy);CHKERRQ(ierr);
  ierr = _fault.setFaultDisp(_bcF);CHKERRQ(ierr);
  ierr = _fault.computeVel();CHKERRQ(ierr);

  setSurfDisp();

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting OnlyAsthenosphere::resetInitialConds in asthenosphere.cpp\n");CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode OnlyAsthenosphere::d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                 it_vec dvarBegin,it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting OnlyAsthenosphere::d_dt in asthenosphere.cpp\n");CHKERRQ(ierr);
#endif

  // update boundaries
  ierr = VecCopy(*varBegin,_bcF);CHKERRQ(ierr);
  ierr = VecScale(_bcF,0.5);CHKERRQ(ierr);
  ierr = VecSet(_bcR,_vp*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_bcR,1.0,_bcRShift);CHKERRQ(ierr);

  // solve for displacement
  ierr = _sbp.setRhs(_rhs,_bcF,_bcR,_bcS,_bcD);CHKERRQ(ierr); // update rhs from BCs
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_ksp,_rhs,_uhat);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  // solve for tauSpring = 2*mu*strainSpring
  //                     = mu*d/dy(uhat) - 2*mu*strainDamper
  //                     = -2[ -0.5*mu*d/dy(uhat) + mu*strainDamper ]
  ierr = MatMult(_sbp._Dy_Iz,_uhat,_sigma_xy);CHKERRQ(ierr);
  ierr = VecScale(_sigma_xy,-0.5);CHKERRQ(ierr); // rather than making a temporary vector to handle subtraction
  ierr = MatMultAdd(_mu,*(varBegin+2),_sigma_xy,_sigma_xy);
  ierr = VecScale(_sigma_xy,-2.0);CHKERRQ(ierr);

  ierr = _fault.setTau(_sigma_xy);CHKERRQ(ierr);

  // set rates for faultDisp and state
  ierr = _fault.d_dt(varBegin,varEnd, dvarBegin, dvarEnd);


  // set rate for strainDamper
  // d/dt(strainDamper) = tauSpring/(2*eta)
  ierr = VecCopy(_sigma_xy,*(dvarBegin+2));CHKERRQ(ierr);
  ierr = VecScale(*(dvarBegin+2),0.5/_visc);CHKERRQ(ierr);
  ierr = VecCopy(*(dvarBegin+2),_strainDamperRate);CHKERRQ(ierr);


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending OnlyAsthenosphere::d_dt in asthenosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode OnlyAsthenosphere::integrate()
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
  ierr = _quadrature->setInitialConds(_var, 3);CHKERRQ(ierr);

  ierr = _quadrature->integrate(this);CHKERRQ(ierr);
  _integrateTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending integrate in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode OnlyAsthenosphere::timeMonitor(const PetscReal time,const PetscInt stepCount,
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


PetscErrorCode OnlyAsthenosphere::writeStep()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting OnlyAsthenosphere::writeStep in asthenosphere.cpp at step %i\n",_stepCount);CHKERRQ(ierr);
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

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"strainDamper").c_str(),
             FILE_MODE_WRITE,&_strainDamperViewer);CHKERRQ(ierr);
    ierr = VecView(_strainDamper,_strainDamperViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_strainDamperViewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"strainDamper").c_str(),
                                   FILE_MODE_APPEND,&_strainDamperViewer);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"strainDamperRate").c_str(),
             FILE_MODE_WRITE,&_strainDamperRateViewer);CHKERRQ(ierr);
    ierr = VecView(_strainDamperRate,_strainDamperRateViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_strainDamperRateViewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"strainDamperRate").c_str(),
                                   FILE_MODE_APPEND,&_strainDamperRateViewer);CHKERRQ(ierr);
  }
  else {
    ierr = VecView(_surfDisp,_surfDispViewer);CHKERRQ(ierr);
    ierr = VecView(_strainDamper,_strainDamperViewer);CHKERRQ(ierr);
    ierr = VecView(_strainDamperRate,_strainDamperRateViewer);CHKERRQ(ierr);
  }
  ierr = _fault.writeStep(_outputDir,_stepCount);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(_timeViewer, "%.15e\n",_currTime);CHKERRQ(ierr);


  _writeTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending writeStep in asthenosphere.cpp at step %i\n",_stepCount);CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode OnlyAsthenosphere::view()
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
