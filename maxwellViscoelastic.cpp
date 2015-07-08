#include "maxwellViscoelastic.hpp"

// ONLY WORKS FOR 1D SYMMETRIC PROBLEMS!!!!

SymmMaxwellViscoelastic::SymmMaxwellViscoelastic(Domain& D)
: SymmLinearElastic(D), _visc(D._visc),
  _strainViscPlus(NULL),_dstrainViscPlus(NULL),
  _strainViscPlusV(NULL),_dstrainViscPlusV(NULL)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting maxwellViscoelastic::maxwellViscoelastic in maxwellViscoelastic.cpp\n");
  #endif

  VecDuplicate(_uPlus,&_strainViscPlus); PetscObjectSetName((PetscObject) _strainViscPlus, "_strainViscPlus");
  VecSet(_strainViscPlus,0.0);

  VecDuplicate(_uPlus,&_dstrainViscPlus); PetscObjectSetName((PetscObject) _dstrainViscPlus, "_dstrainViscPlus");
  VecSet(_dstrainViscPlus,0.0);

  // add viscous strain to integrated variables, stored in _fault._var
  _fault._var.push_back(_strainViscPlus);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending maxwellViscoelastic::maxwellViscoelastic in maxwellViscoelastic.cpp\n");
  #endif
}

SymmMaxwellViscoelastic::~SymmMaxwellViscoelastic()
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting maxwellViscoelastic::~maxwellViscoelastic in maxwellViscoelastic.cpp\n");
  #endif

  // from maxwellViscoelastic
  VecDestroy(&_strainViscPlus);
  VecDestroy(&_dstrainViscPlus);
  PetscViewerDestroy(&_strainViscPlusV);
  PetscViewerDestroy(&_dstrainViscPlusV);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending maxwellViscoelastic::~maxwellViscoelastic in maxwellViscoelastic.cpp\n");
  #endif
}


// update initial conds after BCs have been set by exterior function
PetscErrorCode SymmMaxwellViscoelastic::resetInitialConds()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting maxwellViscoelastic::resetInitialConds in maxwellViscoelastic.cpp\n");CHKERRQ(ierr);
  #endif

  ierr = _sbpPlus.setRhs(_rhsPlus,_bcLPlus,_bcRPlus,_bcTPlus,_bcBPlus);CHKERRQ(ierr);

  ierr = KSPSolve(_kspPlus,_rhsPlus,_uPlus);CHKERRQ(ierr);

  ierr = MatMult(_sbpPlus._Dy_Iz,_uPlus,_sigma_xyPlus);CHKERRQ(ierr);
  ierr = _fault.setTauQS(_sigma_xyPlus,_sigma_xyPlus);CHKERRQ(ierr);
  ierr = _fault.setFaultDisp(_bcLPlus,_bcLPlus);CHKERRQ(ierr);
  ierr = _fault.computeVel();CHKERRQ(ierr);

  setSurfDisp();

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting maxwellViscoelastic::resetInitialConds in maxwellViscoelastic.cpp\n");CHKERRQ(ierr);
  #endif
  return ierr;
}

PetscErrorCode SymmMaxwellViscoelastic::d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                 it_vec dvarBegin,it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting maxwellViscoelastic::d_dt in maxwellViscoelastic.cpp\n");CHKERRQ(ierr);
#endif

  // update boundaries
  ierr = VecCopy(*(varBegin+1),_bcLPlus);CHKERRQ(ierr);
  ierr = VecScale(_bcLPlus,0.5);CHKERRQ(ierr); // var holds slip velocity, bcL is displacement at y=0+
  ierr = VecSet(_bcRPlus,_vL*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_bcRPlus,1.0,_bcRPlusShift);CHKERRQ(ierr);

  // solve for displacement
  ierr = _sbpPlus.setRhs(_rhsPlus,_bcLPlus,_bcRPlus,_bcTPlus,_bcBPlus);CHKERRQ(ierr); // update rhs from BCs
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspPlus,_rhsPlus,_uPlus);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

// original code that was here
  //~// set rate for strainDamper
  //~// solve for tauSpring = 2*mu*strainSpring
  //~//                     = mu*d/dy(uhat) - 2*mu*strainDamper
  //~//                     = -2[ -0.5*mu*d/dy(uhat) + mu*strainDamper ]
  //~ierr = MatMult(_sbpPlus._Dy_Iz,_uPlus,_sigma_xyPlus);CHKERRQ(ierr);
  //~ierr = VecScale(_sigma_xyPlus,-0.5);CHKERRQ(ierr); // rather than making a temporary vector to handle subtraction
  //~ierr = MatMultAdd(_muPlus,*(varBegin+2),_sigma_xyPlus,_sigma_xyPlus);
  //~ierr = VecScale(_sigma_xyPlus,-2.0);CHKERRQ(ierr);
//~
  //~ierr = _fault.setTauQS(_sigma_xyPlus,_sigma_xyPlus);CHKERRQ(ierr);
//~
  //~// set rates for faultDisp and state
  //~ierr = _fault.d_dt(varBegin,varEnd, dvarBegin, dvarEnd);
//~
  //~// set rate for strainDamper
  //~// d/dt(strainDamper) = tauSpring/(2*eta)
  //~ierr = VecCopy(_sigma_xyPlus,*(dvarBegin+2));CHKERRQ(ierr);
  //~ierr = VecScale(*(dvarBegin+2),0.5/_visc);CHKERRQ(ierr);
  //~ierr = VecCopy(*(dvarBegin+2),_dstrainViscPlus);CHKERRQ(ierr);



// new code
  /* solve for tauStress = 2*mu*strainLinElastic
   *                     = mu*d/dy(uhat) - 2*mu*strainVisc
   *                     = -2[ -0.5*mu*d/dy(uhat) + mu*strainVisc ]
   */
  ierr = MatMult(_sbpPlus._Dy_Iz,_uPlus,_sigma_xyPlus);CHKERRQ(ierr);
  ierr = VecScale(_sigma_xyPlus,-0.5);CHKERRQ(ierr); // rather than making a temporary vector to handle subtraction
  ierr = MatMultAdd(_muPlus,*(varBegin+2),_sigma_xyPlus,_sigma_xyPlus);
  ierr = VecScale(_sigma_xyPlus,-2.0);CHKERRQ(ierr);


  // set shear traction on fault from this
  ierr = _fault.setTauQS(_sigma_xyPlus,_sigma_xyPlus);CHKERRQ(ierr);

  // set rates for faultDisp and state
  ierr = _fault.d_dt(varBegin,varEnd, dvarBegin, dvarEnd);

  // set rate for viscoelastic strain: d/dt _strainViscPlus = tauStress/(2*eta)
  ierr = VecCopy(_sigma_xyPlus,*(dvarBegin+2));CHKERRQ(ierr);
  ierr = VecScale(*(dvarBegin+2),0.5/_visc);CHKERRQ(ierr);
  //~ierr = VecCopy(*(dvarBegin+2),_dstrainViscPlus);CHKERRQ(ierr); // I think dvar has a shallow copy of dStrainVisc




#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending maxwellViscoelastic::d_dt in maxwellViscoelastic.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

// now that LinearElastic defines fault, don't need this
//~PetscErrorCode maxwellViscoelastic::integrate()
//~{
  //~PetscErrorCode ierr = 0;
//~#if VERBOSE > 1
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting integrate in lithosphere.cpp\n");CHKERRQ(ierr);
//~#endif
  //~double startTime = MPI_Wtime();
//~
  //~// call odeSolver routine integrate here
  //~_quadrature->setTolerance(_atol);CHKERRQ(ierr);
  //~_quadrature->setTimeStepBounds(_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
  //~ierr = _quadrature->setTimeRange(_initTime,_maxTime);
  //~ierr = _quadrature->setInitialConds(_var);CHKERRQ(ierr);
//~
  //~ierr = _quadrature->integrate(this);CHKERRQ(ierr);
  //~_integrateTime += MPI_Wtime() - startTime;
//~#if VERBOSE > 1
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending integrate in lithosphere.cpp\n");CHKERRQ(ierr);
//~#endif
  //~return ierr;
//~}

PetscErrorCode SymmMaxwellViscoelastic::timeMonitor(const PetscReal time,const PetscInt stepCount,
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


PetscErrorCode SymmMaxwellViscoelastic::writeStep()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting maxwellViscoelastic::writeStep in maxwellViscoelastic.cpp at step %i\n",_stepCount);CHKERRQ(ierr);
#endif
  double startTime = MPI_Wtime();

  if (_stepCount==0) {
    ierr = _sbpPlus.writeOps(_outputDir);CHKERRQ(ierr);
    ierr = _fault.writeContext(_outputDir);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(_outputDir+"time.txt").c_str(),&_timeViewer);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDispPlus").c_str(),
                                 FILE_MODE_WRITE,&_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_surfDispPlus,_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDispPlus").c_str(),
                                   FILE_MODE_APPEND,&_surfDispPlusViewer);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"strainViscPlus").c_str(),
             FILE_MODE_WRITE,&_strainViscPlusV);CHKERRQ(ierr);
    ierr = VecView(_strainViscPlus,_strainViscPlusV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_strainViscPlusV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"strainViscPlus").c_str(),
                                   FILE_MODE_APPEND,&_strainViscPlusV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"dstrainViscPlus").c_str(),
             FILE_MODE_WRITE,&_dstrainViscPlusV);CHKERRQ(ierr);
    ierr = VecView(_dstrainViscPlus,_dstrainViscPlusV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_dstrainViscPlusV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"dstrainViscPlus").c_str(),
                                   FILE_MODE_APPEND,&_dstrainViscPlusV);CHKERRQ(ierr);
  }
  else {
    ierr = VecView(_surfDispPlus,_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_strainViscPlus,_strainViscPlusV);CHKERRQ(ierr);
    ierr = VecView(_dstrainViscPlus,_dstrainViscPlusV);CHKERRQ(ierr);
  }
  ierr = _fault.writeStep(_outputDir,_stepCount);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(_timeViewer, "%.15e\n",_currTime);CHKERRQ(ierr);


  _writeTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending writeStep in maxwellViscoelastic.cpp at step %i\n",_stepCount);CHKERRQ(ierr);
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



//======================================================================

TwoDMaxwellViscoelastic::TwoDMaxwellViscoelastic(Domain& D)
: FullLinearElastic(D), _visc(D._visc),
  _strainViscPlus(NULL),_dstrainViscPlus(NULL),
  _strainViscMinus(NULL),_dstrainViscMinus(NULL),
  _strainViscPlusV(NULL),_dstrainViscPlusV(NULL),
  _strainViscMinusV(NULL),_dstrainViscMinusV(NULL)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting TwoDMaxwellViscoelastic::TwoDMaxwellViscoelastic in maxwellViscoelastic.cpp\n");
  #endif

  VecDuplicate(_uPlus,&_strainViscPlus); PetscObjectSetName((PetscObject) _strainViscPlus, "_strainViscPlus");
  VecSet(_strainViscPlus,0.0);
  VecDuplicate(_uPlus,&_strainViscMinus); PetscObjectSetName((PetscObject) _strainViscMinus, "_strainViscMinus");
  VecSet(_strainViscMinus,0.0);

  VecDuplicate(_uPlus,&_dstrainViscPlus); PetscObjectSetName((PetscObject) _dstrainViscPlus, "_dstrainViscPlus");
  VecSet(_dstrainViscPlus,0.0);
  VecDuplicate(_uPlus,&_dstrainViscMinus); PetscObjectSetName((PetscObject) _dstrainViscMinus, "_dstrainViscMinus");
  VecSet(_dstrainViscMinus,0.0);

  // add viscous strain to integrated variables, stored in _fault._var
  _fault._var.push_back(_strainViscPlus);
  _fault._var.push_back(_strainViscMinus);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending TwoDMaxwellViscoelastic::TwoDMaxwellViscoelastic in maxwellViscoelastic.cpp\n");
  #endif
}

TwoDMaxwellViscoelastic::~TwoDMaxwellViscoelastic()
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting TwoDMaxwellViscoelastic::~maxwellViscoelastic in maxwellViscoelastic.cpp\n");
  #endif

  // from maxwellViscoelastic
  VecDestroy(&_strainViscPlus);
  VecDestroy(&_strainViscMinus);
  VecDestroy(&_dstrainViscPlus);
  VecDestroy(&_dstrainViscMinus);
  PetscViewerDestroy(&_strainViscMinusV);
  PetscViewerDestroy(&_strainViscMinusV);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending TwoDMaxwellViscoelastic::~TwoDMaxwellViscoelastic in maxwellViscoelastic.cpp\n");
  #endif
}

// update initial conds after BCs have been set by exterior function
PetscErrorCode TwoDMaxwellViscoelastic::resetInitialConds()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting TwoDMaxwellViscoelastic::resetInitialConds in maxwellViscoelastic.cpp\n");CHKERRQ(ierr);
  #endif

  ierr = _sbpPlus.setRhs(_rhsPlus,_bcLPlus,_bcRPlus,_bcTPlus,_bcBPlus);CHKERRQ(ierr);

  ierr = KSPSolve(_kspPlus,_rhsPlus,_uPlus);CHKERRQ(ierr);

  ierr = MatMult(_sbpPlus._Dy_Iz,_uPlus,_sigma_xyPlus);CHKERRQ(ierr);
  ierr = _fault.setTauQS(_sigma_xyPlus,_sigma_xyPlus);CHKERRQ(ierr);
  ierr = _fault.setFaultDisp(_bcLPlus,_bcLPlus);CHKERRQ(ierr);
  ierr = _fault.computeVel();CHKERRQ(ierr);

  setSurfDisp();

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting TwoDMaxwellViscoelastic::resetInitialConds in maxwellViscoelastic.cpp\n");CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode TwoDMaxwellViscoelastic::d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                 it_vec dvarBegin,it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting TwoDMaxwellViscoelastic::d_dt in maxwellViscoelastic.cpp\n");CHKERRQ(ierr);
#endif

  // update boundaries
  ierr = VecCopy(*(varBegin+1),_bcLPlus);CHKERRQ(ierr);
  ierr = VecScale(_bcLPlus,0.5);CHKERRQ(ierr); // var holds slip velocity, bcL is displacement at y=0+
  ierr = VecSet(_bcRPlus,_vL*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_bcRPlus,1.0,_bcRPlusShift);CHKERRQ(ierr);

  // solve for displacement
  ierr = _sbpPlus.setRhs(_rhsPlus,_bcLPlus,_bcRPlus,_bcTPlus,_bcBPlus);CHKERRQ(ierr); // update rhs from BCs
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspPlus,_rhsPlus,_uPlus);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();


// new code
  /* solve for tauStress = 2*mu*strainLinElastic
   *                     = mu*d/dy(uhat) - 2*mu*strainVisc
   *                     = -2[ -0.5*mu*d/dy(uhat) + mu*strainVisc ]
   */
  ierr = MatMult(_sbpPlus._Dy_Iz,_uPlus,_sigma_xyPlus);CHKERRQ(ierr);
  ierr = VecScale(_sigma_xyPlus,-0.5);CHKERRQ(ierr); // rather than making a temporary vector to handle subtraction
  ierr = MatMultAdd(_muPlus,*(varBegin+2),_sigma_xyPlus,_sigma_xyPlus);
  ierr = VecScale(_sigma_xyPlus,-2.0);CHKERRQ(ierr);

  // set shear traction on fault from this
  ierr = _fault.setTauQS(_sigma_xyPlus,_sigma_xyPlus);CHKERRQ(ierr);

  // set rates for faultDisp and state
  ierr = _fault.d_dt(varBegin,varEnd, dvarBegin, dvarEnd);

  // set rate for viscoelastic strain: d/dt _strainViscPlus = tauStress/(2*eta)
  ierr = VecCopy(_sigma_xyPlus,*(dvarBegin+2));CHKERRQ(ierr);
  ierr = VecScale(*(dvarBegin+2),0.5/_visc);CHKERRQ(ierr);


#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending TwoDMaxwellViscoelastic::d_dt in maxwellViscoelastic.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode TwoDMaxwellViscoelastic::timeMonitor(const PetscReal time,const PetscInt stepCount,
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

PetscErrorCode TwoDMaxwellViscoelastic::writeStep()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting TwoDMaxwellViscoelastic::writeStep in maxwellViscoelastic.cpp at step %i\n",_stepCount);CHKERRQ(ierr);
#endif
  double startTime = MPI_Wtime();

  if (_stepCount==0) {
    ierr = _sbpPlus.writeOps(_outputDir);CHKERRQ(ierr);
    ierr = _fault.writeContext(_outputDir);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(_outputDir+"time.txt").c_str(),&_timeViewer);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDispPlus").c_str(),
                                 FILE_MODE_WRITE,&_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_surfDispPlus,_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDispPlus").c_str(),
                                   FILE_MODE_APPEND,&_surfDispPlusViewer);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"strainVisc").c_str(),
             FILE_MODE_WRITE,&_strainViscPlusV);CHKERRQ(ierr);
    ierr = VecView(_strainViscPlus,_strainViscPlusV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_strainViscPlusV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"strainVisc").c_str(),
                                   FILE_MODE_APPEND,&_strainViscPlusV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"dstrainVisc").c_str(),
             FILE_MODE_WRITE,&_dstrainViscPlusV);CHKERRQ(ierr);
    ierr = VecView(_dstrainViscPlus,_dstrainViscPlusV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_dstrainViscPlusV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"dstrainVisc").c_str(),
                                   FILE_MODE_APPEND,&_dstrainViscPlusV);CHKERRQ(ierr);
  }
  else {
    ierr = VecView(_surfDispPlus,_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_strainViscPlus,_strainViscPlusV);CHKERRQ(ierr);
    ierr = VecView(_dstrainViscPlus,_dstrainViscPlusV);CHKERRQ(ierr);
  }
  ierr = _fault.writeStep(_outputDir,_stepCount);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(_timeViewer, "%.15e\n",_currTime);CHKERRQ(ierr);


  _writeTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending writeStep in TwoDMaxwellViscoelastic.cpp at step %i\n",_stepCount);CHKERRQ(ierr);
#endif
  return ierr;
}

PetscErrorCode TwoDMaxwellViscoelastic::view()
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
