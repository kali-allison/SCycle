#include "maxwellViscoelastic.hpp"

// ONLY WORKS FOR 1D SYMMETRIC PROBLEMS!!!!

SymmMaxwellViscoelastic::SymmMaxwellViscoelastic(Domain& D)
: SymmLinearElastic(D), _visc(D._visc),
  _strainV_xyPlus(NULL),_dstrainV_xyPlus(NULL),
  _strainV_xzPlus(NULL),_dstrainV_xzPlus(NULL),
  _strainV_xyPlusV(NULL),_dstrainV_xyPlusV(NULL),
  _strainV_xzPlusV(NULL),_dstrainV_xzPlusV(NULL)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting maxwellViscoelastic::maxwellViscoelastic in maxwellViscoelastic.cpp\n");
  #endif

  VecDuplicate(_uPlus,&_strainV_xyPlus);
  PetscObjectSetName((PetscObject) _strainV_xyPlus, "_strainV_xyPlus");
  VecSet(_strainV_xyPlus,0.0);
  VecDuplicate(_uPlus,&_dstrainV_xyPlus);
  PetscObjectSetName((PetscObject) _dstrainV_xyPlus, "_dstrainV_xyPlus");
  VecSet(_dstrainV_xyPlus,0.0);

  VecDuplicate(_uPlus,&_strainV_xzPlus);
  PetscObjectSetName((PetscObject) _strainV_xzPlus, "_strainV_xzPlus");
  VecSet(_strainV_xzPlus,0.0);
  VecDuplicate(_uPlus,&_dstrainV_xzPlus);
  PetscObjectSetName((PetscObject) _dstrainV_xzPlus, "_dstrainV_xzPlus");
  VecSet(_dstrainV_xzPlus,0.0);

  // add viscous strain to integrated variables, stored in _fault._var
  _fault._var.push_back(_strainV_xyPlus);
  _fault._var.push_back(_strainV_xzPlus);

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
  VecDestroy(&_strainV_xyPlus);
  VecDestroy(&_strainV_xzPlus);
  VecDestroy(&_dstrainV_xyPlus);
  VecDestroy(&_dstrainV_xzPlus);
  PetscViewerDestroy(&_strainV_xyPlusV);
  PetscViewerDestroy(&_strainV_xzPlusV);
  PetscViewerDestroy(&_dstrainV_xyPlusV);
  PetscViewerDestroy(&_dstrainV_xzPlusV);

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

  ierr = MatMult(_sbpPlus._muxDy_Iz,_uPlus,_sigma_xyPlus);CHKERRQ(ierr);
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

  // update strains
  ierr = VecCopy(*(varBegin+2),_strainV_xyPlus);CHKERRQ(ierr);
  ierr = VecCopy(*(varBegin+3),_strainV_xzPlus);CHKERRQ(ierr);

  // solve for displacement
  ierr = _sbpPlus.setRhs(_rhsPlus,_bcLPlus,_bcRPlus,_bcTPlus,_bcBPlus);CHKERRQ(ierr); // update rhs from BCs

  // add source terms to rhs: d/dy( 2*mu*strainV_xy) + d/dz( 2*mu*strainV_xz)
  Vec strainV_xyPlus_y=NULL;
  ierr = VecDuplicate(_strainV_xyPlus,&strainV_xyPlus_y);CHKERRQ(ierr);
  ierr = MatMult(_sbpPlus._Dy_Izx2mu,_strainV_xyPlus,strainV_xyPlus_y);CHKERRQ(ierr);

  Vec strainV_xzPlus_z=NULL;
  ierr = VecDuplicate(_strainV_xzPlus,&strainV_xzPlus_z);CHKERRQ(ierr);
  ierr = MatMult(_sbpPlus._Iy_Dzx2mu,_strainV_xzPlus,strainV_xzPlus_z);CHKERRQ(ierr);

  ierr = VecAXPY(_rhsPlus,1.0,strainV_xyPlus_y);CHKERRQ(ierr);
  ierr = VecAXPY(_rhsPlus,1.0,strainV_xzPlus_z);CHKERRQ(ierr);

  VecDestroy(&strainV_xyPlus_y);
  VecDestroy(&strainV_xzPlus_z);

  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspPlus,_rhsPlus,_uPlus);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();


  /* solve for sigma_xyPlus = 2*mu*strain_xyElastic
   *                        = mu*d/dy(uhat) - 2*mu*strainVisc_xy
   *                        = -2[ -0.5*mu*d/dy(uhat) + mu*strainVisc_xy ]
   */
  ierr = MatMult(_sbpPlus._muxDy_Iz,_uPlus,_sigma_xyPlus);CHKERRQ(ierr);
  ierr = VecScale(_sigma_xyPlus,-0.5);CHKERRQ(ierr); // rather than making a temporary vector to handle subtraction
  ierr = MatMultAdd(_muPlus,*(varBegin+2),_sigma_xyPlus,_sigma_xyPlus);
  ierr = VecScale(_sigma_xyPlus,-2.0);CHKERRQ(ierr);

  // set shear traction on fault from this
  ierr = _fault.setTauQS(_sigma_xyPlus,NULL);CHKERRQ(ierr);

  // set rates for slip and state
  ierr = _fault.d_dt(varBegin,varEnd, dvarBegin, dvarEnd);

  // set rate for viscoelastic strain_xy: d/dt _strainVPlus = tauStress/(2*eta)
  ierr = VecCopy(_sigma_xyPlus,*(dvarBegin+2));CHKERRQ(ierr);
  ierr = VecScale(*(dvarBegin+2),0.5/_visc);CHKERRQ(ierr);


  /* solve for sigma_xzPlus = 2*mu*strain_xzElastic
   *                        = mu*d/dz(uhat) - 2*mu*strainVisc_xz
   *                        = -2[ -0.5*mu*d/dz(uhat) + mu*strainVisc_xz ]
   */
  Vec strainTot_xz,temp;
  ierr = VecDuplicate(_sigma_xyPlus,&strainTot_xz);CHKERRQ(ierr);
  ierr = VecDuplicate(_sigma_xyPlus,&temp);CHKERRQ(ierr);
  ierr = MatMult(_sbpPlus._Iy_Dz,_uPlus,strainTot_xz);CHKERRQ(ierr);
  ierr = VecWAXPY(temp,-1.0,_strainV_xzPlus,strainTot_xz);CHKERRQ(ierr);
  ierr = MatMult(_muPlus,temp,*(dvarBegin+3));CHKERRQ(ierr);
  ierr = VecScale(*(dvarBegin+3),1.0/_visc);CHKERRQ(ierr);

  VecDestroy(&temp);
  VecDestroy(&strainTot_xz);


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
             FILE_MODE_WRITE,&_strainV_xyPlusV);CHKERRQ(ierr);
    ierr = VecView(_strainV_xyPlus,_strainV_xyPlusV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_strainV_xyPlusV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"strainViscPlus").c_str(),
                                   FILE_MODE_APPEND,&_strainV_xyPlusV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"dstrainViscPlus").c_str(),
             FILE_MODE_WRITE,&_dstrainV_xyPlusV);CHKERRQ(ierr);
    ierr = VecView(_dstrainV_xyPlus,_dstrainV_xyPlusV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_dstrainV_xyPlusV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"dstrainViscPlus").c_str(),
                                   FILE_MODE_APPEND,&_dstrainV_xyPlusV);CHKERRQ(ierr);
  }
  else {
    ierr = VecView(_surfDispPlus,_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_strainV_xyPlus,_strainV_xyPlusV);CHKERRQ(ierr);
    ierr = VecView(_dstrainV_xyPlus,_dstrainV_xyPlusV);CHKERRQ(ierr);
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
