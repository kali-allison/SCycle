#include "maxwellViscoelastic.hpp"

// ONLY WORKS FOR 1D SYMMETRIC PROBLEMS!!!!

SymmMaxwellViscoelastic::SymmMaxwellViscoelastic(Domain& D)
: SymmLinearElastic(D), _visc(D._visc),
  _strainV_xyPlus(NULL),_dstrainV_xyPlus(NULL),
  _strainV_xzPlus(NULL),_dstrainV_xzPlus(NULL),
  _strainV_xyPlusV(NULL),_dstrainV_xyPlusV(NULL),
  _strainV_xzPlusV(NULL),_dstrainV_xzPlusV(NULL),
  _epsTotxy(NULL),_epsTotxyV(NULL),
    _sigma_xyPlusV(NULL)
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

  VecDuplicate(_uPlus,&_epsTotxy);
  VecSet(_epsTotxy,0.0);

  // add viscous strain to integrated variables, stored in _fault._var
  _fault._var.push_back(_strainV_xyPlus);
  if (_Nz > 1) { _fault._var.push_back(_strainV_xzPlus); }

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
  PetscInt Ii,Istart,Iend;

  // update boundaries
  ierr = VecCopy(*(varBegin+1),_bcLPlus);CHKERRQ(ierr);
  ierr = VecScale(_bcLPlus,0.5);CHKERRQ(ierr); // var holds slip velocity, bcL is displacement at y=0+
  ierr = VecSet(_bcRPlus,_vL*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_bcRPlus,1.0,_bcRPlusShift);CHKERRQ(ierr);

  // update strains
  ierr = VecCopy(*(varBegin+2),_strainV_xyPlus);CHKERRQ(ierr);
  if (_Nz > 1) { ierr = VecCopy(*(varBegin+3),_strainV_xzPlus);CHKERRQ(ierr); }


  // add source terms to rhs: d/dy( 2*mu*strainV_xy) + d/dz( 2*mu*strainV_xz)
  Vec sourcexy,sourcexz,sourcexy_y,sourcexz_z;
  VecDuplicate(_strainV_xyPlus,&sourcexy);
  VecDuplicate(_strainV_xyPlus,&sourcexy_y);
  VecDuplicate(_strainV_xzPlus,&sourcexz);
  VecDuplicate(_strainV_xzPlus,&sourcexz_z);
  PetscScalar epsxy,epsxz,epsxy_y,epsxz_z;
  VecGetOwnershipRange(_strainV_xyPlus,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    VecGetValues(_strainV_xyPlus,1,&Ii,&epsxy);
    VecGetValues(_strainV_xzPlus,1,&Ii,&epsxz);

    epsxy_y = 2.0 * _muArrPlus[Ii] * epsxy;
    epsxz_z = 2.0 * _muArrPlus[Ii] * epsxz;

    VecSetValues(sourcexy,1,&Ii,&epsxy_y,INSERT_VALUES);
    VecSetValues(sourcexz,1,&Ii,&epsxz_z,INSERT_VALUES);
  }
  VecAssemblyBegin(sourcexy); VecAssemblyBegin(sourcexz);
  VecAssemblyEnd(sourcexy);   VecAssemblyEnd(sourcexz);

  ierr = MatMult(_sbpPlus._Dy_Iz,sourcexy,sourcexy_y);CHKERRQ(ierr);
  ierr = MatMult(_sbpPlus._Iy_Dz,sourcexz,sourcexz_z);CHKERRQ(ierr);

  Vec Hxsourcexy_y, Hxsourcexz_z;
  VecDuplicate(sourcexy_y,&Hxsourcexy_y);
  VecDuplicate(sourcexz_z,&Hxsourcexz_z);
  ierr = MatMult(_sbpPlus._H,sourcexy_y,Hxsourcexy_y);
  ierr = MatMult(_sbpPlus._H,sourcexz_z,Hxsourcexz_z);

  // solve for displacement
  ierr = _sbpPlus.setRhs(_rhsPlus,_bcLPlus,_bcRPlus,_bcTPlus,_bcBPlus);CHKERRQ(ierr); // update rhs from BCs
  ierr = VecAXPY(_rhsPlus,1.0,Hxsourcexy_y);CHKERRQ(ierr);
  ierr = VecAXPY(_rhsPlus,1.0,Hxsourcexz_z);CHKERRQ(ierr);

  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspPlus,_rhsPlus,_uPlus);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  VecDestroy(&sourcexy);
  VecDestroy(&sourcexz);
  VecDestroy(&sourcexy_y);
  VecDestroy(&sourcexz_z);
  VecDestroy(&Hxsourcexy_y);
  VecDestroy(&Hxsourcexz_z);


  // compute strains and rates more cleanly by iterating over vectors
  // (this may be slower, depending how it's parallelized)
  Vec epsTotxy,epsTotxz;
  VecDuplicate(_sigma_xyPlus,&epsTotxy);
  VecDuplicate(_sigma_xyPlus,&epsTotxz);
  MatMult(_sbpPlus._Dy_Iz,_uPlus,epsTotxy);
  VecScale(epsTotxy,0.5);
  MatMult(_sbpPlus._Iy_Dz,_uPlus,epsTotxz);
  VecScale(epsTotxz,0.5);

  // temporarily save total strain epsxy
  VecCopy(epsTotxy,_epsTotxy);

  PetscScalar deps,visc,epsTot,epsVisc,sigmaxy;
  VecGetOwnershipRange(*(dvarBegin+2),&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    VecGetValues(_visc,1,&Ii,&visc);
    VecGetValues(epsTotxy,1,&Ii,&epsTot);
    VecGetValues(_strainV_xyPlus,1,&Ii,&epsVisc);

    // solve for sigma_xyPlus = 2*mu*strain_xyElastic
    //                        = 2*mu*(0.5*d/dy(uhat) - strainVisc_xy)
    sigmaxy = 2.0 * _muArrPlus[Ii] * (epsTot - epsVisc);
    VecSetValues(_sigma_xyPlus,1,&Ii,&sigmaxy,INSERT_VALUES);

    // d/dt epsxy = mu/visc * ( 0.5*d/dy u - epsxy)
    deps = _muArrPlus[Ii]/visc * (epsTot - epsVisc);
    VecSetValues(*(dvarBegin+2),1,&Ii,&deps,INSERT_VALUES);

    // d/dt epsxz = mu/visc * ( 0.5*d/dz u - epsxz)
    VecGetValues(epsTotxz,1,&Ii,&epsTot);
    VecGetValues(_strainV_xzPlus,1,&Ii,&epsVisc);
    deps = _muArrPlus[Ii]/visc * (epsTot - epsVisc);
    VecSetValues(*(dvarBegin+3),1,&Ii,&deps,INSERT_VALUES);
  }
  VecAssemblyBegin(_sigma_xyPlus);
  VecAssemblyBegin(*(dvarBegin+2));
  VecAssemblyBegin(*(dvarBegin+3));

  VecAssemblyEnd(_sigma_xyPlus);
  VecAssemblyEnd(*(dvarBegin+2));
  VecAssemblyEnd(*(dvarBegin+3));

  VecDestroy(&epsTotxy);
  VecDestroy(&epsTotxz);


    // set shear traction on fault from this
  ierr = _fault.setTauQS(_sigma_xyPlus,NULL);CHKERRQ(ierr);

  // set rates for slip and state
  ierr = _fault.d_dt(varBegin,varEnd, dvarBegin, dvarEnd);

  //~// set velocity to zero, keeping fault locked!!!!!!!!
  //~VecSet(*(dvarBegin+0),0.0);
  //~VecSet(*(dvarBegin+1),0.0);



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

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"dispPlus").c_str(),
             FILE_MODE_WRITE,&_uPlusV);CHKERRQ(ierr);
    ierr = VecView(_uPlus,_uPlusV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_uPlusV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"dispPlus").c_str(),
                                   FILE_MODE_APPEND,&_uPlusV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcRPlus").c_str(),
             FILE_MODE_WRITE,&_bcRPlusV);CHKERRQ(ierr);
    ierr = VecView(_bcRPlus,_bcRPlusV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_bcRPlusV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcRPlus").c_str(),
                                   FILE_MODE_APPEND,&_bcRPlusV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"strainV_xyPlus").c_str(),
             FILE_MODE_WRITE,&_strainV_xyPlusV);CHKERRQ(ierr);
    ierr = VecView(_strainV_xyPlus,_strainV_xyPlusV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_strainV_xyPlusV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"strainV_xyPlus").c_str(),
                                   FILE_MODE_APPEND,&_strainV_xyPlusV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"dstrainV_xyPlus").c_str(),
             FILE_MODE_WRITE,&_dstrainV_xyPlusV);CHKERRQ(ierr);
    ierr = VecView(_dstrainV_xyPlus,_dstrainV_xyPlusV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_dstrainV_xyPlusV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"dstrainV_xyPlus").c_str(),
                                   FILE_MODE_APPEND,&_dstrainV_xyPlusV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"sigma_xyPlus").c_str(),
             FILE_MODE_WRITE,&_sigma_xyPlusV);CHKERRQ(ierr);
    ierr = VecView(_sigma_xyPlus,_sigma_xyPlusV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_sigma_xyPlusV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"sigma_xyPlus").c_str(),
                                   FILE_MODE_APPEND,&_sigma_xyPlusV);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"epsTotxy").c_str(),
             FILE_MODE_WRITE,&_epsTotxyV);CHKERRQ(ierr);
    ierr = VecView(_epsTotxy,_epsTotxyV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_epsTotxyV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"epsTotxy").c_str(),
                                   FILE_MODE_APPEND,&_epsTotxyV);CHKERRQ(ierr);

    if (_Nz>1)
    {
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"strainV_xzPlus").c_str(),
               FILE_MODE_WRITE,&_strainV_xzPlusV);CHKERRQ(ierr);
      ierr = VecView(_strainV_xzPlus,_strainV_xzPlusV);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_strainV_xzPlusV);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"strainV_xzPlus").c_str(),
                                     FILE_MODE_APPEND,&_strainV_xzPlusV);CHKERRQ(ierr);

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"dstrainV_xzPlus").c_str(),
               FILE_MODE_WRITE,&_dstrainV_xzPlusV);CHKERRQ(ierr);
      ierr = VecView(_dstrainV_xzPlus,_dstrainV_xzPlusV);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_dstrainV_xzPlusV);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"dstrainV_xzPlus").c_str(),
                                     FILE_MODE_APPEND,&_dstrainV_xzPlusV);CHKERRQ(ierr);
    }
  }
  else {
    ierr = VecView(_surfDispPlus,_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_strainV_xyPlus,_strainV_xyPlusV);CHKERRQ(ierr);
    ierr = VecView(_dstrainV_xyPlus,_dstrainV_xyPlusV);CHKERRQ(ierr);
    ierr = VecView(_sigma_xyPlus,_sigma_xyPlusV);CHKERRQ(ierr);
    ierr = VecView(_bcRPlus,_bcRPlusV);CHKERRQ(ierr);
    ierr = VecView(_uPlus,_uPlusV);CHKERRQ(ierr);
    ierr = VecView(_epsTotxy,_epsTotxyV);CHKERRQ(ierr);

    if (_Nz>1)
    {
      ierr = VecView(_strainV_xzPlus,_strainV_xzPlusV);CHKERRQ(ierr);
      ierr = VecView(_dstrainV_xyPlus,_dstrainV_xyPlusV);CHKERRQ(ierr);
    }
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
