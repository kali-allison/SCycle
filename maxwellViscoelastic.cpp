#include "maxwellViscoelastic.hpp"

// ONLY WORKS FOR 1D SYMMETRIC PROBLEMS!!!!

SymmMaxwellViscoelastic::SymmMaxwellViscoelastic(Domain& D)
: SymmLinearElastic(D), _visc(D._visc),
  _epsVxyP(NULL),_depsVxyP(NULL),
  _epsVxzP(NULL),_depsVxzP(NULL),
  _epsVxyPV(NULL),_depsVxyPV(NULL),
  _epsVxzPV(NULL),_depsVxzPV(NULL),
  _epsTotxyP(NULL),_epsTotxzP(NULL),
  _epsTotxyPV(NULL),_epsTotxzPV(NULL),
  _stressxzP(NULL),_stressxyPV(NULL),_stressxzPV(NULL)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting maxwellViscoelastic::maxwellViscoelastic in maxwellViscoelastic.cpp\n");
  #endif

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
  if (_Nz > 1) { _fault._var.push_back(_epsVxzP); }

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
    PetscPrintf(PETSC_COMM_WORLD,"Ending maxwellViscoelastic::~maxwellViscoelastic in maxwellViscoelastic.cpp\n");
  #endif
}


PetscErrorCode SymmMaxwellViscoelastic::integrate()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting LinearElastic::integrate in lithosphere.cpp\n");CHKERRQ(ierr);
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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending LinearElastic::integrate in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


// update initial conds after BCs have been set by exterior function
PetscErrorCode SymmMaxwellViscoelastic::resetInitialConds()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting maxwellViscoelastic::resetInitialConds in maxwellViscoelastic.cpp\n");CHKERRQ(ierr);
  #endif

  ierr = _sbpP.setRhs(_rhsP,_bcLP,_bcRP,_bcTP,_bcBP);CHKERRQ(ierr);

  ierr = KSPSolve(_kspP,_rhsP,_uP);CHKERRQ(ierr);

  ierr = MatMult(_sbpP._muxDy_Iz,_uP,_stressxyP);CHKERRQ(ierr);
  ierr = _fault.setTauQS(_stressxyP,_stressxyP);CHKERRQ(ierr);
  ierr = _fault.setFaultDisp(_bcLP,_bcLP);CHKERRQ(ierr);
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
  ierr = VecCopy(*(varBegin+1),_bcLP);CHKERRQ(ierr);
  ierr = VecScale(_bcLP,0.5);CHKERRQ(ierr); // var holds slip velocity, bcL is displacement at y=0+
  ierr = VecSet(_bcRP,_vL*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_bcRP,1.0,_bcRPShift);CHKERRQ(ierr);

  // add source terms to rhs: d/dy( 2*mu*strainV_xy) + d/dz( 2*mu*strainV_xz)
  Vec sourcexy,sourcexy_y;
  PetscScalar epsxy,epsxy_y;
  VecDuplicate(_epsVxyP,&sourcexy);
  VecDuplicate(_epsVxyP,&sourcexy_y);
  VecGetOwnershipRange(_epsVxyP,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    VecGetValues(*(varBegin+2),1,&Ii,&epsxy);
    epsxy_y = 2.0 * _muArrPlus[Ii] * epsxy;
    VecSetValues(sourcexy,1,&Ii,&epsxy_y,INSERT_VALUES);
  }
  VecAssemblyBegin(sourcexy);
  VecAssemblyEnd(sourcexy);

  ierr = MatMult(_sbpP._Dy_Iz,sourcexy,sourcexy_y);CHKERRQ(ierr);
  Vec Hxsourcexy_y;
  VecDuplicate(sourcexy_y,&Hxsourcexy_y);
  ierr = MatMult(_sbpP._H,sourcexy_y,Hxsourcexy_y);

  // set up rhs vector
  ierr = _sbpP.setRhs(_rhsP,_bcLP,_bcRP,_bcTP,_bcBP);CHKERRQ(ierr); // update rhs from BCs
  ierr = VecAXPY(_rhsP,1.0,Hxsourcexy_y);CHKERRQ(ierr);


  // clean up memory used thus far
  VecDestroy(&sourcexy);
  VecDestroy(&sourcexy_y);
  VecDestroy(&Hxsourcexy_y);

  if (_Nz > 1)
  {
    Vec sourcexz,sourcexz_z;
    VecDuplicate(_epsVxzP,&sourcexz);
    VecDuplicate(_epsVxzP,&sourcexz_z);
    PetscScalar epsxz,epsxz_z;
    VecGetOwnershipRange(_epsVxyP,&Istart,&Iend);
    for (Ii=Istart;Ii<Iend;Ii++) {
      VecGetValues(*(varBegin+3),1,&Ii,&epsxz);
      epsxz_z = 2.0 * _muArrPlus[Ii] * epsxz;
      VecSetValues(sourcexz,1,&Ii,&epsxz_z,INSERT_VALUES);
    }
    VecAssemblyBegin(sourcexz);
    VecAssemblyEnd(sourcexz);

    ierr = MatMult(_sbpP._Iy_Dz,sourcexz,sourcexz_z);CHKERRQ(ierr);
    Vec Hxsourcexz_z;
    VecDuplicate(sourcexz_z,&Hxsourcexz_z);
    ierr = MatMult(_sbpP._H,sourcexz_z,Hxsourcexz_z);

    // include strain epsxz in rhs vector
    ierr = VecAXPY(_rhsP,1.0,Hxsourcexz_z);CHKERRQ(ierr);

    // clean up memory
    VecDestroy(&sourcexz);
    VecDestroy(&sourcexz_z);
    VecDestroy(&Hxsourcexz_z);
  }


  // solve fo rdisplacement
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspP,_rhsP,_uP);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();




  // compute strains and rates more cleanly by iterating over vectors
  // (this may be slower, depending how it's parallelized)
  MatMult(_sbpP._Dy_Iz,_uP,_epsTotxyP);
  VecScale(_epsTotxyP,0.5);

  MatMult(_sbpP._Iy_Dz,_uP,_epsTotxzP);
  VecScale(_epsTotxzP,0.5);

  PetscScalar deps,visc,epsTot,epsVisc,sigmaxy,sigmaxz;
  VecGetOwnershipRange(*(dvarBegin+2),&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    VecGetValues(_visc,1,&Ii,&visc);
    VecGetValues(_epsTotxyP,1,&Ii,&epsTot);
    VecGetValues(*(varBegin+2),1,&Ii,&epsVisc);

    // solve for stressxyP = 2*mu*epsExy (elastic strain)
    //                     = 2*mu*(0.5*d/dy(uhat) - epsVxy)
    sigmaxy = 2.0 * _muArrPlus[Ii] * (epsTot - epsVisc);
    VecSetValues(_stressxyP,1,&Ii,&sigmaxy,INSERT_VALUES);

    // d/dt epsVxy = mu/visc * ( 0.5*d/dy u - epsxy)
    deps = _muArrPlus[Ii]/visc * (epsTot - epsVisc);
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

  if (_Nz > 1) {
    VecAssemblyBegin(_stressxzP);
    VecAssemblyBegin(*(dvarBegin+3));

    VecAssemblyEnd(_stressxzP);
    VecAssemblyEnd(*(dvarBegin+3));
  }


    // set shear traction on fault from this
  ierr = _fault.setTauQS(_stressxyP,NULL);CHKERRQ(ierr);

  // set rates for slip and state
  ierr = _fault.d_dt(varBegin,varEnd, dvarBegin, dvarEnd);

/*
  // Set dvar values to test integration function.
  // To use these tests, comment out all the above code.
  VecSet(*dvarBegin,0.0);
  VecSet(*(dvarBegin+1),0.0);
  //~VecSet(*(dvarBegin+2),0.0);
  VecSet(*(dvarBegin+3),0.0);

  // df/dt = 1 -> f(t) = t
  //~VecSet(*(dvarBegin+2),1.0); // checked: 8/10/2015 6:08 pm

  // df/dt = time -> f(t) = 0.5 t^2
  //~VecSet(*(dvarBegin+2),time); // checked: 8/10/2015 6:09 pm

  // df/dt = 5*time -> f(t) = 0.5 * 5 * t^2
  //~VecSet(*(dvarBegin+2),5.0 * time); // checked: 8/10/2015 6:10 pm

  // df/dt = 5*(time - f) -> t + 1/5 * [exp(-5*t) -1]
  // checked: 8/10/2015 6:16 pm
  //~VecSet(*(dvarBegin+2),5.0*time);
  //~VecAXPY(*(dvarBegin+2),-5.0,*(varBegin+2));
*/


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
    // write contextual fields
    ierr = _sbpP.writeOps(_outputDir);CHKERRQ(ierr);
    ierr = _fault.writeContext(_outputDir);CHKERRQ(ierr);

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

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"depsVxyP").c_str(),
              FILE_MODE_WRITE,&_depsVxyPV);CHKERRQ(ierr);
    ierr = VecView(_depsVxyP,_depsVxyPV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_depsVxyPV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"depsVxyP").c_str(),
                                   FILE_MODE_APPEND,&_depsVxyPV);CHKERRQ(ierr);
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

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"depsVxzP").c_str(),
               FILE_MODE_WRITE,&_depsVxzPV);CHKERRQ(ierr);
      ierr = VecView(_depsVxzP,_depsVxzPV);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&_depsVxzPV);CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"depsVxzP").c_str(),
                                   FILE_MODE_APPEND,&_depsVxzPV);CHKERRQ(ierr);
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
    ierr = VecView(_depsVxyP,_depsVxyPV);CHKERRQ(ierr);
    if (_Nz>1)
    {
      ierr = VecView(_epsTotxzP,_epsTotxzPV);CHKERRQ(ierr);
      ierr = VecView(_stressxzP,_stressxzPV);CHKERRQ(ierr);
      ierr = VecView(_epsVxzP,_epsVxzPV);CHKERRQ(ierr);
      ierr = VecView(_depsVxzP,_depsVxzPV);CHKERRQ(ierr);
    }
  }


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
