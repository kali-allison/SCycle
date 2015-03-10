#include "lithosphere.hpp"


using namespace std;


Lithosphere::Lithosphere(Domain&D)
: _order(D._order),_Ny(D._Ny),_Nz(D._Nz),
  _Ly(D._Ly),_Lz(D._Lz),_dy(_Ly/(_Ny-1.)),_dz(_Lz/(_Nz-1.)),
  _problemType(D._problemType),_outputDir(D._outputDir),
  _v0(D._v0),_vp(D._vp),
  _muArrPlus(D._muArrPlus),_muPlus(D._muPlus),
  _bcRplusShift(NULL),_surfDispPlus(NULL),
  _rhsPlus(NULL),_uhatPlus(NULL),_sigma_xyPlus(NULL),
  _linSolver(D._linSolver),_kspPlus(NULL),_pcPlus(NULL),
  _kspTol(D._kspTol),_sbpPlus(D,*D._muArrPlus,D._muPlus),
  _timeIntegrator(D._timeIntegrator),
  _strideLength(D._strideLength),_maxStepCount(D._maxStepCount),
  _initTime(D._initTime),_currTime(_initTime),_maxTime(D._maxTime),
  _minDeltaT(D._minDeltaT),_maxDeltaT(D._maxDeltaT),
  _stepCount(0),_atol(D._atol),_initDeltaT(D._initDeltaT),
  _timeViewer(NULL),_surfDispPlusViewer(NULL),
  _integrateTime(0),_writeTime(0),_linSolveTime(0),_factorTime(0),_linSolveCount(0),
  _bcTplus(NULL),_bcRplus(NULL),_bcBplus(NULL),_bcFplus(NULL),
  _fault(NULL)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting Lithosphere::Lithosphere in lithosphere.cpp.\n");
#endif

  // boundary conditions
  VecCreate(PETSC_COMM_WORLD,&_bcFplus);
  VecSetSizes(_bcFplus,PETSC_DECIDE,_Nz);
  VecSetFromOptions(_bcFplus);     PetscObjectSetName((PetscObject) _bcFplus, "_bcFplus");
  VecSet(_bcFplus,0.0);

  VecDuplicate(_bcFplus,&_bcRplusShift); PetscObjectSetName((PetscObject) _bcRplusShift, "bcRplusShift");
  VecDuplicate(_bcFplus,&_bcRplus); PetscObjectSetName((PetscObject) _bcRplus, "bcRplus");
  VecSet(_bcRplus,_vp*_initTime/2.0);

  VecCreate(PETSC_COMM_WORLD,&_bcTplus);
  VecSetSizes(_bcTplus,PETSC_DECIDE,_Ny);
  VecSetFromOptions(_bcTplus);     PetscObjectSetName((PetscObject) _bcTplus, "_bcTplus");
  VecSet(_bcTplus,0.0);

  VecDuplicate(_bcTplus,&_bcBplus); PetscObjectSetName((PetscObject) _bcBplus, "_bcBplus");
  VecSet(_bcBplus,0.0);

  KSPCreate(PETSC_COMM_WORLD,&_kspPlus);
  setupKSP(_sbpPlus,_kspPlus,_pcPlus);

  VecCreate(PETSC_COMM_WORLD,&_rhsPlus);
  VecSetSizes(_rhsPlus,PETSC_DECIDE,_Ny*_Nz);
  VecSetFromOptions(_rhsPlus);



  VecDuplicate(_bcTplus,&_surfDispPlus); PetscObjectSetName((PetscObject) _surfDispPlus, "_surfDispPlus");

  if (_timeIntegrator.compare("FEuler")==0) {
    _quadrature = new FEuler(_maxStepCount,_maxTime,_initDeltaT,D._timeControlType);
  }
  else if (_timeIntegrator.compare("RK32")==0) {
    _quadrature = new RK32(_maxStepCount,_maxTime,_initDeltaT,D._timeControlType);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: timeIntegrator type type not understood\n");
    assert(0>1); // automatically fail
  }

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending Lithosphere::Lithosphere in lithosphere.cpp.\n");
#endif
}


Lithosphere::~Lithosphere()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting Lithosphere::~Lithosphere in lithosphere.cpp.\n");
#endif

  // boundary conditions
  VecDestroy(&_bcFplus);
  VecDestroy(&_bcRplus);
  VecDestroy(&_bcTplus);
  VecDestroy(&_bcBplus);

  // body fields
  VecDestroy(&_rhsPlus);
  VecDestroy(&_uhatPlus);
  VecDestroy(&_sigma_xyPlus);
  VecDestroy(&_surfDispPlus);

  KSPDestroy(&_kspPlus);

  PetscViewerDestroy(&_timeViewer);
  PetscViewerDestroy(&_surfDispPlusViewer);


#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending Lithosphere::~Lithosphere in lithosphere.cpp.\n");
#endif
}


/*
 * Set up the Krylov Subspace and Preconditioner (KSP) environment. A
 * table of options available through PETSc and linked external packages
 * is available at
 * http://www.mcs.anl.gov/petsc/documentation/linearsolvertable.html.
 *
 * The methods implemented here are:
 *     Algorithm             Package           input file syntax
 * algebraic multigrid       HYPRE                AMG
 * direct LU                 MUMPS                MUMPSLU
 * direct Cholesky           MUMPS                MUMPSCHOLESKY         !!! TESTING NOW
 *
 * A list of options for each algorithm that can be set can be optained
 * by running the code with the argument main <input file> -help and
 * searching through the output for "Preconditioner (PC) options" and
 * "Krylov Method (KSP) options".
 */
PetscErrorCode Lithosphere::setupKSP(SbpOps& sbp,KSP& ksp,PC& pc)
{
  PetscErrorCode ierr = 0;

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting Lithosphere::setupKSP in lithosphere.cpp\n");CHKERRQ(ierr);
#endif

  //~ierr = KSPSetType(_ksp,KSPGMRES);CHKERRQ(ierr);
  //~ierr = KSPSetOperators(_ksp,_A,_A,SAME_PRECONDITIONER);CHKERRQ(ierr);
  //~ierr = KSPGetPC(_ksp,&_pc);CHKERRQ(ierr);


  // use PETSc's direct LU - only available on 1 processor!!!
  //~ierr = PCSetType(D.pc,PCLU);CHKERRQ(ierr);

  if (_linSolver.compare("AMG")==0) {
    // use HYPRE
    ierr = KSPSetType(ksp,KSPRICHARDSON);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,sbp._A,sbp._A,SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCHYPRE);CHKERRQ(ierr);
    ierr = PCHYPRESetType(pc,"boomeramg");CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,_kspTol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    ierr = PCFactorSetLevels(pc,4);CHKERRQ(ierr);
  }
//~
  else if (_linSolver.compare("MUMPSLU")==0) {
    // use direct LU from MUMPS
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,sbp._A,sbp._A,SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    PCSetType(pc,PCLU);
    PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);
    PCFactorSetUpMatSolverPackage(pc);
  }

  else if (_linSolver.compare("MUMPSCHOLESKY")==0) {
    // use direct LL^T (Cholesky factorization) from MUMPS
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,sbp._A,sbp._A,SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    PCSetType(pc,PCCHOLESKY);
    PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);
    PCFactorSetUpMatSolverPackage(pc);
  }
  else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"ERROR: linSolver type not understood\n");
    assert(0>1);
  }

  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending Lithosphere::setupKSP in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode Lithosphere::integrate()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting Lithosphere::integrate in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  double startTime = MPI_Wtime();

  // call odeSolver routine integrate here
  _quadrature->setTolerance(_atol);CHKERRQ(ierr);
  _quadrature->setTimeStepBounds(_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
  ierr = _quadrature->setTimeRange(_initTime,_maxTime);
  ierr = _quadrature->setInitialConds(_fault->_var);CHKERRQ(ierr);

  ierr = _quadrature->integrate(this);CHKERRQ(ierr);
  _integrateTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending Lithosphere::integrate in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}




PetscErrorCode Lithosphere::timeMonitor(const PetscReal time,const PetscInt stepCount,
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

PetscErrorCode Lithosphere::view()
{
  PetscErrorCode ierr = 0;
  ierr = _quadrature->view();
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in integration (s): %g\n",_integrateTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent setting up linear solve context (e.g. factoring) (s): %g\n",_factorTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of times linear system was solved: %i\n",_linSolveCount);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent solving linear system (s): %g\n",_linSolveTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);

  //~ierr = KSPView(_kspPlus,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  return ierr;
}











//================= Symmetric Lithosphere Functions ========================

SymmLithosphere::SymmLithosphere(Domain&D)
: Lithosphere(D)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting SymmLithosphere::SymmLithosphere in lithosphere.cpp.\n");
#endif

  _fault = new SymmFault(D);

  // almost everything is covered by base class' constructor, except the
  // construction of _fault, and populating bcRshift

  setShifts(); // set _bcRplusShift
  VecAXPY(_bcRplus,1.0,_bcRplusShift);


  _sbpPlus.setRhs(_rhsPlus,_bcFplus,_bcRplus,_bcTplus,_bcBplus);

  VecDuplicate(_rhsPlus,&_uhatPlus);
  double startTime = MPI_Wtime();
  KSPSolve(_kspPlus,_rhsPlus,_uhatPlus);
  _factorTime += MPI_Wtime() - startTime;

  VecDuplicate(_rhsPlus,&_sigma_xyPlus);
  MatMult(_sbpPlus._Dy_Iz,_uhatPlus,_sigma_xyPlus);


  _fault->setTauQS(_sigma_xyPlus,NULL);
  _fault->setFaultDisp(_bcFplus,NULL);
  _fault->computeVel();

  setSurfDisp();

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending SymmLithosphere::SymmLithosphere in lithosphere.cpp.\n");
#endif
}

SymmLithosphere::~SymmLithosphere(){};



PetscErrorCode SymmLithosphere::computeShearStress()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmLithosphere::computeShearStress in lithosphere.cpp.\n");CHKERRQ(ierr);
#endif

  ierr = MatMult(_sbpPlus._Dy_Iz,_uhatPlus,_sigma_xyPlus);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmLithosphere::computeShearStress in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}

// destructor is covered by base class



PetscErrorCode SymmLithosphere::setShifts()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmLithosphere::setShifts in lithosphere.cpp\n");CHKERRQ(ierr);
#endif


  PetscInt Ii,Istart,Iend;
  PetscScalar v,bcRshift;
  ierr = VecGetOwnershipRange(_bcRplusShift,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    v = _fault->getTauInf(Ii);
    //~bcRshift = v*_Ly/_muArrPlus[_Ny*_Nz-_Nz+Ii]; // use last values of muArr
    bcRshift = v*_Ly/_muArrPlus[Ii]; // use first values of muArr
    //~bcRshift = 0.;
    ierr = VecSetValue(_bcRplusShift,Ii,bcRshift,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_bcRplusShift);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcRplusShift);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmLithosphere::setShifts in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}




PetscErrorCode SymmLithosphere::setSurfDisp()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmLithosphere::setSurfDisp in lithosphere.cpp\n");CHKERRQ(ierr);
#endif

  PetscInt    Ii,Istart,Iend;
  PetscScalar u,y,z;
  ierr = VecGetOwnershipRange(_uhatPlus,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    z = Ii-_Nz*(Ii/_Nz);
    y = Ii/_Nz;
    if (z == 0) {
      ierr = VecGetValues(_uhatPlus,1,&Ii,&u);CHKERRQ(ierr);
      ierr = VecSetValue(_surfDispPlus,y,u,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_surfDispPlus);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_surfDispPlus);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmLithosphere::setSurfDisp in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}



PetscErrorCode SymmLithosphere::writeStep()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmLithosphere::writeStep in lithosphere.cpp at step %i\n",_stepCount);CHKERRQ(ierr);
#endif
  double startTime = MPI_Wtime();

  if (_stepCount==0) {
    ierr = _sbpPlus.writeOps(_outputDir);CHKERRQ(ierr);
    ierr = _fault->writeContext(_outputDir);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(_outputDir+"time.txt").c_str(),&_timeViewer);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDispPlus").c_str(),FILE_MODE_WRITE,
                                 &_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_surfDispPlus,_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDispPlus").c_str(),
                                   FILE_MODE_APPEND,&_surfDispPlusViewer);CHKERRQ(ierr);

    //~// boundary conditions
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcRplus").c_str(),FILE_MODE_WRITE,
                                 &_bcRplusV);CHKERRQ(ierr);
    ierr = VecView(_bcRplus,_bcRplusV);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_bcRplusV);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcRplus").c_str(),
                                   FILE_MODE_APPEND,&_bcRplusV);CHKERRQ(ierr);
//~
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcRplusShift").c_str(),FILE_MODE_WRITE,
                                 //~&_bcRplusShiftV);CHKERRQ(ierr);
    //~ierr = VecView(_bcRplusShift,_bcRplusShiftV);CHKERRQ(ierr);
    //~ierr = PetscViewerDestroy(&_bcRplusShiftV);CHKERRQ(ierr);
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcRplusShift").c_str(),
                                   //~FILE_MODE_APPEND,&_bcRplusShiftV);CHKERRQ(ierr);
//~
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcFplus").c_str(),FILE_MODE_WRITE,
                                 //~&_bcFplusV);CHKERRQ(ierr);
    //~ierr = VecView(_bcFplus,_bcFplusV);CHKERRQ(ierr);
    //~ierr = PetscViewerDestroy(&_bcFplusV);CHKERRQ(ierr);
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcFplus").c_str(),
                                   //~FILE_MODE_APPEND,&_bcFplusV);CHKERRQ(ierr);
//~
    //~// body fields
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"uBodyPlus").c_str(),FILE_MODE_WRITE,
                                 //~&_uPlusV);CHKERRQ(ierr);
    //~ierr = VecView(_uhatPlus,_uPlusV);CHKERRQ(ierr);
    //~ierr = PetscViewerDestroy(&_uPlusV);CHKERRQ(ierr);
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"uBodyPlus").c_str(),
                                   //~FILE_MODE_APPEND,&_uPlusV);CHKERRQ(ierr);
//~
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"sigmaXYplus").c_str(),FILE_MODE_WRITE,
                                 //~&_sigma_xyPlusV);CHKERRQ(ierr);
    //~ierr = VecView(_sigma_xyPlus,_sigma_xyPlusV);CHKERRQ(ierr);
    //~ierr = PetscViewerDestroy(&_sigma_xyPlusV);CHKERRQ(ierr);
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"sigmaXYplus").c_str(),
                                   //~FILE_MODE_APPEND,&_sigma_xyPlusV);CHKERRQ(ierr);
  }
  else {
    ierr = VecView(_surfDispPlus,_surfDispPlusViewer);CHKERRQ(ierr);

    ierr = VecView(_bcRplus,_bcRplusV);CHKERRQ(ierr);
    //~ierr = VecView(_bcRplusShift,_bcRplusShiftV);CHKERRQ(ierr);
    //~ierr = VecView(_bcFplus,_bcFplusV);CHKERRQ(ierr);

    //~ierr = VecView(_uhatPlus,_uPlusV);CHKERRQ(ierr);
    //~ierr = VecView(_sigma_xyPlus,_sigma_xyPlusV);CHKERRQ(ierr);
  }
  ierr = _fault->writeStep(_outputDir,_stepCount);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(_timeViewer, "%.15e\n",_currTime);CHKERRQ(ierr);

  _writeTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmLithosphere::writeStep in lithosphere.cpp at step %i\n",_stepCount);CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode SymmLithosphere::d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                 it_vec dvarBegin,it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SymmLithosphere::d_dt in lithosphere.cpp: time=%.15e\n",time);CHKERRQ(ierr);
#endif

  // update boundaries
  ierr = VecCopy(*(varBegin+1),_bcFplus);CHKERRQ(ierr);
  ierr = VecScale(_bcFplus,0.5);CHKERRQ(ierr);
  ierr = VecSet(_bcRplus,_vp*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_bcRplus,1.0,_bcRplusShift);CHKERRQ(ierr);

  // solve for displacement
  ierr = _sbpPlus.setRhs(_rhsPlus,_bcFplus,_bcRplus,_bcTplus,_bcBplus);CHKERRQ(ierr);
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspPlus,_rhsPlus,_uhatPlus);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  // solve for shear stress
  ierr = MatMult(_sbpPlus._Dy_Iz,_uhatPlus,_sigma_xyPlus);CHKERRQ(ierr);
  ierr = _fault->setTauQS(_sigma_xyPlus,NULL);CHKERRQ(ierr);
  ierr = _fault->d_dt(varBegin,varEnd, dvarBegin, dvarEnd);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending SymmLithosphere::d_dt in lithosphere.cpp: time=%.15e\n",time);CHKERRQ(ierr);
#endif
  return ierr;
}



// Outputs data at each time step.
PetscErrorCode SymmLithosphere::debug(const PetscReal time,const PetscInt stepCount,
                     const_it_vec varBegin,const_it_vec varEnd,
                     const_it_vec dvarBegin,const_it_vec dvarEnd,const char *stage)
{
  PetscErrorCode ierr = 0;

#if ODEPRINT > 0
  PetscInt       Istart,Iend;
  PetscScalar    gRval,uVal,psiVal,velVal,dQVal;

  //~PetscScalar k = _muArrPlus[0]/2/_Ly;

  ierr= VecGetOwnershipRange(*varBegin,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(*varBegin,1,&Istart,&psiVal);CHKERRQ(ierr);

  ierr = VecGetValues(*(varBegin+1),1,&Istart,&uVal);CHKERRQ(ierr);

  ierr= VecGetOwnershipRange(*dvarBegin,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(*dvarBegin,1,&Istart,&dQVal);CHKERRQ(ierr);
  ierr = VecGetValues(*(dvarBegin+1),1,&Istart,&velVal);CHKERRQ(ierr);

  ierr= VecGetOwnershipRange(_bcRplus,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(_bcRplus,1,&Istart,&gRval);CHKERRQ(ierr);

  if (stepCount == 0) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%-4s %-6s | %-15s %-15s %-15s | %-15s %-15s %-16s | %-15s\n",
                       "Step","Stage","gR","D","Q","VL","V","dQ","time");
    CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%4i %-6s ",stepCount,stage);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," | %.9e %.9e %.9e ",gRval,uVal,psiVal);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," | %.9e %.9e %.9e ",_vp/2.,velVal,dQVal);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," | %.9e\n",time);CHKERRQ(ierr);
#endif
  return ierr;
}












//================= Full Lithosphere (+ and - sides) Functions =========

FullLithosphere::FullLithosphere(Domain&D)
: Lithosphere(D),
  _muArrMinus(D._muArrMinus),_muMinus(D._muMinus),
  _bcRminusShift(NULL),_surfDispMinus(NULL),
  _rhsMinus(NULL),_uhatMinus(NULL),_sigma_xyMinus(NULL),
  _surfDispMinusViewer(NULL),
  _kspMinus(NULL),_pcMinus(NULL),
  _sbpMinus(D,*D._muArrMinus,D._muMinus),
  _bcTminus(NULL),_bcRminus(NULL),_bcBminus(NULL),_bcFminus(NULL)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting FullLithosphere::FullLithosphere in lithosphere.cpp.\n");
#endif

  // sign convention resulting from the fact that I'm indexing from y=0 to y=-Ly
  MatScale(_sbpMinus._Dy_Iz,-1.0);
  _fault = new FullFault(D);

  VecDuplicate(_bcFplus,&_bcRminusShift); PetscObjectSetName((PetscObject) _bcRminusShift, "_bcRminusShift");
  setShifts(); // set position of boundary from steady sliding
  VecAXPY(_bcRplus,1.0,_bcRplusShift);

  // fault initial displacement on minus side
  VecDuplicate(_bcFplus,&_bcFminus); PetscObjectSetName((PetscObject) _bcFminus, "_bcFminus");
  VecSet(_bcFminus,0.0);

  // remote displacement on - side
  VecDuplicate(_bcFplus,&_bcRminus); PetscObjectSetName((PetscObject) _bcRminus, "bcRminus");
  VecSet(_bcRminus,-_vp*_initTime/2.0);
  VecAXPY(_bcRminus,1.0,_bcRminusShift);

  VecDuplicate(_bcTplus,&_bcTminus); PetscObjectSetName((PetscObject) _bcTminus, "bcTminus");
  VecSet(_bcTminus,0.0);
  VecDuplicate(_bcBplus,&_bcBminus); PetscObjectSetName((PetscObject) _bcBminus, "bcBminus");
  VecSet(_bcBminus,0.0);


  _sbpPlus.setRhs(_rhsPlus,_bcFplus,_bcRplus,_bcTplus,_bcBplus);

  VecDuplicate(_rhsPlus,&_uhatPlus);
  double startTime = MPI_Wtime();
  KSPSolve(_kspPlus,_rhsPlus,_uhatPlus);
  _factorTime += MPI_Wtime() - startTime;

  VecDuplicate(_rhsPlus,&_sigma_xyPlus);
  MatMult(_sbpPlus._Dy_Iz,_uhatPlus,_sigma_xyPlus);


  KSPCreate(PETSC_COMM_WORLD,&_kspMinus);
  startTime = MPI_Wtime();
  setupKSP(_sbpMinus,_kspMinus,_pcMinus);
  _factorTime += MPI_Wtime() - startTime;

  VecCreate(PETSC_COMM_WORLD,&_rhsMinus);
  VecSetSizes(_rhsMinus,PETSC_DECIDE,_Ny*_Nz);
  VecSetFromOptions(_rhsMinus);
  _sbpMinus.setRhs(_rhsMinus,_bcFminus,_bcRminus,_bcTminus,_bcBminus);


  VecDuplicate(_rhsMinus,&_uhatMinus);
  KSPSolve(_kspMinus,_rhsMinus,_uhatMinus);
  VecDuplicate(_rhsPlus,&_sigma_xyMinus);
  MatMult(_sbpMinus._Dy_Iz,_uhatMinus,_sigma_xyMinus);

  _fault->setTauQS(_sigma_xyPlus,_sigma_xyMinus);
  _fault->setFaultDisp(_bcFplus,_bcFminus);
  _fault->computeVel();
  VecDuplicate(_bcTminus,&_surfDispMinus); PetscObjectSetName((PetscObject) _surfDispMinus, "_surfDispMinus");
  setSurfDisp();

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending FullLithosphere::FullLithosphere in lithosphere.cpp.\n");
#endif
}

FullLithosphere::~FullLithosphere()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting FullLithosphere::~FullLithosphere in lithosphere.cpp.\n");
#endif

  // boundary conditions: minus side
  VecDestroy(&_bcFminus);
  VecDestroy(&_bcRminusShift);
  VecDestroy(&_bcRminus);
  VecDestroy(&_bcTminus);
  VecDestroy(&_bcBminus);

  // body fields
  VecDestroy(&_rhsMinus);
  VecDestroy(&_uhatMinus);
  VecDestroy(&_sigma_xyMinus);

  VecDestroy(&_surfDispMinus);

  KSPDestroy(&_kspMinus);


  PetscViewerDestroy(&_surfDispMinusViewer);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending FullLithosphere::~FullLithosphere in lithosphere.cpp.\n");
#endif
}


//===================== private member functions =======================


PetscErrorCode FullLithosphere::computeShearStress()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullLithosphere::computeShearStress in lithosphere.cpp.\n");CHKERRQ(ierr);
#endif

  ierr = MatMult(_sbpPlus._Dy_Iz,_uhatPlus,_sigma_xyPlus);CHKERRQ(ierr);
  ierr = MatMult(_sbpMinus._Dy_Iz,_uhatMinus,_sigma_xyMinus);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullLithosphere::computeShearStress in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


/* Set displacement at sides equal to steady-sliding values:
 *   u ~ tau_fric*L/mu
 */
PetscErrorCode FullLithosphere::setShifts()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullLithosphere::setShifts in lithosphere.cpp\n");CHKERRQ(ierr);
#endif


  PetscInt Ii,Istart,Iend;
  PetscScalar v,bcRshift;
  ierr = VecGetOwnershipRange(_bcRplusShift,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    v = _fault->getTauInf(Ii);
    bcRshift = v*_Ly/_muArrPlus[_Ny*_Nz-_Nz+Ii]; // use last values of muArr

    ierr = VecSetValue(_bcRplusShift,Ii,bcRshift,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_bcRplusShift);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcRplusShift);CHKERRQ(ierr);


  ierr = VecGetOwnershipRange(_bcRminusShift,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart;Ii<Iend;Ii++) {
    v = _fault->getTauInf(Ii);
    bcRshift = -v*_Ly/_muArrMinus[_Ny*_Nz-_Nz+Ii]; // use last values of muArr

    ierr = VecSetValue(_bcRminusShift,Ii,bcRshift,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(_bcRminusShift);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_bcRminusShift);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullLithosphere::setShifts in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode FullLithosphere::setSurfDisp()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullLithosphere::setSurfDisp in lithosphere.cpp\n");CHKERRQ(ierr);
#endif

  PetscInt    Ii,Istart,Iend;
  PetscScalar u,y,z;
  ierr = VecGetOwnershipRange(_uhatPlus,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    z = Ii-_Nz*(Ii/_Nz);
    y = Ii/_Nz;
    if (z == 0) {
      ierr = VecGetValues(_uhatPlus,1,&Ii,&u);CHKERRQ(ierr);
      ierr = VecSetValue(_surfDispPlus,y,u,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_surfDispPlus);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_surfDispPlus);CHKERRQ(ierr);


  ierr = VecGetOwnershipRange(_uhatMinus,&Istart,&Iend);
  for (Ii=Istart;Ii<Iend;Ii++) {
    z = Ii-_Nz*(Ii/_Nz);
    y = Ii/_Nz;
    if (z == 0) {
      ierr = VecGetValues(_uhatMinus,1,&Ii,&u);CHKERRQ(ierr);
      ierr = VecSetValue(_surfDispMinus,y,u,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(_surfDispMinus);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(_surfDispMinus);CHKERRQ(ierr);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullLithosphere::setSurfDisp in lithosphere.cpp\n");CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode FullLithosphere::writeStep()
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullLithosphere::writeStep in lithosphere.cpp at step %i\n",_stepCount);CHKERRQ(ierr);
#endif
  double startTime = MPI_Wtime();

  if (_stepCount==0) {
    ierr = _sbpPlus.writeOps(_outputDir+"plus_");CHKERRQ(ierr);
    if (_problemType.compare("full")==0) { ierr = _sbpMinus.writeOps(_outputDir+"minus_");CHKERRQ(ierr); }
    ierr = _fault->writeContext(_outputDir);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(_outputDir+"time.txt").c_str(),&_timeViewer);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDispPlus").c_str(),FILE_MODE_WRITE,
                                 &_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_surfDispPlus,_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDispPlus").c_str(),
                                   FILE_MODE_APPEND,&_surfDispPlusViewer);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDispMinus").c_str(),FILE_MODE_WRITE,
                                 &_surfDispMinusViewer);CHKERRQ(ierr);
    ierr = VecView(_surfDispMinus,_surfDispMinusViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&_surfDispMinusViewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"surfDispMinus").c_str(),
                                   FILE_MODE_APPEND,&_surfDispMinusViewer);CHKERRQ(ierr);


    //~// boundary conditions
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcRplus").c_str(),FILE_MODE_WRITE,
                                 //~&_bcRplusV);CHKERRQ(ierr);
    //~ierr = VecView(_bcRplus,_bcRplusV);CHKERRQ(ierr);
    //~ierr = PetscViewerDestroy(&_bcRplusV);CHKERRQ(ierr);
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcRplus").c_str(),
                                   //~FILE_MODE_APPEND,&_bcRplusV);CHKERRQ(ierr);
//~
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcRplusShift").c_str(),FILE_MODE_WRITE,
                                 //~&_bcRplusShiftV);CHKERRQ(ierr);
    //~ierr = VecView(_bcRplusShift,_bcRplusShiftV);CHKERRQ(ierr);
    //~ierr = PetscViewerDestroy(&_bcRplusShiftV);CHKERRQ(ierr);
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcRplusShift").c_str(),
                                   //~FILE_MODE_APPEND,&_bcRplusShiftV);CHKERRQ(ierr);
//~
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcRminus").c_str(),FILE_MODE_WRITE,
                                 //~&_bcRminusV);CHKERRQ(ierr);
    //~ierr = VecView(_bcRminus,_bcRminusV);CHKERRQ(ierr);
    //~ierr = PetscViewerDestroy(&_bcRminusV);CHKERRQ(ierr);
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcRminus").c_str(),
                                   //~FILE_MODE_APPEND,&_bcRminusV);CHKERRQ(ierr);
//~
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcRminusShift").c_str(),FILE_MODE_WRITE,
                                 //~&_bcRminusShiftV);CHKERRQ(ierr);
    //~ierr = VecView(_bcRminusShift,_bcRminusShiftV);CHKERRQ(ierr);
    //~ierr = PetscViewerDestroy(&_bcRminusShiftV);CHKERRQ(ierr);
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcRminusShift").c_str(),
                                   //~FILE_MODE_APPEND,&_bcRminusShiftV);CHKERRQ(ierr);
//~
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcFplus").c_str(),FILE_MODE_WRITE,
                                 //~&_bcFplusV);CHKERRQ(ierr);
    //~ierr = VecView(_bcFplus,_bcFplusV);CHKERRQ(ierr);
    //~ierr = PetscViewerDestroy(&_bcFplusV);CHKERRQ(ierr);
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcFplus").c_str(),
                                   //~FILE_MODE_APPEND,&_bcFplusV);CHKERRQ(ierr);
//~
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcFminus").c_str(),FILE_MODE_WRITE,
                                 //~&_bcFminusV);CHKERRQ(ierr);
    //~ierr = VecView(_bcFminus,_bcFminusV);CHKERRQ(ierr);
    //~ierr = PetscViewerDestroy(&_bcFminusV);CHKERRQ(ierr);
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"bcFminus").c_str(),
                                   //~FILE_MODE_APPEND,&_bcFminusV);CHKERRQ(ierr);
//~
    //~// body fields
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"uBodyPlus").c_str(),FILE_MODE_WRITE,
                                 //~&_uPlusV);CHKERRQ(ierr);
    //~ierr = VecView(_uhatPlus,_uPlusV);CHKERRQ(ierr);
    //~ierr = PetscViewerDestroy(&_uPlusV);CHKERRQ(ierr);
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"uBodyPlus").c_str(),
                                   //~FILE_MODE_APPEND,&_uPlusV);CHKERRQ(ierr);
//~
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"sigmaXYplus").c_str(),FILE_MODE_WRITE,
                                 //~&_sigma_xyPlusV);CHKERRQ(ierr);
    //~ierr = VecView(_sigma_xyPlus,_sigma_xyPlusV);CHKERRQ(ierr);
    //~ierr = PetscViewerDestroy(&_sigma_xyPlusV);CHKERRQ(ierr);
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"sigmaXYplus").c_str(),
                                   //~FILE_MODE_APPEND,&_sigma_xyPlusV);CHKERRQ(ierr);
//~
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"uBodyMinus").c_str(),FILE_MODE_WRITE,
                                 //~&_uMinusV);CHKERRQ(ierr);
    //~ierr = VecView(_uhatMinus,_uMinusV);CHKERRQ(ierr);
    //~ierr = PetscViewerDestroy(&_uMinusV);CHKERRQ(ierr);
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"uBodyMinus").c_str(),
                                   //~FILE_MODE_APPEND,&_uMinusV);CHKERRQ(ierr);
//~
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"sigmaXYminus").c_str(),FILE_MODE_WRITE,
                                 //~&_sigma_xyMinusV);CHKERRQ(ierr);
    //~ierr = VecView(_sigma_xyMinus,_sigma_xyMinusV);CHKERRQ(ierr);
    //~ierr = PetscViewerDestroy(&_sigma_xyMinusV);CHKERRQ(ierr);
    //~ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(_outputDir+"sigmaXYminus").c_str(),
                                   //~FILE_MODE_APPEND,&_sigma_xyMinusV);CHKERRQ(ierr);
  }
  else {
    ierr = VecView(_surfDispPlus,_surfDispPlusViewer);CHKERRQ(ierr);
    ierr = VecView(_surfDispMinus,_surfDispMinusViewer);CHKERRQ(ierr);

    //~ierr = VecView(_bcRplus,_bcRplusV);CHKERRQ(ierr);
    //~ierr = VecView(_bcRplusShift,_bcRplusShiftV);CHKERRQ(ierr);
    //~ierr = VecView(_bcRminus,_bcRminusV);CHKERRQ(ierr);
    //~ierr = VecView(_bcRminusShift,_bcRminusShiftV);CHKERRQ(ierr);
    //~ierr = VecView(_bcFplus,_bcFplusV);CHKERRQ(ierr);
    //~ierr = VecView(_bcFminus,_bcFminusV);CHKERRQ(ierr);
//~
    //~// body fields
    //~ierr = VecView(_uhatPlus,_uPlusV);CHKERRQ(ierr);
    //~ierr = VecView(_uhatMinus,_uMinusV);CHKERRQ(ierr);
    //~ierr = VecView(_sigma_xyPlus,_sigma_xyPlusV);CHKERRQ(ierr);
    //~ierr = VecView(_sigma_xyMinus,_sigma_xyMinusV);CHKERRQ(ierr);
  }
  ierr = _fault->writeStep(_outputDir,_stepCount);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(_timeViewer, "%.15e\n",_currTime);CHKERRQ(ierr);

  _writeTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullLithosphere::writeStep in lithosphere.cpp at step %i\n",_stepCount);CHKERRQ(ierr);
#endif
  return ierr;
}


PetscErrorCode FullLithosphere::d_dt(const PetscScalar time,const_it_vec varBegin,const_it_vec varEnd,
                 it_vec dvarBegin,it_vec dvarEnd)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting FullLithosphere::d_dt in lithosphere.cpp: time=%.15e\n",time);CHKERRQ(ierr);
#endif

  // update boundaries: + side
  ierr = VecCopy(*(varBegin+1),_bcFplus);CHKERRQ(ierr);
  ierr = VecSet(_bcRplus,_vp*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_bcRplus,1.0,_bcRplusShift);CHKERRQ(ierr);

  // update boundaries: - side
  ierr = VecCopy(*(varBegin+2),_bcFminus);CHKERRQ(ierr);
  ierr = VecSet(_bcRminus,-_vp*time/2.0);CHKERRQ(ierr);
  ierr = VecAXPY(_bcRminus,1.0,_bcRminusShift);CHKERRQ(ierr);


   // solve for displacement: + side
  ierr = _sbpPlus.setRhs(_rhsPlus,_bcFplus,_bcRplus,_bcTplus,_bcBplus);CHKERRQ(ierr);
  double startTime = MPI_Wtime();
  ierr = KSPSolve(_kspPlus,_rhsPlus,_uhatPlus);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;
  ierr = setSurfDisp();

  // solve for displacement: - side
  ierr = _sbpMinus.setRhs(_rhsMinus,_bcFminus,_bcRminus,_bcTminus,_bcBminus);CHKERRQ(ierr);
  startTime = MPI_Wtime();
  ierr = KSPSolve(_kspMinus,_rhsMinus,_uhatMinus);CHKERRQ(ierr);
  _linSolveTime += MPI_Wtime() - startTime;
  _linSolveCount++;


  // solve for shear stress
  ierr = MatMult(_sbpMinus._Dy_Iz,_uhatMinus,_sigma_xyMinus);CHKERRQ(ierr);
  ierr = MatMult(_sbpPlus._Dy_Iz,_uhatPlus,_sigma_xyPlus);CHKERRQ(ierr);
  ierr = _fault->setTauQS(_sigma_xyPlus,_sigma_xyMinus);CHKERRQ(ierr);

  ierr = _fault->d_dt(varBegin,varEnd, dvarBegin, dvarEnd);

#if VERBOSE > 1
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending FullLithosphere::d_dt in lithosphere.cpp: time=%.15e\n",time);CHKERRQ(ierr);
#endif
  return ierr;
}



// Outputs data at each time step.
PetscErrorCode FullLithosphere::debug(const PetscReal time,const PetscInt stepCount,
                     const_it_vec varBegin,const_it_vec varEnd,
                     const_it_vec dvarBegin,const_it_vec dvarEnd,const char *stage)
{
  PetscErrorCode ierr = 0;
#if ODEPRINT > 0
  PetscInt       Istart,Iend;
  PetscScalar    gRvalPlus,gRvalMinus,uValMinus,uValPlus,psiVal,velValMinus,velValPlus,dQVal;

  //~PetscScalar k = _muArrPlus[0]/2/_Ly;

  ierr= VecGetOwnershipRange(*varBegin,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(*varBegin,1,&Istart,&psiVal);CHKERRQ(ierr);

  ierr = VecGetValues(*(varBegin+1),1,&Istart,&uValPlus);CHKERRQ(ierr);
  ierr = VecGetValues(*(varBegin+2),1,&Istart,&uValMinus);CHKERRQ(ierr);

  ierr= VecGetOwnershipRange(*dvarBegin,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(*dvarBegin,1,&Istart,&dQVal);CHKERRQ(ierr);
  ierr = VecGetValues(*(dvarBegin+1),1,&Istart,&velValPlus);CHKERRQ(ierr);
  ierr = VecGetValues(*(dvarBegin+2),1,&Istart,&velValMinus);CHKERRQ(ierr);

  ierr= VecGetOwnershipRange(_bcRplus,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetValues(_bcRplus,1,&Istart,&gRvalPlus);CHKERRQ(ierr);
  ierr = VecGetValues(_bcRminus,1,&Istart,&gRvalMinus);CHKERRQ(ierr);

  if (stepCount == 0) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%-4s|| %-4s %-6s | %-15s %-15s %-15s | %-15s %-15s %-16s | %-15s\n",
                       "Side","Step","Stage","gR","D","Q","VL","V","dQ","time");
    CHKERRQ(ierr);
  }
  // plus side
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%-4s|| %4i %-6s ","+",stepCount,stage);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"| %.9e %.9e %.9e",gRvalPlus,uValMinus,psiVal);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"| %.9e %.9e %.9e",_vp/2.,velValPlus,dQVal);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"| %.9e\n",time);CHKERRQ(ierr);

  // minus side
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"%4i %-6s ","-",stepCount,stage);CHKERRQ(ierr);
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"| %.9e %.9e %.9e ",gRvalMinus,uValMinus,psiVal);CHKERRQ(ierr);
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"| %.9e %.9e %.9e ",-_vp/2.,velValMinus,dQVal);CHKERRQ(ierr);
  //~ierr = PetscPrintf(PETSC_COMM_WORLD,"| %.9e\n",time);CHKERRQ(ierr);
#endif
  return ierr;
}

