#include "odeSolver_WaveEq.hpp"

using namespace std;

OdeSolver_WaveEq::OdeSolver_WaveEq(PetscInt maxNumSteps,PetscScalar initT,PetscScalar finalT,PetscScalar deltaT)
: _initT(initT),_finalT(finalT),_currT(initT),_deltaT(deltaT),
  _maxNumSteps(maxNumSteps),_stepCount(0),
  _lenVar(0),_runTime(0)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolver_WaveEq constructor in odeSolver_waveEq.cpp.\n");
#endif
  double startTime = MPI_Wtime();

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolver_WaveEq constructor in odeSolver_waveEq.cpp.\n");
#endif
}

OdeSolver_WaveEq::~OdeSolver_WaveEq()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK43::destructor in odeSolver.cpp.\n");
#endif

  // destruct temporary containers
  destroyVector(_varNext);
  destroyVector(_var);
  destroyVector(_varPrev);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending RK43::destructor in odeSolver.cpp.\n");
  #endif
}

PetscErrorCode OdeSolver_WaveEq::setStepSize(const PetscReal deltaT) { _deltaT = deltaT;  return 0; }

// if starting with a nonzero initial step count
PetscErrorCode OdeSolver_WaveEq::setInitialStepCount(const PetscReal stepCount)
{
  _stepCount = stepCount;
  return 0;
}

PetscErrorCode OdeSolver_WaveEq::setTimeRange(const PetscReal initT,const PetscReal finalT)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolver_WaveEq::setTimeRange in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();

  _initT = initT;
  _currT = initT;
  _finalT = finalT;

  _runTime += MPI_Wtime() - startTime;
  return 0;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolver_WaveEq::setTimeRange in odeSolver.cpp.\n");
#endif
}


PetscErrorCode OdeSolver_WaveEq::view()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolver_WaveEq::view in odeSolver_WaveEq.cpp.\n");
#endif
  PetscErrorCode ierr = 0;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTimeSolver summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   integration algorithm: wave equation\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time interval: %g to %g\n",
                     _initT,_finalT);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   total number of steps taken: %i/%i\n",
                     _stepCount,_maxNumSteps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   final time reached: %g\n",
                     _currT);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   total run time: %g\n",
                     _runTime);CHKERRQ(ierr);
  return 0;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending WaveEq::view in odeSolver_waveEq.cpp.\n");
#endif
}



PetscErrorCode OdeSolver_WaveEq::setInitialConds(std::map<string,Vec>& var)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting WaveEq::setInitialConds in odeSolver_waveEq.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  PetscErrorCode ierr = 0;

  for (map<string,Vec>::iterator it = var.begin(); it != var.end(); it++ ) {

    // allocate n: var
    VecDuplicate(var[it->first],&_var[it->first]); VecCopy(var[it->first],_var[it->first]);

    // allocate n-1: varPrev
    VecDuplicate(var[it->first],&_varPrev[it->first]); VecCopy(var[it->first],_varPrev[it->first]);

    // allocate n+1: varNext
    VecDuplicate(var[it->first],&_varNext[it->first]); VecSet(_varNext[it->first],0.);
  }

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending WaveEq::setInitialConds in odeSolver_waveEq.cpp.\n");
#endif
  return ierr;
}

PetscErrorCode OdeSolver_WaveEq::setInitialConds(std::map<string,Vec>& var,std::map<string,Vec>& varPrev)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting WaveEq::setInitialConds in odeSolver_waveEq.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  PetscErrorCode ierr = 0;

  for (map<string,Vec>::iterator it = var.begin(); it != var.end(); it++ ) {

    // allocate n: var
    VecDuplicate(var[it->first],&_var[it->first]); VecCopy(var[it->first],_var[it->first]);

    // allocate n-1: varPrev
    VecDuplicate(varPrev[it->first],&_varPrev[it->first]); VecCopy(varPrev[it->first],_varPrev[it->first]);

    // allocate n+1: varNext
    VecDuplicate(var[it->first],&_varNext[it->first]); VecSet(_varNext[it->first],0.);
  }

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending WaveEq::setInitialConds in odeSolver_waveEq.cpp.\n");
#endif
  return ierr;
}



PetscErrorCode OdeSolver_WaveEq::integrate(IntegratorContext_WaveEq *obj)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolver_WaveEq::integrate in odeSolver_waveEq.cpp.\n");
#endif
  PetscErrorCode ierr = 0;
  double startTime = MPI_Wtime();
  int   stopIntegration = 0;

  if (_finalT==_initT) { return ierr; }
  else if (_deltaT==0) { _deltaT = (_finalT-_initT)/_maxNumSteps; }

  // write initial condition
  ierr = obj->timeMonitor(_currT,_deltaT,_stepCount,stopIntegration);CHKERRQ(ierr); // write first step

  while (_stepCount<_maxNumSteps && _currT<_finalT) {
    _currT = _currT + _deltaT;
    if (_currT>_finalT) { _currT = _finalT; }
    _stepCount++;
    ierr = obj->d_dt(_currT,_deltaT,_varNext,_var,_varPrev);CHKERRQ(ierr);

    // accept time step and update
    for (map<string,Vec>::iterator it = _var.begin(); it != _var.end(); it++ ) {
      VecCopy(_var[it->first],_varPrev[it->first]);
      VecCopy(_varNext[it->first],_var[it->first]);
      VecSet(_varNext[it->first],0.0);
    }

    ierr = obj->timeMonitor(_currT,_deltaT,_stepCount,stopIntegration);CHKERRQ(ierr);
    if (stopIntegration > 0) { PetscPrintf(PETSC_COMM_WORLD,"OdeSolver WaveEq: Detected stop time integration request.\n"); break; }


  }

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending FEuler::integrate in odeSolver_waveEq.cpp.\n");
#endif
  return ierr;
}

PetscErrorCode OdeSolver_WaveEq::loadCheckpoint(const std::string inputDir)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolver_WaveEq::loadCheckpoint in odeSolver.cpp.\n");
  #endif
  PetscErrorCode ierr;

  string fileName = inputDir + "checkpoint.h5";

  // load saved checkpoint data
  PetscViewer viewer;

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/odeSolver_WaveEq");                                                  CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "odeSolver_WaveEq_chkpt_data", "stepCount", PETSC_INT, NULL, &_stepCount); CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "odeSolver_WaveEq_chkpt_data", "deltaT", PETSC_SCALAR, NULL, &_deltaT); CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                                                              CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/odeSolver_WaveEq/varNext"); CHKERRQ(ierr);
  for (map<string,Vec>::iterator it = _varNext.begin(); it!=_varNext.end(); it++ ) {
    ierr = PetscObjectSetName((PetscObject) _varNext[it->first], (it->first).c_str()); CHKERRQ(ierr);
    ierr = VecLoad(_varNext[it->first], viewer);                          CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(viewer); CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/odeSolver_WaveEq/var"); CHKERRQ(ierr);
  for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
    ierr = PetscObjectSetName((PetscObject) _var[it->first], (it->first).c_str()); CHKERRQ(ierr);
    ierr = VecLoad(_var[it->first], viewer);                          CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PushGroup(viewer, "/odeSolver_WaveEq/varPrev"); CHKERRQ(ierr);
  for (map<string,Vec>::iterator it = _varPrev.begin(); it!=_varPrev.end(); it++ ) {
    ierr = PetscObjectSetName((PetscObject) _varPrev[it->first], (it->first).c_str()); CHKERRQ(ierr);
    ierr = VecLoad(_varPrev[it->first], viewer);                          CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(viewer); CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolver_WaveEq::loadCheckpoint in odeSolver.cpp.\n");
  #endif

  return ierr;
}

PetscErrorCode OdeSolver_WaveEq::writeCheckpoint(PetscViewer &viewer)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolver_WaveEq::writeCheckpoint in odeSolver.cpp.\n");
  #endif
  PetscErrorCode ierr;

  // initiate Vec to serve as underlying data set for step count and deltaT to be written out as attributes
  Vec temp;
  VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 1, &temp);
  VecSetBlockSize(temp, 1);
  PetscObjectSetName((PetscObject) temp, "odeSolver_WaveEq_chkpt_data");
  VecSet(temp,0.);

  ierr = PetscViewerHDF5PushGroup(viewer, "/odeSolver_WaveEq");        CHKERRQ(ierr);
  ierr = VecView(temp, viewer);                                         CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer, "odeSolver_WaveEq_chkpt_data", "stepCount", PETSC_INT, &_stepCount); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer, "odeSolver_WaveEq_chkpt_data", "deltaT", PETSC_SCALAR, &_deltaT); CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer); CHKERRQ(ierr);
  VecDestroy(&temp);

  ierr = PetscViewerHDF5PushGroup(viewer, "/odeSolver_WaveEq/var"); CHKERRQ(ierr);
  for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
    ierr = PetscObjectSetName((PetscObject) _var[it->first], (it->first).c_str()); CHKERRQ(ierr);
    ierr = VecView(_var[it->first], viewer);                          CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(viewer); CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/odeSolver_WaveEq/varPrev"); CHKERRQ(ierr);
  for (map<string,Vec>::iterator it = _varPrev.begin(); it!=_varPrev.end(); it++ ) {
    ierr = PetscObjectSetName((PetscObject) _varPrev[it->first], (it->first).c_str()); CHKERRQ(ierr);
    ierr = VecView(_varPrev[it->first], viewer);                          CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(viewer); CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/odeSolver_WaveEq/varNext"); CHKERRQ(ierr);
  for (map<string,Vec>::iterator it = _varNext.begin(); it!=_varNext.end(); it++ ) {
    ierr = PetscObjectSetName((PetscObject) _varNext[it->first], (it->first).c_str()); CHKERRQ(ierr);
    ierr = VecView(_varNext[it->first], viewer);                          CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(viewer); CHKERRQ(ierr);




  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolver_WaveEq::writeCheckpoint in odeSolver.cpp.\n");
  #endif

  return ierr;
}
