#include "odeSolver_WaveImex.hpp"

using namespace std;

OdeSolver_WaveImex::OdeSolver_WaveImex(PetscInt maxNumSteps,PetscScalar initT,PetscScalar finalT,PetscScalar deltaT)
: _initT(initT),_finalT(finalT),_currT(initT),_deltaT(deltaT),
  _maxNumSteps(maxNumSteps),_stepCount(0),
  _lenVar(0),_runTime(0)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolver_WaveImex constructor in odeSolver_waveImex.cpp.\n");
#endif
  double startTime = MPI_Wtime();

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolver_WaveImex constructor in odeSolver_waveImex.cpp.\n");
#endif
}

PetscErrorCode OdeSolver_WaveImex::setStepSize(const PetscReal deltaT)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolver_WaveImex::setStepSize in odeSolver_waveImex.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  _deltaT = deltaT;
  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolver_WaveImex::setStepSize in odeSolver_waveImex.cpp.\n");
#endif
  return 0;
}

PetscErrorCode OdeSolver_WaveImex::getCurrT(PetscScalar& currT){
  PetscErrorCode ierr = 0;
  currT = _currT; 
  return ierr;
}

PetscErrorCode OdeSolver_WaveImex::view()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolver_WaveImex::view in odeSolver_WaveImex.cpp.\n");
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
  PetscPrintf(PETSC_COMM_WORLD,"Ending WaveEq::view in odeSolver_waveImex.cpp.\n");
#endif
}

PetscErrorCode OdeSolver_WaveImex::setInitialConds(std::map<string,Vec>& varEx, std::map<string,Vec>& varIm)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting WaveEq::setInitialConds in odeSolver_waveImex.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  PetscErrorCode ierr = 0;

  _varEx = *(&varEx);
  for (map<string,Vec>::iterator it = _varEx.begin(); it!=_varEx.end(); it++ ) {
    Vec temp;
    ierr = VecDuplicate(_varEx[it->first],&temp); CHKERRQ(ierr);
    ierr = VecSet(temp,0.0); CHKERRQ(ierr);
    _varPrev[it->first] = temp;
  }
  _varPrev["slip"] = _varEx["dslip"];

  _varImex = varIm;
  for (map<string,Vec>::iterator it=_varImex.begin(); it!=_varImex.end(); it++ ) {
    Vec vardTIm;
    ierr = VecDuplicate(_varImex[it->first],&vardTIm); CHKERRQ(ierr);
    ierr = VecSet(vardTIm,0.0); CHKERRQ(ierr);
    _varImexPrev[it->first] = vardTIm;
  }

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending WaveEq::setInitialConds in odeSolver_waveImex.cpp.\n");
#endif
  return ierr;
}

PetscErrorCode OdeSolver_WaveImex::integrate(IntegratorContextWave *obj)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolver_WaveImex::integrate in odeSolver_waveImex.cpp.\n");
#endif
  PetscErrorCode ierr = 0;
  double startTime = MPI_Wtime();
  int   stopIntegration = 0;

  if (_finalT==_initT) { return ierr; }
  else if (_deltaT==0) { _deltaT = (_finalT-_initT)/_maxNumSteps; }

  // set initial condition
  ierr = obj->d_dt(_currT,_varEx,_varPrev, _varImex, _varImexPrev);CHKERRQ(ierr);

  ierr = obj->timeMonitor(_currT,_stepCount,_varEx,_varPrev,stopIntegration);CHKERRQ(ierr); // write first step

  while (_stepCount<_maxNumSteps && _currT<_finalT) {
    ierr = obj->d_dt(_currT,_varEx,_varPrev, _varImex, _varImexPrev);CHKERRQ(ierr);

    _currT = _currT + _deltaT;
    if (_currT>_finalT) { _currT = _finalT; }
    _stepCount++;
    ierr = obj->timeMonitor(_currT,_stepCount,_varEx,_varPrev,stopIntegration);CHKERRQ(ierr);
  }

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending FEuler::integrate in odeSolver_waveImex.cpp.\n");
#endif
  return ierr;
}


