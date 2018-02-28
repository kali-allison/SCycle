#include "odeSolver_WaveEq.hpp"

using namespace std;

OdeSolver_WaveEq::OdeSolver_WaveEq(PetscInt maxNumSteps,PetscScalar initT,PetscScalar finalT,PetscScalar deltaT)
: _initT(0),_finalT(finalT),_currT(0),_deltaT(deltaT),
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

PetscErrorCode OdeSolver_WaveEq::setStepSize(const PetscReal deltaT)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolver_WaveEq::setStepSize in odeSolver_waveEq.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  _deltaT = deltaT;
  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolver_WaveEq::setStepSize in odeSolver_waveEq.cpp.\n");
#endif
  return 0;
}

PetscErrorCode OdeSolver_WaveEq::getCurrT(PetscScalar& currT){
  currT = _currT; 
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

  _var = *(&var);
  for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
    Vec temp;
    ierr = VecDuplicate(_var[it->first],&temp); CHKERRQ(ierr);
    ierr = VecSet(temp,0.0); CHKERRQ(ierr);
    _varPrev[it->first] = temp;
  }
  _varPrev["slip"] = _var["dslip"];
  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending WaveEq::setInitialConds in odeSolver_waveEq.cpp.\n");
#endif
  return ierr;
}

PetscErrorCode OdeSolver_WaveEq::integrate(IntegratorContextWave *obj)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolver_WaveEq::integrate in odeSolver_waveEq.cpp.\n");
#endif
  PetscErrorCode ierr = 0;
  double startTime = MPI_Wtime();
  int   stopIntegration = 0;

  if (_finalT==_initT) { return ierr; }
  else if (_deltaT==0) { _deltaT = (_finalT-_initT)/_maxNumSteps; }

  // set initial condition
  ierr = obj->d_dt(_currT,_var,_varPrev);CHKERRQ(ierr);

  ierr = obj->timeMonitor(_currT,_stepCount,_var,_varPrev,stopIntegration);CHKERRQ(ierr); // write first step

  while (_stepCount<_maxNumSteps && _currT<_finalT) {
    ierr = obj->d_dt(_currT,_var,_varPrev);CHKERRQ(ierr);

    _currT = _currT + _deltaT;
    if (_currT>_finalT) { _currT = _finalT; }
    _stepCount++;
    ierr = obj->timeMonitor(_currT,_stepCount,_var,_varPrev,stopIntegration);CHKERRQ(ierr);
  }

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending FEuler::integrate in odeSolver_waveEq.cpp.\n");
#endif
  return ierr;
}


