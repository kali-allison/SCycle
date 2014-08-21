#include "odeSolver.hpp"

enum CONTROL { controlP,controlPI,controlPID };
const CONTROL controlType = controlP;

using namespace std;

OdeSolver::OdeSolver(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT)
:_initT(0),_finalT(finalT),_currT(0),_deltaT(deltaT),
_maxNumSteps(maxNumSteps),_stepCount(0),
_var(NULL),_dvar(NULL),_lenVar(0),//_userContext(NULL),
_runTime(0)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolver constructor in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();

  //~_rhsFunc = &newtempRhsFunc;
  //~_timeMonitor = &newtempTimeMonitor;

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolver constructor in odeSolver.cpp.\n");
#endif
}

OdeSolver::~OdeSolver()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolver destructor in odeSolver.cpp.\n");
#endif
  if (_dvar!=0) {
    for (int ind=0;ind<_lenVar;ind++) {
      VecDestroy(&_dvar[ind]);
    }
    delete[] _dvar;
  }
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolver destructor in odeSolver.cpp.\n");
#endif
}


PetscErrorCode OdeSolver::setTimeRange(const PetscReal initT,const PetscReal finalT)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolver::setTimeRange in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();

  _initT = initT;
  _currT = initT;
  _finalT = finalT;

  _runTime += MPI_Wtime() - startTime;
  return 0;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolver::setTimeRange in odeSolver.cpp.\n");
#endif
}

PetscErrorCode OdeSolver::setStepSize(const PetscReal deltaT)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolver::setStepSize in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  _deltaT = deltaT;
  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolver::setStepSize in odeSolver.cpp.\n");
#endif
  return 0;
}


//~PetscErrorCode OdeSolver::setRhsFunc(PetscErrorCode (*rhsFunc)(const PetscReal,const int,Vec*,Vec*,void*))
//~{
  //~double startTime = MPI_Wtime();
  //~_rhsFunc = rhsFunc;
  //~_runTime += MPI_Wtime() - startTime;
  //~return 0;
//~}

//~PetscErrorCode OdeSolver::setUserContext(void * userContext)
//~{
  //~double startTime = MPI_Wtime();
  //~_userContext = userContext;
  //~_runTime += MPI_Wtime() - startTime;
  //~return 0;
//~}


//~PetscErrorCode OdeSolver::setTimeMonitor(PetscErrorCode (*timeMonitor)(const PetscReal,const PetscInt,const Vec*,const int,void*))
//~{
  //~double startTime = MPI_Wtime();
  //~_timeMonitor = timeMonitor;
  //~_runTime += MPI_Wtime() - startTime;
  //~return 0;
//~}

//================= FEuler child class functions =======================

FEuler::FEuler(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT)
: OdeSolver(maxNumSteps,finalT,deltaT)
{}

PetscErrorCode FEuler::view()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting FEuler::view in odeSolver.cpp.\n");
#endif
  PetscErrorCode ierr;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTimeSolver summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   integration algorithm: forward euler\n");CHKERRQ(ierr);
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
  PetscPrintf(PETSC_COMM_WORLD,"Ending FEuler::view in odeSolver.cpp.\n");
#endif
}

PetscErrorCode FEuler::setInitialConds(Vec *var, const int lenVar)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting FEuler::setInitialConds in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  PetscErrorCode ierr = 0;
  PetscScalar    zero=0.0;

  _var = var;
  _lenVar = lenVar;

  _dvar = new Vec[_lenVar];
  for (int ind=0;ind<_lenVar;ind++) {
    ierr = VecDuplicate(_var[ind],&_dvar[ind]);CHKERRQ(ierr);
    ierr = VecSet(_dvar[ind],zero);CHKERRQ(ierr);
  }

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending FEuler::setInitialConds in odeSolver.cpp.\n");
#endif
  return ierr;
}

PetscErrorCode FEuler::integrate(Lithosphere *obj)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting FEuler::integrate in odeSolver.cpp.\n");
#endif
  PetscErrorCode ierr = 0;
  double startTime = MPI_Wtime();

  if (_finalT==_initT) { return ierr; }
  else if (_deltaT==0) { _deltaT = (_finalT-_initT)/_maxNumSteps; }

  while (_stepCount<_maxNumSteps && _currT<_finalT) {

    ierr = obj->d_dt(_currT,_var,_dvar);CHKERRQ(ierr);
    //~ierr = obj->debug(_currT,_stepCount,_var,_dvar,"FE");CHKERRQ(ierr);
    for (int varInd=0;varInd<_lenVar;varInd++) {
      ierr = VecAXPY(_var[varInd],_deltaT,_dvar[varInd]);CHKERRQ(ierr); // var = var + deltaT*dvar
    }
    _currT = _currT + _deltaT;
    if (_currT>_finalT) { _currT = _finalT; }
    _stepCount++;
    ierr = obj->timeMonitor(_currT,_stepCount,_var,_dvar);CHKERRQ(ierr);
  }

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending FEuler::integrate in odeSolver.cpp.\n");
#endif
  return ierr;
}



//================= RK32 child class functions =========================

RK32::RK32(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT)
: OdeSolver(maxNumSteps,finalT,deltaT),
  _minDeltaT(0),_maxDeltaT(finalT),
  _atol(1e-9),_kappa(0.9),_ord(2.0),
  _numRejectedSteps(0),_numMinSteps(0),_numMaxSteps(0)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32::constructor in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();

  _absErr[0] = 0;_absErr[1] = 0;_absErr[2] = 0;

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolver::constructor in odeSolver.cpp.\n");
#endif
}

RK32::~RK32()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32::destructor in odeSolver.cpp.\n");
#endif
  // destruct temporary containers
  for (int ind=0;ind<_lenVar;ind++) {
    VecDestroy(&_varHalfdT[ind]); VecDestroy(&_dvarHalfdT[ind]);
    VecDestroy(&_vardT[ind]);     VecDestroy(&_dvardT[ind]);
    VecDestroy(&_var2nd[ind]);    VecDestroy(&_dvar2nd[ind]);
    VecDestroy(&_var3rd[ind]);
    VecDestroy(&_errVec[ind]);
  }
  delete[] _varHalfdT; delete[] _dvarHalfdT;
  delete[] _vardT;     delete[] _dvardT;
  delete[] _var2nd;    delete[] _dvar2nd;
  delete[] _var3rd;
  delete[] _errVec;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32::destructor in odeSolver.cpp.\n");
#endif
}

PetscErrorCode RK32::view()
{
  PetscErrorCode ierr;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTime Integration summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   integration algorithm: runge-kutta (3,2)\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   control scheme: ");CHKERRQ(ierr);
  if (controlType == controlP) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"P\n");CHKERRQ(ierr);
  }
  else if (controlType == controlPI) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"PI\n");CHKERRQ(ierr);
  }
  else if (controlType == controlPID) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"PID\n");CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time interval: %g to %g\n",
                     _initT,_finalT);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   permitted step size range: [%g,%g]\n",
                     _minDeltaT,_maxDeltaT);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   total number of steps taken: %i/%i\n",
                     _stepCount,_maxNumSteps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   final time reached: %g\n",
                     _currT);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   tolerance: %g\n",
                     _atol);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of rejected steps: %i\n",
                     _numRejectedSteps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of times min step size enforced: %i\n",
                     _numMinSteps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of times max step size enforced: %i\n",
                     _numMaxSteps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   total run time: %g\n",
                     _runTime);CHKERRQ(ierr);

  return 0;
}

PetscErrorCode RK32::setTolerance(const PetscReal tol)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32::setTolerance in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  _atol = tol;

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32::setTolerance in odeSolver.cpp.\n");
#endif
  return 0;
}

PetscErrorCode RK32::setInitialConds(Vec *var, const int lenVar)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32::setInitialConds in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  PetscErrorCode ierr = 0;
  PetscScalar    zero=0.0;

  _var = var;
  _lenVar = lenVar;

  _dvar = new Vec[_lenVar];

  _varHalfdT = new Vec[_lenVar]; _dvarHalfdT = new Vec[_lenVar];
  _vardT     = new Vec[_lenVar];     _dvardT = new Vec[_lenVar];
  _var2nd    = new Vec[_lenVar];    _dvar2nd = new Vec[_lenVar];
  _var3rd    = new Vec[_lenVar];

  _errVec  = new Vec[_lenVar];

  for (int ind=0;ind<_lenVar;ind++) {
    ierr = VecDuplicate(_var[ind],&_dvar[ind]);CHKERRQ(ierr);
    ierr = VecSet(_dvar[ind],zero);CHKERRQ(ierr);

    ierr = VecDuplicate(_var[ind],&_varHalfdT[ind]);CHKERRQ(ierr);
    ierr = VecDuplicate(_var[ind],&_dvarHalfdT[ind]);CHKERRQ(ierr);
    ierr = VecDuplicate(_var[ind],&_vardT[ind]);CHKERRQ(ierr);
    ierr = VecDuplicate(_var[ind],&_dvardT[ind]);CHKERRQ(ierr);
    ierr = VecDuplicate(_var[ind],&_var2nd[ind]);CHKERRQ(ierr);
    ierr = VecDuplicate(_var[ind],&_dvar2nd[ind]);CHKERRQ(ierr);
    ierr = VecDuplicate(_var[ind],&_var3rd[ind]);CHKERRQ(ierr);

    ierr = VecDuplicate(_var[ind],&_errVec[ind]);CHKERRQ(ierr);
  }

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32::setInitialConds in odeSolver.cpp.\n");
#endif
  return ierr;
}

PetscErrorCode RK32::setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32::setTimeStepBounds in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  _minDeltaT = minDeltaT;
  _maxDeltaT = maxDeltaT;
  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32::setTimeStepBounds in odeSolver.cpp.\n");
#endif
  return 0;
}

PetscReal RK32::computeStepSize(const PetscReal totErr)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32::computeStepSize in odeSolver.cpp.\n");
#endif
  PetscReal stepRatio;

  if (controlType == controlP) {
    // if using integral feedback controller (I)
    PetscReal alpha = 1./(1.+_ord);
    stepRatio = _kappa*pow(_atol/totErr,alpha);
  }
  else if (controlType == controlPID) {

    //if using proportional-integral-derivative feedback (PID)
    _absErr[(_stepCount+_numRejectedSteps-1)%3] = totErr;

    PetscReal alpha = 0.49/_ord;
    PetscReal beta  = 0.34/_ord;
    PetscReal gamma = 0.1/_ord;

    if (_stepCount < 4) {
      stepRatio = _kappa*pow(_atol/totErr,1./(1.+_ord));
    }
    else {
      stepRatio = _kappa*pow(_atol/_absErr[(_stepCount+_numRejectedSteps-1)%3],alpha)
                             *pow(_atol/_absErr[(_stepCount+_numRejectedSteps-2)%3],beta)
                             *pow(_atol/_absErr[(_stepCount+_numRejectedSteps-3)%3],gamma);
    }
  }

    //~PetscPrintf(PETSC_COMM_WORLD,"   _stepCount %i,absErr[0]=%e,absErr[1]=%e,absErr[2]=%e\n",
                //~_stepCount,_absErr[0],_absErr[1],_absErr[2]);
    //~PetscPrintf(PETSC_COMM_WORLD,"   (_stepCount-1)%%3 = %i,(_stepCount-2)%%3 = %i,(_stepCount-3)%%3=%i\n",
                //~(_stepCount+_numRejectedSteps-2)%3,(_stepCount+_numRejectedSteps-3)%3,(_stepCount+_numRejectedSteps-4)%3);

  PetscReal deltaT = stepRatio*_deltaT;

  // respect bounds on min and max possible step size
  deltaT=min(_maxDeltaT,deltaT);
  deltaT = max(_minDeltaT,deltaT);

  if (_minDeltaT == deltaT) {
    _numMinSteps++;
  }
  else if (_maxDeltaT == deltaT) {
    _numMaxSteps++;
  }
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32::computeStepSize in odeSolver.cpp.\n");
#endif
  return deltaT;
}

PetscReal RK32::computeError()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32::computeError in odeSolver.cpp.\n");
#endif
  PetscErrorCode ierr = 0;
  PetscReal      err[_lenVar],totErr=0.0;
  PetscInt       size;


  for (int ind=0;ind<_lenVar;ind++) {
    ierr = VecWAXPY(_errVec[ind],-1.0,_var2nd[ind],_var3rd[ind]);CHKERRQ(ierr);

    // error based on max norm
    //~ierr = VecNorm(_errVec[ind],NORM_INFINITY,&err[ind]);CHKERRQ(ierr);
    //~if (err[ind]>totErr) { totErr=err[ind]; }

    // error based on weighted 2 norm
    VecDot(_errVec[ind],_errVec[ind],&err[ind]);
    VecGetSize(_errVec[ind],&size);
    totErr += err[ind]/size;
  }
  totErr = sqrt(totErr);


  // abs error of slip
  //~ierr = VecNorm(_errVec[0],NORM_INFINITY,&totErr);CHKERRQ(ierr);
  //~totErr = abs(totErr);
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32::computeError in odeSolver.cpp.\n");
#endif
  return totErr;
}


PetscErrorCode RK32::integrate(Lithosphere *obj)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32::integrate in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();

  PetscErrorCode ierr=0;
  PetscReal      totErr=0.0;
  PetscInt       attemptCount = 0;

  if (_finalT==_initT) { return ierr; }
  else if (_deltaT==0) { _deltaT = (_finalT-_initT)/_maxNumSteps; }

  // set initial condition
  ierr = obj->d_dt(_currT,_var,_dvar);CHKERRQ(ierr);
  //~ierr = obj->debug(_currT,_stepCount,_var,_dvar,"IC");CHKERRQ(ierr);
  while (_stepCount<_maxNumSteps && _currT<_finalT) {

    _stepCount++;
    attemptCount = 0;
    while (attemptCount<100) { // repeat until time step is acceptable
      attemptCount++;
      if (attemptCount>=100) {PetscPrintf(PETSC_COMM_WORLD,"   WARNING: maximum number of attempts reached\n"); }
      //~ierr = PetscPrintf(PETSC_COMM_WORLD,"   attemptCount=%i\n",attemptCount);CHKERRQ(ierr);
      if (_currT+_deltaT>_finalT) { _deltaT=_finalT-_currT; }

      // stage 1: integrate fields to _currT + 0.5*deltaT
      for (int ind=0;ind<_lenVar;ind++) {
        ierr = VecWAXPY(_varHalfdT[ind],0.5*_deltaT,_dvar[ind],_var[ind]);CHKERRQ(ierr);
      }
      ierr = obj->d_dt(_currT+0.5*_deltaT,_varHalfdT,_dvarHalfdT);CHKERRQ(ierr);
      //~ierr = obj->debug(_currT+0.5*_deltaT,_stepCount,_varHalfdT,_dvarHalfdT,"t+dt/2");CHKERRQ(ierr);

      // stage 2: integrate fields to _currT + _deltaT
      for (int ind=0;ind<_lenVar;ind++) {
        ierr = VecWAXPY(_vardT[ind],-_deltaT,_dvar[ind],_var[ind]);CHKERRQ(ierr);
        ierr = VecAXPY(_vardT[ind],2*_deltaT,_dvarHalfdT[ind]);CHKERRQ(ierr);
      }
      ierr = obj->d_dt(_currT+_deltaT,_vardT,_dvardT);CHKERRQ(ierr);
      //~ierr = obj->debug(_currT+_deltaT,_stepCount,_vardT,_dvardT,"t+dt");CHKERRQ(ierr);

      // 2nd and 3rd order update
      for (int ind=0;ind<_lenVar;ind++) {
        ierr = VecWAXPY(_var2nd[ind],0.5*_deltaT,_dvar[ind],_var[ind]);CHKERRQ(ierr);
        ierr = VecAXPY(_var2nd[ind],0.5*_deltaT,_dvardT[ind]);CHKERRQ(ierr);

        ierr = VecWAXPY(_var3rd[ind],_deltaT/6.0,_dvar[ind],_var[ind]);CHKERRQ(ierr);
        ierr = VecAXPY(_var3rd[ind],2*_deltaT/3.0,_dvarHalfdT[ind]);CHKERRQ(ierr);
        ierr = VecAXPY(_var3rd[ind],_deltaT/6.0,_dvardT[ind]);CHKERRQ(ierr);
      }
      //~ierr = obj->debug(_currT+_deltaT,_stepCount,_var2nd,_dvardT,"2nd");CHKERRQ(ierr);
      //~ierr = obj->debug(_currT+_deltaT,_stepCount,_var3rd,_dvardT,"3rd");CHKERRQ(ierr);

      // calculate error
      totErr = computeError();

      if (totErr<_atol) { break; }
      _deltaT = computeStepSize(totErr);
      if (_minDeltaT == _deltaT) { break; }

      _numRejectedSteps++;
    }
    _currT = _currT+_deltaT;

    // accept 3rd order solution as update
    for (int ind=0;ind<_lenVar;ind++) {
      ierr = VecCopy(_var3rd[ind],_var[ind]);CHKERRQ(ierr);
    }
    ierr = obj->d_dt(_currT,_var,_dvar);CHKERRQ(ierr);
    //~ierr = obj->debug(_currT,_stepCount,_var,_dvar,"F");CHKERRQ(ierr);

    if (totErr!=0.0) {
      _deltaT = computeStepSize(totErr);
    }

    ierr = obj->timeMonitor(_currT,_stepCount,_var,_dvar);CHKERRQ(ierr);
  }

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32::integrate in odeSolver.cpp.\n");
#endif
  return ierr;
}



//================= placeholder functions ================================

PetscErrorCode newtempRhsFunc(const PetscReal time, const int lenVar, Vec *var, Vec *dvar, void*userContext)
{
  PetscErrorCode ierr = 0;
  SETERRQ(PETSC_COMM_WORLD,1,"rhsFunc not defined.\n");
  return ierr;
}

PetscErrorCode newtempTimeMonitor(const PetscReal time, const PetscInt stepCount,
                               const Vec *var, const int lenVar, void*userContext)
{
  PetscErrorCode ierr = 0;
  return ierr;
}
