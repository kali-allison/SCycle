#include "odeSolver.hpp"

//~enum CONTROL { controlP,controlPI,controlPID };
//~const CONTROL controlType = controlPID;

using namespace std;

OdeSolver::OdeSolver(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType)
: _initT(0),_finalT(finalT),_currT(0),_deltaT(deltaT),
  _maxNumSteps(maxNumSteps),_stepCount(0),
  //~_var(NULL),_dvar(NULL),
  _lenVar(0),
  _runTime(0),
  _controlType(controlType)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolver constructor in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();

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

  // because I don't allocate the contents of _var, I don't delete them in this class either
  //~for_each(_var.begin(),_var.end(),DeleteVecObject()); // from Effective STL
  //~for_each(_dvar.begin(),_dvar.end(),DeleteVecObject());

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

//================= FEuler child class functions =======================

FEuler::FEuler(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType)
: OdeSolver(maxNumSteps,finalT,deltaT,controlType)
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

PetscErrorCode FEuler::setInitialConds(vector<Vec>& var)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting FEuler::setInitialConds in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  PetscErrorCode ierr = 0;

  _var = var; // shallow copy
  _lenVar = var.size();

  _dvar.reserve(_lenVar);
  for (int ind=0;ind<_lenVar;ind++) {
    ierr = VecDuplicate(_var[ind],&_dvar[ind]);CHKERRQ(ierr);
    ierr = VecSet(_dvar[ind],0.0);CHKERRQ(ierr);
  }

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending FEuler::setInitialConds in odeSolver.cpp.\n");
#endif
  return ierr;
}

PetscErrorCode FEuler::integrate(IntegratorContext *obj)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting FEuler::integrate in odeSolver.cpp.\n");
#endif
  PetscErrorCode ierr = 0;
  double startTime = MPI_Wtime();

  if (_finalT==_initT) { return ierr; }
  else if (_deltaT==0) { _deltaT = (_finalT-_initT)/_maxNumSteps; }

  while (_stepCount<_maxNumSteps && _currT<_finalT) {

    ierr = obj->d_dt(_currT,_var.begin(),_var.end(),_dvar.begin(),_dvar.end());CHKERRQ(ierr);
    //~ierr = obj->debug(_currT,_stepCount,_var,_dvar,"FE");CHKERRQ(ierr);
    ierr = obj->debug(_currT,_stepCount,_var.begin(),_var.end(),_dvar.begin(),_dvar.end(),"FE");CHKERRQ(ierr);
    for (int varInd=0;varInd<_lenVar;varInd++) {
      ierr = VecAXPY(_var[varInd],_deltaT,_dvar[varInd]);CHKERRQ(ierr); // var = var + deltaT*dvar
    }
    _currT = _currT + _deltaT;
    if (_currT>_finalT) { _currT = _finalT; }
    _stepCount++;
    ierr = obj->timeMonitor(_currT,_stepCount,_var.begin(),_var.end(),_dvar.begin(),_dvar.end());CHKERRQ(ierr);
  }

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending FEuler::integrate in odeSolver.cpp.\n");
#endif
  return ierr;
}



//================= RK32 child class functions =========================

RK32::RK32(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType)
: OdeSolver(maxNumSteps,finalT,deltaT,controlType),
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
  if (_controlType.compare("P") == 0) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"P\n");CHKERRQ(ierr);
  }
  else if (_controlType.compare("PI") == 0) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"PI\n");CHKERRQ(ierr);
  }
  else if (_controlType.compare("PID") == 0) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"PID\n");CHKERRQ(ierr);
  }
  else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"ERROR: timeControlType not understood\n");CHKERRQ(ierr);
    assert(0>1); // automatically fail, because I can't figure out how to use exit commands properly
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

PetscErrorCode RK32::setInitialConds(vector<Vec>& var)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32::setInitialConds in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  PetscErrorCode ierr = 0;

  _var = var;
  _lenVar = var.size();

  _errVec.reserve(_lenVar);

  _varHalfdT.reserve(_lenVar); _dvarHalfdT.reserve(_lenVar);
  _vardT    .reserve(_lenVar);     _dvardT.reserve(_lenVar);
  _var2nd   .reserve(_lenVar);    _dvar2nd.reserve(_lenVar);
  _var3rd   .reserve(_lenVar);

  _dvar.reserve(_lenVar);
  for (int ind=0;ind<_lenVar;ind++) {
    ierr = VecDuplicate(_var[ind],&_dvar[ind]);CHKERRQ(ierr);
    ierr = VecSet(_dvar[ind],0.0);CHKERRQ(ierr);

    ierr = VecDuplicate(_var[ind],&_varHalfdT[ind]);CHKERRQ(ierr);
        ierr = VecSet(_varHalfdT[ind],0.0);CHKERRQ(ierr);
    ierr = VecDuplicate(_var[ind],&_dvarHalfdT[ind]);CHKERRQ(ierr);
        ierr = VecSet(_dvarHalfdT[ind],0.0);CHKERRQ(ierr);
    ierr = VecDuplicate(_var[ind],&_vardT[ind]);CHKERRQ(ierr);
        ierr = VecSet(_vardT[ind],0.0);CHKERRQ(ierr);
    ierr = VecDuplicate(_var[ind],&_dvardT[ind]);CHKERRQ(ierr);
        ierr = VecSet(_dvardT[ind],0.0);CHKERRQ(ierr);
    ierr = VecDuplicate(_var[ind],&_var2nd[ind]);CHKERRQ(ierr);
        ierr = VecSet(_var2nd[ind],0.0);CHKERRQ(ierr);
    ierr = VecDuplicate(_var[ind],&_dvar2nd[ind]);CHKERRQ(ierr);
        ierr = VecSet(_dvar2nd[ind],0.0);CHKERRQ(ierr);
    ierr = VecDuplicate(_var[ind],&_var3rd[ind]);CHKERRQ(ierr);
        ierr = VecSet(_var3rd[ind],0.0);CHKERRQ(ierr);

    ierr = VecDuplicate(_var[ind],&_errVec[ind]);CHKERRQ(ierr);
        ierr = VecSet(_errVec[ind],0.0);CHKERRQ(ierr);
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

  if (_controlType.compare("P") == 0) {
    // if using integral feedback controller (I)
    PetscReal alpha = 1./(1.+_ord);
    stepRatio = _kappa*pow(_atol/totErr,alpha);
  }
  else if (_controlType.compare("PID") == 0) {

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
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: timeControlType not understood\n");
    assert(0>1); // automatically fail, because I can't figure out how to use exit commands properly
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
  PetscInt       size,totSize=0;


  //~for (int ind=0;ind<_lenVar;ind++) {
  for (int ind=0;ind<1;ind++) {
    ierr = VecWAXPY(_errVec[ind],-1.0,_var2nd[ind],_var3rd[ind]);CHKERRQ(ierr);

    // error based on max norm
    ierr = VecNorm(_errVec[ind],NORM_INFINITY,&err[ind]);CHKERRQ(ierr);
    if (err[ind]>totErr) { totErr=err[ind]; }

    // error based on weighted 2 norm
    //~VecDot(_errVec[ind],_errVec[ind],&err[ind]);
    //~VecGetSize(_errVec[ind],&size);
    //~totErr += sqrt(err[ind]/size);
    //~totErr += err[ind];
    //~totSize += size;
  }
  //~totErr = sqrt(totErr/totSize);
  //~totErr = sqrt(totErr/size);


  // abs error of slip
  //~ierr = VecNorm(_errVec[0],NORM_INFINITY,&totErr);CHKERRQ(ierr);
  //~totErr = abs(totErr);
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32::computeError in odeSolver.cpp.\n");
#endif
  return totErr;
}


PetscErrorCode RK32::integrate(IntegratorContext *obj)
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
  ierr = obj->d_dt(_currT,_var.begin(),_var.end(),_dvar.begin(),_dvar.end());CHKERRQ(ierr);
  ierr = obj->debug(_currT,_stepCount,_var.begin(),_var.end(),_dvar.begin(),_dvar.end(),"IC");CHKERRQ(ierr);
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
      ierr = obj->d_dt(_currT+0.5*_deltaT,_varHalfdT.begin(),_varHalfdT.end(),_dvarHalfdT.begin(),_dvarHalfdT.end());CHKERRQ(ierr);
      ierr = obj->debug(_currT+0.5*_deltaT,_stepCount,_varHalfdT.begin(),_varHalfdT.end(),_dvarHalfdT.begin(),_dvarHalfdT.end(),"t+dt/2");CHKERRQ(ierr);

      // stage 2: integrate fields to _currT + _deltaT
      for (int ind=0;ind<_lenVar;ind++) {
        ierr = VecWAXPY(_vardT[ind],-_deltaT,_dvar[ind],_var[ind]);CHKERRQ(ierr);
        ierr = VecAXPY(_vardT[ind],2*_deltaT,_dvarHalfdT[ind]);CHKERRQ(ierr);
      }
      ierr = obj->d_dt(_currT+_deltaT,_vardT.begin(),_vardT.end(),_dvardT.begin(),_dvardT.end());CHKERRQ(ierr);
      ierr = obj->debug(_currT+_deltaT,_stepCount,_vardT.begin(),_vardT.end(),_dvardT.begin(),_dvardT.end(),"t+dt");CHKERRQ(ierr);

      // 2nd and 3rd order update
      for (int ind=0;ind<_lenVar;ind++) {
        ierr = VecWAXPY(_var2nd[ind],0.5*_deltaT,_dvar[ind],_var[ind]);CHKERRQ(ierr);
        ierr = VecAXPY(_var2nd[ind],0.5*_deltaT,_dvardT[ind]);CHKERRQ(ierr);

        ierr = VecWAXPY(_var3rd[ind],_deltaT/6.0,_dvar[ind],_var[ind]);CHKERRQ(ierr);
        ierr = VecAXPY(_var3rd[ind],2*_deltaT/3.0,_dvarHalfdT[ind]);CHKERRQ(ierr);
        ierr = VecAXPY(_var3rd[ind],_deltaT/6.0,_dvardT[ind]);CHKERRQ(ierr);
      }
      ierr = obj->debug(_currT+_deltaT,_stepCount,_var2nd.begin(),_var2nd.end(),_dvardT.begin(),_dvardT.end(),"2nd");CHKERRQ(ierr);
      ierr = obj->debug(_currT+_deltaT,_stepCount,_var3rd.begin(),_var3rd.end(),_dvardT.begin(),_dvardT.end(),"3rd");CHKERRQ(ierr);


      // calculate error
      totErr = computeError();
      #if ODEPRINT > 0
        ierr = PetscPrintf(PETSC_COMM_WORLD,"    totErr = %.15e\n",totErr);
      #endif
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
    ierr = obj->d_dt(_currT,_var.begin(),_var.end(),_dvar.begin(),_dvar.end());CHKERRQ(ierr);
    ierr = obj->debug(_currT,_stepCount,_var.begin(),_var.end(),_dvar.begin(),_dvar.end(),"F");CHKERRQ(ierr);

    if (totErr!=0.0) {
      _deltaT = computeStepSize(totErr);
    }

    ierr = obj->timeMonitor(_currT,_stepCount,_var.begin(),_var.end(),_dvar.begin(),_dvar.end());CHKERRQ(ierr);
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
