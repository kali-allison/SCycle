#include "odeSolver.hpp"

using namespace std;

OdeSolver::OdeSolver(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType)
: _initT(0),_finalT(finalT),_currT(0),_deltaT(deltaT),
  _maxNumSteps(maxNumSteps),_stepCount(0),
  _runTime(0),
  _controlType(controlType),_normType("L2_absolute")
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


// if starting with a nonzero initial step count
PetscErrorCode OdeSolver::setInitialStepCount(const PetscReal stepCount) {
  _stepCount = stepCount;
  _maxNumSteps = stepCount + _maxNumSteps;
  return 0;
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

PetscErrorCode OdeSolver::setToleranceType(const std::string normType)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolver::setToleranceType in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  _normType = normType;
  assert(_normType.compare("L2_relative")==0 ||
      _normType.compare("L2_absolute")==0 );

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolver::setToleranceType in odeSolver.cpp.\n");
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
  PetscErrorCode ierr = 0;

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

PetscErrorCode FEuler::setInitialConds(map<string,Vec>& var)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting FEuler::setInitialConds in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  PetscErrorCode ierr = 0;

  _var = var; // shallow copy

  for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
    Vec temp;
    ierr = VecDuplicate(_var[it->first],&temp); CHKERRQ(ierr);
    ierr = VecSet(temp,0.0); CHKERRQ(ierr);
    _dvar[it->first] = temp;
  }


  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending FEuler::setInitialConds in odeSolver.cpp.\n");
#endif
  return ierr;
}

PetscErrorCode FEuler::integrate(IntegratorContextEx *obj)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting FEuler::integrate in odeSolver.cpp.\n");
#endif
  PetscErrorCode ierr = 0;
  double startTime = MPI_Wtime();
  int stopIntegration = 0;

  if (_finalT==_initT) { return ierr; }
  else if (_deltaT==0) { _deltaT = (_finalT-_initT)/_maxNumSteps; }

  // set initial condition
  ierr = obj->d_dt(_currT,_var,_dvar);CHKERRQ(ierr);
  ierr = obj->timeMonitor(_currT,_deltaT,_stepCount,stopIntegration); CHKERRQ(ierr); // write first step

  while (_stepCount<_maxNumSteps && _currT<_finalT) {

    ierr = obj->d_dt(_currT,_var,_dvar);CHKERRQ(ierr);
    for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
      ierr = VecAXPY(_var[it->first],_deltaT,_dvar[it->first]);CHKERRQ(ierr); // var = var + deltaT*dvar
    }

    _currT = _currT + _deltaT;
    if (_currT>_finalT) { _currT = _finalT; }
    _stepCount++;
    ierr = obj->timeMonitor(_currT,_deltaT,_stepCount,stopIntegration); CHKERRQ(ierr);
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
  _atol(1e-9),_kappa(0.9),_ord(3.0),
  _numRejectedSteps(0),_numMinSteps(0),_numMaxSteps(0)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32::constructor in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();

  _errA.resize(2);
  _errA.push_front(0);
  _errA.push_front(0);

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
  destroyVector(_dvar);
  destroyVector(_k1);
  destroyVector(_f1);
  destroyVector(_k2);
  destroyVector(_f2);
  destroyVector(_y2);
  destroyVector(_y3);


  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending RK32::destructor in odeSolver.cpp.\n");
  #endif
}

PetscErrorCode RK32::view()
{
  PetscErrorCode ierr = 0;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTime Integration summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   integration algorithm: runge-kutta (3,2)\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   control scheme: %s\n",_controlType.c_str());CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   variables used in determining time step = %s\n",vector2str(_errInds).c_str());CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   scale factors = %s\n",vector2str(_scale).c_str());CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time interval: %g to %g\n",_initT,_finalT);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   permitted step size range: [%g,%g]\n",_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   total number of steps taken: %i/%i\n",_stepCount,_maxNumSteps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   final time reached: %g\n",_currT);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   tolerance: %g\n",_totTol);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of rejected steps: %i\n",_numRejectedSteps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of times min step size enforced: %i\n",_numMinSteps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of times max step size enforced: %i\n",_numMaxSteps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   total run time: %g\n",_runTime);CHKERRQ(ierr);

  return 0;
}

PetscErrorCode RK32::setTolerance(const PetscReal tol)
{
  _atol = tol;
  _rtol = tol;
  _totTol = tol;
  return 0;
}

PetscErrorCode RK32::setInitialConds(std::map<string,Vec>& var)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32::setInitialConds in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  PetscErrorCode ierr = 0;

  _var = var;
  for (map<string,Vec>::iterator it=var.begin(); it!=var.end(); it++ ) {
    Vec dvar;
    ierr = VecDuplicate(_var[it->first],&dvar); CHKERRQ(ierr);
    ierr = VecSet(dvar,0.0); CHKERRQ(ierr);
    _dvar[it->first] = dvar;

    Vec varHalfdT;
    ierr = VecDuplicate(_var[it->first],&varHalfdT); CHKERRQ(ierr);
    ierr = VecSet(varHalfdT,0.0); CHKERRQ(ierr);
    _k1[it->first] = varHalfdT;

    Vec dvarHalfdT;
    ierr = VecDuplicate(_var[it->first],&dvarHalfdT); CHKERRQ(ierr);
    ierr = VecSet(dvarHalfdT,0.0); CHKERRQ(ierr);
    _f1[it->first] = dvarHalfdT;

    Vec vardT;
    ierr = VecDuplicate(_var[it->first],&vardT); CHKERRQ(ierr);
    ierr = VecSet(vardT,0.0); CHKERRQ(ierr);
    _k2[it->first] = vardT;

    Vec dvardT;
    ierr = VecDuplicate(_var[it->first],&dvardT); CHKERRQ(ierr);
    ierr = VecSet(dvardT,0.0); CHKERRQ(ierr);
    _f2[it->first] = dvardT;

    Vec var2nd;
    ierr = VecDuplicate(_var[it->first],&var2nd); CHKERRQ(ierr);
    ierr = VecSet(var2nd,0.0); CHKERRQ(ierr);
    _y2[it->first] = var2nd;

    Vec var3rd;
    ierr = VecDuplicate(_var[it->first],&var3rd); CHKERRQ(ierr);
    ierr = VecSet(var3rd,0.0); CHKERRQ(ierr);
    _y3[it->first] = var3rd;
  }

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32::setInitialConds in odeSolver.cpp.\n");
#endif
  return ierr;
}

PetscErrorCode RK32::setErrInds(std::vector<string>& errInds) { _errInds = errInds; return 0; }

PetscErrorCode RK32::setErrInds(std::vector<string>& errInds, std::vector<double> scale)
{
  _errInds = errInds;
  _scale = scale;
  return 0;
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
    stepRatio = _kappa*pow(_totTol/totErr,alpha);
  }
  else if (_controlType.compare("PID") == 0) {
    //if using proportional-integral-derivative feedback (PID)

    PetscReal alpha = 0.49/_ord;
    PetscReal beta  = 0.34/_ord;
    PetscReal gamma = 0.1/_ord;

    if (_stepCount < 3) {
      stepRatio = _kappa*pow(_totTol/totErr,1./(1.+_ord));
    }
    else {
      stepRatio = _kappa * pow(_totTol/totErr,alpha)
                         * pow(_errA[0]/_totTol,beta)
                         * pow(_totTol/_errA[1],gamma);
    }
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: timeControlType not understood\n");
    assert(0>1); // automatically fail
  }

  PetscReal deltaT = stepRatio*_deltaT;

  // respect bounds on min and max possible step size
  deltaT = min(_deltaT*5.0,deltaT); // cap growth rate of step size
  deltaT=min(_maxDeltaT,deltaT); // absolute max
  deltaT = max(_minDeltaT,deltaT);

  if (_minDeltaT == deltaT) { _numMinSteps++; }
  else if (_maxDeltaT == deltaT) { _numMaxSteps++; }
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
  PetscReal      err,_totErr=0.0;

    // if using absolute error for control
  // error: the absolute L2 error, weighted by N and a user-inputted scale factor
  // tolerance: the absolute tolerance
  if (_normType.compare("L2_absolute")==0) {
    for(std::vector<int>::size_type i = 0; i != _errInds.size(); i++) {
      std::string key = _errInds[i];
      Vec errVec; VecDuplicate(_y3[key],&errVec); VecSet(errVec,0.0);
      ierr = VecWAXPY(errVec,-1.0,_y3[key],_y2[key]); CHKERRQ(ierr);
      VecNorm(errVec,NORM_2,&err);
      VecDestroy(&errVec);

      PetscInt N = 0; VecGetSize(_y3[key],&N);
      _totErr += err / (sqrt(N) * _scale[i]);
    }
  }

  // if using relative error for control
  // error: the absolute L2 error, scaled by the L2 norm of the solution
  // and a user-inputted scale factor
  // tolerance: the relative tolerance
  if (_normType.compare("L2_relative")==0) {
    for(std::vector<int>::size_type i = 0; i != _errInds.size(); i++) {
      std::string key = _errInds[i];
      Vec errVec; VecDuplicate(_y3[key],&errVec); VecSet(errVec,0.0);
      ierr = VecWAXPY(errVec,-1.0,_y3[key],_y2[key]); CHKERRQ(ierr);
      VecNorm(errVec,NORM_2,&err);
      VecDestroy(&errVec);

      PetscReal s = 0; VecNorm(_y3[key],NORM_2,&s);
      _totErr += err / (s * _scale[i]);
    }
  }

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32::computeError in odeSolver.cpp.\n");
#endif
  return _totErr;
}


PetscErrorCode RK32::integrate(IntegratorContextEx *obj)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32::integrate in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();

  PetscErrorCode ierr=0;
  PetscReal      _totErr=0.0;
  PetscInt       attemptCount = 0;
  int            stopIntegration = 0;

  // build default errInds if it hasn't been defined already
  if (_errInds.size()==0) {
    for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
      _errInds.push_back(it->first);
    }
  }

  // check that errInds is valid
  for(std::vector<int>::size_type i = 0; i != _errInds.size(); i++) {
    std::string key = _errInds[i];
    if (_var.find(key) == _var.end()) {
      PetscPrintf(PETSC_COMM_WORLD,"ERROR: %s is not an element of explicitly integrated variable!\n",key.c_str());
    }
    assert(_var.find(key) != _var.end());
  }

  // set up scaling for elements in errInds
  if (_scale.size() == 0) { // if 0 entries, set all to 1
    for(std::vector<int>::size_type i = 0; i != _errInds.size(); i++) {
      _scale.push_back(1.0);
    }
  }
  assert(_scale.size() == _errInds.size());


  if (_finalT==_initT) { return ierr; }
  else if (_deltaT==0) { _deltaT = (_finalT-_initT)/_maxNumSteps; }
  if (_maxNumSteps == 0) { return ierr; }

  // set initial condition
  ierr = obj->d_dt(_currT,_var,_dvar);CHKERRQ(ierr);
  ierr = obj->timeMonitor(_currT,_deltaT,_stepCount,stopIntegration); CHKERRQ(ierr); // write first step



  while (_stepCount<_maxNumSteps && _currT<_finalT) {

    _stepCount++;
    attemptCount = 0;
    while (attemptCount<100) { // repeat until time step is acceptable
      attemptCount++;
      if (attemptCount>=100) {PetscPrintf(PETSC_COMM_WORLD,"   WARNING: maximum number of attempts reached\n"); }
      //~ierr = PetscPrintf(PETSC_COMM_WORLD,"   attemptCount=%i\n",attemptCount);CHKERRQ(ierr);
      if (_currT+_deltaT>_finalT) { _deltaT=_finalT-_currT; }

      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        VecSet(_k1[it->first],0.0); VecSet(_f1[it->first],0.0);
        VecSet(_k2[it->first],0.0);     VecSet(_f2[it->first],0.0);
        VecSet(_y2[it->first],0.0);
        VecSet(_y3[it->first],0.0);
      }

      // stage 1: integrate fields to _currT + 0.5*deltaT
      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        ierr = VecWAXPY(_k1[it->first],0.5*_deltaT,_dvar[it->first],_var[it->first]);CHKERRQ(ierr);
      }
      ierr = obj->d_dt(_currT+0.5*_deltaT,_k1,_f1);CHKERRQ(ierr);

      // stage 2: integrate fields to _currT + _deltaT
      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        ierr = VecWAXPY(_k2[it->first],-_deltaT,_dvar[it->first],_var[it->first]);CHKERRQ(ierr);
        ierr = VecAXPY(_k2[it->first],2*_deltaT,_f1[it->first]);CHKERRQ(ierr);
      }
      ierr = obj->d_dt(_currT+_deltaT,_k2,_f2);CHKERRQ(ierr);

      // 2nd and 3rd order update
      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        ierr = VecWAXPY(_y2[it->first],0.5*_deltaT,_dvar[it->first],_var[it->first]);CHKERRQ(ierr);
        ierr = VecAXPY(_y2[it->first],0.5*_deltaT,_f2[it->first]);CHKERRQ(ierr);

        ierr = VecWAXPY(_y3[it->first],_deltaT/6.0,_dvar[it->first],_var[it->first]);CHKERRQ(ierr);
        ierr = VecAXPY(_y3[it->first],2*_deltaT/3.0,_f1[it->first]);CHKERRQ(ierr);
        ierr = VecAXPY(_y3[it->first],_deltaT/6.0,_f2[it->first]);CHKERRQ(ierr);
      }

      // calculate error
      _totErr = computeError();
      if (_totErr <= _atol) { break; }
      _deltaT = computeStepSize(_totErr);
      if (_minDeltaT == _deltaT) { break; }

      _numRejectedSteps++;
    }

    // accept 3rd order solution as update
    _currT = _currT+_deltaT;
    for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
      VecSet(_var[it->first],0.0);
      ierr = VecCopy(_y3[it->first],_var[it->first]);CHKERRQ(ierr);
      VecSet(_dvar[it->first],0.0);
    }
    ierr = obj->d_dt(_currT,_var,_dvar);CHKERRQ(ierr);

    if (_totErr!=0.0) { _deltaT = computeStepSize(_totErr); }
    _errA.push_front(_totErr); // record error for use when estimating time step

    ierr = obj->timeMonitor(_currT,_deltaT,_stepCount,stopIntegration); CHKERRQ(ierr);

    if (stopIntegration > 0) { PetscPrintf(PETSC_COMM_WORLD,"RK32: Detected stop time integration request.\n"); break; }
  }

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32::integrate in odeSolver.cpp.\n");
#endif
  return ierr;
}


//======================================================================
//======================================================================

RK43::RK43(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType)
: OdeSolver(maxNumSteps,finalT,deltaT,controlType),
  _minDeltaT(0),_maxDeltaT(finalT),
  _atol(1e-9),_kappa(0.9),_ord(4.0),
  _numRejectedSteps(0),_numMinSteps(0),_numMaxSteps(0)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK43::constructor in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();

  _errA.resize(2);
  _errA.push_front(0);
  _errA.push_front(0);

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolver::constructor in odeSolver.cpp.\n");
#endif
}

RK43::~RK43()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK43::destructor in odeSolver.cpp.\n");
#endif

  // destruct temporary containers
  destroyVector(_f1);
  destroyVector(_f2);
  destroyVector(_f3);
  destroyVector(_f4);
  destroyVector(_f5);
  destroyVector(_f6);
  destroyVector(_k2);
  destroyVector(_k3);
  destroyVector(_k4);
  destroyVector(_k5);
  destroyVector(_k6);
  destroyVector(_y4);
  destroyVector(_y3);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending RK43::destructor in odeSolver.cpp.\n");
  #endif
}

PetscErrorCode RK43::view()
{
  PetscErrorCode ierr = 0;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTime Integration summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   integration algorithm: runge-kutta (4,3)\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   control scheme: %s\n",_controlType.c_str());CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   variables used in determining time step = %s\n",vector2str(_errInds).c_str());CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   scale factors = %s\n",vector2str(_scale).c_str());CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time interval: %g to %g\n",_initT,_finalT);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   permitted step size range: [%g,%g]\n",_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   total number of steps taken: %i/%i\n",_stepCount,_maxNumSteps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   final time reached: %g\n",_currT);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   tolerance: %g\n",_totTol);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of rejected steps: %i\n",_numRejectedSteps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of times min step size enforced: %i\n",_numMinSteps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of times max step size enforced: %i\n",_numMaxSteps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   total run time: %g\n",_runTime);CHKERRQ(ierr);

  return 0;
}

PetscErrorCode RK43::setTolerance(const PetscReal tol)
{
  _atol = tol;
  _rtol = tol;
  _totTol = tol;
  return 0;
}

PetscErrorCode RK43::setInitialConds(std::map<string,Vec>& var)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK43::setInitialConds in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  PetscErrorCode ierr = 0;

  _var = var;
  for (map<string,Vec>::iterator it=var.begin(); it!=var.end(); it++ ) {
    Vec dvar;
    ierr = VecDuplicate(_var[it->first],&dvar); CHKERRQ(ierr);
    ierr = VecSet(dvar,0.0); CHKERRQ(ierr);
    _dvar[it->first] = dvar;

    //~ Vec f1;
    //~ ierr = VecDuplicate(_var[it->first],&f1); CHKERRQ(ierr);
    //~ ierr = VecSet(f1,0.0); CHKERRQ(ierr);
    //~ _f1[it->first] = f1;

    Vec f2;
    ierr = VecDuplicate(_var[it->first],&f2); CHKERRQ(ierr);
    ierr = VecSet(f2,0.0); CHKERRQ(ierr);
    _f2[it->first] = f2;

    Vec f3;
    ierr = VecDuplicate(_var[it->first],&f3); CHKERRQ(ierr);
    ierr = VecSet(f3,0.0); CHKERRQ(ierr);
    _f3[it->first] = f3;

    Vec f4;
    ierr = VecDuplicate(_var[it->first],&f4); CHKERRQ(ierr);
    ierr = VecSet(f4,0.0); CHKERRQ(ierr);
    _f4[it->first] = f4;

    Vec f5;
    ierr = VecDuplicate(_var[it->first],&f5); CHKERRQ(ierr);
    ierr = VecSet(f5,0.0); CHKERRQ(ierr);
    _f5[it->first] = f5;

    Vec f6;
    ierr = VecDuplicate(_var[it->first],&f6); CHKERRQ(ierr);
    ierr = VecSet(f6,0.0); CHKERRQ(ierr);
    _f6[it->first] = f6;


    //~ Vec k1;
    //~ ierr = VecDuplicate(_var[it->first],&k1); CHKERRQ(ierr);
    //~ ierr = VecSet(k1,0.0); CHKERRQ(ierr);
    //~ _k1[it->first] = k1;

    Vec k2;
    ierr = VecDuplicate(_var[it->first],&k2); CHKERRQ(ierr);
    ierr = VecSet(k2,0.0); CHKERRQ(ierr);
    _k2[it->first] = k2;

    Vec k3;
    ierr = VecDuplicate(_var[it->first],&k3); CHKERRQ(ierr);
    ierr = VecSet(k3,0.0); CHKERRQ(ierr);
    _k3[it->first] = k3;

    Vec k4;
    ierr = VecDuplicate(_var[it->first],&k4); CHKERRQ(ierr);
    ierr = VecSet(k4,0.0); CHKERRQ(ierr);
    _k4[it->first] = k4;

    Vec k5;
    ierr = VecDuplicate(_var[it->first],&k5); CHKERRQ(ierr);
    ierr = VecSet(k5,0.0); CHKERRQ(ierr);
    _k5[it->first] = k5;

    Vec k6;
    ierr = VecDuplicate(_var[it->first],&k6); CHKERRQ(ierr);
    ierr = VecSet(k6,0.0); CHKERRQ(ierr);
    _k6[it->first] = k6;


    Vec y3;
    ierr = VecDuplicate(_var[it->first],&y3); CHKERRQ(ierr);
    ierr = VecSet(y3,0.0); CHKERRQ(ierr);
    _y3[it->first] = y3;

    Vec y4;
    ierr = VecDuplicate(_var[it->first],&y4); CHKERRQ(ierr);
    ierr = VecSet(y4,0.0); CHKERRQ(ierr);
    _y4[it->first] = y4;
  }

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK43::setInitialConds in odeSolver.cpp.\n");
#endif
  return ierr;
}

PetscErrorCode RK43::setErrInds(std::vector<string>& errInds) { _errInds = errInds; return 0; }

PetscErrorCode RK43::setErrInds(std::vector<string>& errInds, std::vector<double> scale)
{
  _errInds = errInds;
  _scale = scale;
  return 0;
}

PetscErrorCode RK43::setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK43::setTimeStepBounds in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  _minDeltaT = minDeltaT;
  _maxDeltaT = maxDeltaT;
  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK43::setTimeStepBounds in odeSolver.cpp.\n");
#endif
  return 0;
}

PetscReal RK43::computeStepSize(const PetscReal totErr)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK43::computeStepSize in odeSolver.cpp.\n");
#endif
  PetscReal stepRatio;

  if (_controlType.compare("P") == 0) {
    // if using integral feedback controller (I)
    PetscReal alpha = 1./(1.+_ord);
    stepRatio = _kappa*pow(_totTol/totErr,alpha);
  }
  else if (_controlType.compare("PID") == 0) {
    //if using proportional-integral-derivative feedback (PID)
    PetscReal alpha = 0.49/_ord;
    PetscReal beta  = 0.34/_ord;
    PetscReal gamma = 0.1/_ord;

    if (_stepCount < 4) {
      stepRatio = _kappa*pow(_totTol/totErr,1./(1.+_ord));
    }
    else {
      stepRatio = _kappa * pow(_totTol/totErr,alpha)
                         * pow(_errA[0]/_totTol,beta)
                         * pow(_totTol/_errA[1],gamma);
    }
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: timeControlType not understood\n");
    assert(0>1); // automatically fail, because I can't figure out how to use exit commands properly
  }
  PetscReal deltaT = stepRatio*_deltaT;

  // respect bounds on min and max possible step size
  deltaT = min(_deltaT*5.0,deltaT); // cap growth rate of step size
  deltaT= min(_maxDeltaT,deltaT); // absolute max
  deltaT = max(_minDeltaT,deltaT);

  if (_minDeltaT == deltaT) { _numMinSteps++; }
  else if (_maxDeltaT == deltaT) { _numMaxSteps++; }
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK43::computeStepSize in odeSolver.cpp.\n");
#endif
  return deltaT;
}

PetscReal RK43::computeError()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK43::computeError in odeSolver.cpp.\n");
#endif
  PetscErrorCode ierr = 0;
  PetscReal      err,_totErr=0.0;

  // if using absolute error for control
  // error: the absolute L2 error, weighted by N and a user-inputted scale factor
  // tolerance: the absolute tolerance
  if (_normType.compare("L2_absolute")==0) {
    for(std::vector<int>::size_type i = 0; i != _errInds.size(); i++) {
      std::string key = _errInds[i];
      Vec errVec; VecDuplicate(_y3[key],&errVec); VecSet(errVec,0.0);
      ierr = VecWAXPY(errVec,-1.0,_y4[key],_y3[key]); CHKERRQ(ierr);
      VecNorm(errVec,NORM_2,&err);
      VecDestroy(&errVec);

      PetscInt N = 0; VecGetSize(_y4[key],&N);
      _totErr += err / (sqrt(N) * _scale[i]);
    }
  }

  // if using relative error for control
  // error: the absolute L2 error, scaled by the L2 norm of the solution
  // and a user-inputted scale factor
  // tolerance: the relative tolerance
  if (_normType.compare("L2_relative")==0) {
    for(std::vector<int>::size_type i = 0; i != _errInds.size(); i++) {
      std::string key = _errInds[i];
      Vec errVec; VecDuplicate(_y3[key],&errVec); VecSet(errVec,0.0);
      ierr = VecWAXPY(errVec,-1.0,_y4[key],_y3[key]); CHKERRQ(ierr);
      VecNorm(errVec,NORM_2,&err);
      VecDestroy(&errVec);

      PetscReal s = 0; VecNorm(_y4[key],NORM_2,&s);
      _totErr += err / (s * _scale[i]);
    }
  }

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK43::computeError in odeSolver.cpp.\n");
#endif
  return _totErr;
}


PetscErrorCode RK43::integrate(IntegratorContextEx *obj)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK43::integrate in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();

  PetscErrorCode ierr=0;
  PetscReal      _totErr=0.0;
  PetscInt       attemptCount = 0;
  int            stopIntegration = 0;

  // coefficients
  //~ PetscScalar c1 = 0.;
  PetscScalar c2 = 1./2.;
  PetscScalar c3 = 83./250.;
  PetscScalar c4 = 31./50.;
  PetscScalar c5 = 17./20.;
  PetscScalar c6 = 1.;

  PetscScalar b1 = 82889./524892.;
  //~ PetscScalar b2 = 0.;
  PetscScalar b3 = 15625./83664.;
  PetscScalar b4 = 69875./102672.;
  PetscScalar b5 = -2260./8211.;
  PetscScalar b6 = 1./4.;


  PetscScalar hb1 = 4586570599./29645900160.;
  //~ PetscScalar hb2 = 0.;
  PetscScalar hb3 = 178811875./945068544.;
  PetscScalar hb4 = 814220225./1159782912.;
  PetscScalar hb5 = -3700637./11593932.;
  PetscScalar hb6 = 61727./225920.;

  PetscScalar a21 = 1./2.;

  PetscScalar a31 = 13861./62500.;
  PetscScalar a32 = 6889./62500.;

  PetscScalar a41 = -116923316275./ 2393684061468.;
  PetscScalar a42 = -2731218467317./15368042101831.;
  PetscScalar a43 = 9408046702089./11113171139209.;

  PetscScalar a51 = -451086348788./ 2902428689909.;
  PetscScalar a52 = -2682348792572./7519795681897.;
  PetscScalar a53 = 12662868775082./11960479115383.;
  PetscScalar a54 = 3355817975965./11060851509271.;

  PetscScalar a61 = 647845179188./3216320057751.;
  PetscScalar a62 = 73281519250./8382639484533.;
  PetscScalar a63 = 552539513391./3454668386233.;
  PetscScalar a64 = 3354512671639./8306763924573.;
  PetscScalar a65 = 4040./17871.;


  // build default errInds
  if (_errInds.size()==0) {
    for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
      _errInds.push_back(it->first);
    }
  }

  // check that errInds is valid
  for(std::vector<int>::size_type i = 0; i != _errInds.size(); i++) {
    std::string key = _errInds[i];
    if (_var.find(key) == _var.end()) {
      PetscPrintf(PETSC_COMM_WORLD,"ERROR: %s is not an element of explicitly integrated variable!\n",key.c_str());
    }
    assert(_var.find(key) != _var.end());
  }

  // set up scaling for elements in errInds
  if (_scale.size() == 0) { // if 0 entries, set all to 1
    for(std::vector<int>::size_type i = 0; i != _errInds.size(); i++) {
      _scale.push_back(1.0);
    }
  }
  assert(_scale.size() == _errInds.size());

  if (_finalT==_initT) { return ierr; }
  if (_deltaT==0) { _deltaT = (_finalT-_initT)/_maxNumSteps; }
  if (_maxNumSteps == 0) { return ierr; }

  // set initial condition
  ierr = obj->d_dt(_currT,_var,_dvar);CHKERRQ(ierr);
  _f1 = _dvar;
  ierr = obj->timeMonitor(_currT,_deltaT,_stepCount,stopIntegration); CHKERRQ(ierr); // write first step



  while (_stepCount<_maxNumSteps && _currT<_finalT) {
    _stepCount++;
    attemptCount = 0;
    while (attemptCount<100) { // repeat until time step is acceptable
      attemptCount++;
      if (attemptCount>=100) {PetscPrintf(PETSC_COMM_WORLD,"   WARNING: maximum number of attempts reached\n"); }

      if (_currT+_deltaT>_finalT) { _deltaT=_finalT-_currT; }

      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        VecSet(_k2[it->first],0.0);
        VecSet(_k3[it->first],0.0); VecSet(_k4[it->first],0.0);
        VecSet(_k5[it->first],0.0); VecSet(_k6[it->first],0.0);
        VecSet(_f2[it->first],0.0);
        VecSet(_f3[it->first],0.0); VecSet(_f4[it->first],0.0);
        VecSet(_f5[it->first],0.0); VecSet(_f6[it->first],0.0);
      }

      // stage 1: k1 = var, compute f1 = f(k1)
      _f1 = _dvar;

      // stage 2: compute k2
      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        ierr = VecWAXPY(_k2[it->first],a21*_deltaT,_f1[it->first],_var[it->first]); CHKERRQ(ierr);
      }
      ierr = obj->d_dt(_currT+c2*_deltaT,_k2,_f2);CHKERRQ(ierr); // compute f2

      // stage 3: compute k3
      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        ierr = VecWAXPY(_k3[it->first],a31*_deltaT,_f1[it->first],_var[it->first]); CHKERRQ(ierr);
        ierr = VecAXPY(_k3[it->first],a32*_deltaT,_f2[it->first]); CHKERRQ(ierr);
      }
      ierr = obj->d_dt(_currT+c3*_deltaT,_k3,_f3);CHKERRQ(ierr); // compute f3

      // stage 4
      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        ierr = VecWAXPY(_k4[it->first],a41*_deltaT,_f1[it->first],_var[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k4[it->first],a42*_deltaT,_f2[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k4[it->first],a43*_deltaT,_f3[it->first]); CHKERRQ(ierr);
      }
      ierr = obj->d_dt(_currT+c4*_deltaT,_k4,_f4);CHKERRQ(ierr); // compute f4

      // stage 5
      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        ierr = VecWAXPY(_k5[it->first],a51*_deltaT,_f1[it->first],_var[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k5[it->first],a52*_deltaT,_f2[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k5[it->first],a53*_deltaT,_f3[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k5[it->first],a54*_deltaT,_f4[it->first]); CHKERRQ(ierr);
      }
      ierr = obj->d_dt(_currT+c5*_deltaT,_k5,_f5);CHKERRQ(ierr); // compute f5

      // stage 6
      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        ierr = VecWAXPY(_k6[it->first],a61*_deltaT,_f1[it->first],_var[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k6[it->first],a62*_deltaT,_f2[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k6[it->first],a63*_deltaT,_f3[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k6[it->first],a64*_deltaT,_f4[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k6[it->first],a65*_deltaT,_f5[it->first]); CHKERRQ(ierr);
      }
      ierr = obj->d_dt(_currT+c6*_deltaT,_k6,_f6);CHKERRQ(ierr); // compute f6

      // 3rd and 4th order updates
      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        ierr = VecWAXPY(_y3[it->first],hb1*_deltaT,_f1[it->first],_var[it->first]); CHKERRQ(ierr);
        //~ ierr = VecAXPY(_y3[it->first],hb2*_deltaT,_f2[it->first]); CHKERRQ(ierr); // hb2 = 0
        ierr = VecAXPY(_y3[it->first],hb3*_deltaT,_f3[it->first]); CHKERRQ(ierr);
        ierr = VecAXPY(_y3[it->first],hb4*_deltaT,_f4[it->first]); CHKERRQ(ierr);
        ierr = VecAXPY(_y3[it->first],hb5*_deltaT,_f5[it->first]); CHKERRQ(ierr);
        ierr = VecAXPY(_y3[it->first],hb6*_deltaT,_f6[it->first]); CHKERRQ(ierr);


        ierr = VecWAXPY(_y4[it->first],b1*_deltaT,_f1[it->first],_var[it->first]); CHKERRQ(ierr);
        //~ ierr = VecAXPY(_y4[it->first],b2*_deltaT,_f2[it->first]); CHKERRQ(ierr); // b2 = 0
        ierr = VecAXPY(_y4[it->first],b3*_deltaT,_f3[it->first]); CHKERRQ(ierr);
        ierr = VecAXPY(_y4[it->first],b4*_deltaT,_f4[it->first]); CHKERRQ(ierr);
        ierr = VecAXPY(_y4[it->first],b5*_deltaT,_f5[it->first]); CHKERRQ(ierr);
        ierr = VecAXPY(_y4[it->first],b6*_deltaT,_f6[it->first]); CHKERRQ(ierr);
      }

      // calculate error
      _totErr = computeError();
      if (_totErr<_atol) { break; } // accept step
      _deltaT = computeStepSize(_totErr);
      if (_minDeltaT == _deltaT) { break; }

      _numRejectedSteps++;
    }

    // accept 4th order solution as update
    _currT = _currT+_deltaT;
    for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
      ierr = VecCopy(_y4[it->first],_var[it->first]);CHKERRQ(ierr);
      VecSet(_dvar[it->first],0.0);
    }
    ierr = obj->d_dt(_currT,_var,_dvar);CHKERRQ(ierr);
    ierr = obj->timeMonitor(_currT,_deltaT,_stepCount,stopIntegration); CHKERRQ(ierr);
    if (stopIntegration > 0) { PetscPrintf(PETSC_COMM_WORLD,"RK43: Detected stop time integration request.\n"); break; }

    if (_totErr!=0.0) { _deltaT = computeStepSize(_totErr); }
    _errA.push_front(_totErr); // record error for use when estimating time step
  }

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK43::integrate in odeSolver.cpp.\n");
#endif
  return ierr;
}




