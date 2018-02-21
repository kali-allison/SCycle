#include "odeSolverImex.hpp"

using namespace std;

OdeSolverImex::OdeSolverImex(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType)
: _initT(0),_finalT(finalT),_currT(0),_deltaT(deltaT),
  _maxNumSteps(maxNumSteps),_stepCount(0),
  _lenVar(0),_runTime(0),_controlType(controlType),_normType("L2_absolute"),
  _minDeltaT(0),_maxDeltaT(finalT),
  _atol(1e-9),
  _numRejectedSteps(0),_numMinSteps(0),_numMaxSteps(0)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolverImex constructor in RK32_WBE.cpp.\n");
#endif
  double startTime = MPI_Wtime();

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolverImex constructor in RK32_WBE.cpp.\n");
#endif
}

PetscErrorCode OdeSolverImex::setToleranceType(const std::string normType)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolverImex::setToleranceType in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  _normType = normType;
  assert(_normType.compare("L2_relative")==0 ||
      _normType.compare("L2_absolute")==0 ||
      _normType.compare("max")==0 );

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolverImex::setToleranceType in odeSolver.cpp.\n");
#endif
  return 0;
}


RK32_WBE::RK32_WBE(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType)
: OdeSolverImex(maxNumSteps,finalT,deltaT,controlType),
  _kappa(0.9),_ord(3.0)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32_WBE constructor in RK32_WBE.cpp.\n");
#endif
  double startTime = MPI_Wtime();

  _errA.resize(2);
  _errA.push_front(0);
  _errA.push_front(0);

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32_WBE constructor in RK32_WBE.cpp.\n");
#endif
}

RK32_WBE::~RK32_WBE()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32_WBE destructor in RK32_WBE.cpp.\n");
#endif

  // because I don't allocate the contents of _varEx, I don't delete them in this class either

  // destruct temporary containers
  destroyVector(_dvar);
  destroyVector(_varHalfdT);
  destroyVector(_dvarHalfdT);
  destroyVector(_vardT);
  destroyVector(_dvardT);
  destroyVector(_var2nd);
  destroyVector(_dvar2nd);
  destroyVector(_var3rd);


#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32_WBE destructor in RK32_WBE.cpp.\n");
#endif
}


PetscErrorCode RK32_WBE::setTimeRange(const PetscReal initT,const PetscReal finalT)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32_WBE::setTimeRange in RK32_WBE.cpp.\n");
#endif
  double startTime = MPI_Wtime();

  _initT = initT;
  _currT = initT;
  _finalT = finalT;

  _runTime += MPI_Wtime() - startTime;
  return 0;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32_WBE::setTimeRange in RK32_WBE.cpp.\n");
#endif
}

PetscErrorCode RK32_WBE::setStepSize(const PetscReal deltaT)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32_WBE::setStepSize in RK32_WBE.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  _deltaT = deltaT;
  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32_WBE::setStepSize in RK32_WBE.cpp.\n");
#endif
  return 0;
}


PetscErrorCode RK32_WBE::view()
{
  PetscErrorCode ierr = 0;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTime Integration summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   integration algorithm: IMEX runge-kutta (3,2)\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   control scheme: %s\n",_controlType.c_str());CHKERRQ(ierr);

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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);

  return 0;
}

PetscErrorCode RK32_WBE::setTolerance(const PetscReal tol)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32_WBE::setTolerance in RK32_WBE.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  _atol = tol;

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32_WBE::setTolerance in RK32_WBE.cpp.\n");
#endif
  return 0;
}

PetscErrorCode RK32_WBE::setInitialConds(std::map<string,Vec>& varEx,std::map<string,Vec>& varIm)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32_WBE::setInitialConds in RK32_WBE.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  PetscErrorCode ierr = 0;

  // explicit part
  _varEx = varEx;
  for (map<string,Vec>::iterator it=_varEx.begin(); it!=_varEx.end(); it++ ) {
    Vec dvar;
    ierr = VecDuplicate(_varEx[it->first],&dvar); CHKERRQ(ierr);
    ierr = VecSet(dvar,0.0); CHKERRQ(ierr);
    _dvar[it->first] = dvar;

    Vec varHalfdT;
    ierr = VecDuplicate(_varEx[it->first],&varHalfdT); CHKERRQ(ierr);
    ierr = VecSet(varHalfdT,0.0); CHKERRQ(ierr);
    _varHalfdT[it->first] = varHalfdT;

    Vec dvarHalfdT;
    ierr = VecDuplicate(_varEx[it->first],&dvarHalfdT); CHKERRQ(ierr);
    ierr = VecSet(dvarHalfdT,0.0); CHKERRQ(ierr);
    _dvarHalfdT[it->first] = dvarHalfdT;

    Vec vardT;
    ierr = VecDuplicate(_varEx[it->first],&vardT); CHKERRQ(ierr);
    ierr = VecSet(vardT,0.0); CHKERRQ(ierr);
    _vardT[it->first] = vardT;

    Vec dvardT;
    ierr = VecDuplicate(_varEx[it->first],&dvardT); CHKERRQ(ierr);
    ierr = VecSet(dvardT,0.0); CHKERRQ(ierr);
    _dvardT[it->first] = dvardT;

    Vec var2nd;
    ierr = VecDuplicate(_varEx[it->first],&var2nd); CHKERRQ(ierr);
    ierr = VecSet(var2nd,0.0); CHKERRQ(ierr);
    _var2nd[it->first] = var2nd;

    Vec dvar2nd;
    ierr = VecDuplicate(_varEx[it->first],&dvar2nd); CHKERRQ(ierr);
    ierr = VecSet(dvar2nd,0.0); CHKERRQ(ierr);
    _dvar2nd[it->first] = dvar2nd;

    Vec var3rd;
    ierr = VecDuplicate(_varEx[it->first],&var3rd); CHKERRQ(ierr);
    ierr = VecSet(var3rd,0.0); CHKERRQ(ierr);
    _var3rd[it->first] = var3rd;
  }

  // implicit part, computed once per time step
  _varIm = varIm;
  for (map<string,Vec>::iterator it=_varIm.begin(); it!=_varIm.end(); it++ ) {
    Vec vardTIm;
    ierr = VecDuplicate(_varIm[it->first],&vardTIm); CHKERRQ(ierr);
    ierr = VecSet(vardTIm,0.0); CHKERRQ(ierr);
    _vardTIm[it->first] = vardTIm;
  }

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32_WBE::setInitialConds in RK32_WBE.cpp.\n");
#endif
  return ierr;
}

PetscErrorCode RK32_WBE::setErrInds(std::vector<string>& errInds)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32_WBE::setErrInds in RK32_WBE.cpp.\n");
#endif
  _errInds = errInds;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32_WBE::setTimeStepBounds in RK32_WBE.cpp.\n");
#endif
  return 0;
}

PetscErrorCode RK32_WBE::setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32_WBE::setTimeStepBounds in RK32_WBE.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  _minDeltaT = minDeltaT;
  _maxDeltaT = maxDeltaT;
  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32_WBE::setTimeStepBounds in RK32_WBE.cpp.\n");
#endif
  return 0;
}

PetscReal RK32_WBE::computeStepSize(const PetscReal _totErr)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32_WBE::computeStepSize in RK32_WBE.cpp.\n");
#endif
  PetscReal stepRatio;

  if (_controlType.compare("P") == 0) {
    // if using integral feedback controller (I)
    PetscReal alpha = 1./(1.+_ord);
    stepRatio = _kappa*pow(_atol/_totErr,alpha);
  }
  else if (_controlType.compare("PID") == 0) {
    //if using proportional-integral-derivative feedback (PID)

    PetscReal alpha = 0.49/_ord;
    PetscReal beta  = 0.34/_ord;
    PetscReal gamma = 0.1/_ord;

    if (_stepCount < 3) {
      stepRatio = _kappa*pow(_atol/_totErr,1./(1.+_ord));
    }
    else {
      stepRatio = _kappa * pow(_atol/_totErr,alpha)
                         * pow(_errA[0]/_atol,beta)
                         * pow(_atol/_errA[1],gamma);
    }
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: timeControlType not understood\n");
    assert(0>1); // automatically fail
  }

  PetscReal deltaT = stepRatio*_deltaT;

  // respect bounds on min and max possible step size
  deltaT=min(_maxDeltaT,deltaT); // absolute max
  deltaT = min(_deltaT*5.0,deltaT); // cap growth rate of step size
  deltaT = max(_minDeltaT,deltaT);

  if (_minDeltaT == deltaT) {
    _numMinSteps++;
  }
  else if (_maxDeltaT == deltaT) {
    _numMaxSteps++;
  }
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32_WBE::computeStepSize in RK32_WBE.cpp.\n");
#endif
  return deltaT;
}

PetscReal RK32_WBE::computeError()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32_WBE::computeError in RK32_WBE.cpp.\n");
#endif
  PetscErrorCode ierr = 0;
  PetscReal      err,_totErr=0.0;


  if (_normType.compare("L2_relative")==0) { // relative error
    for(std::vector<int>::size_type i = 0; i != _errInds.size(); i++) {
      std::string key = _errInds[i];
      Vec errVec;
      PetscScalar size;
      VecDuplicate(_var3rd[key],&errVec);
      ierr = VecWAXPY(errVec,-1.0,_var3rd[key],_var2nd[key]);CHKERRQ(ierr);
      VecNorm(errVec,NORM_2,&err);
      VecNorm(_var3rd[key],NORM_2,&size);
      if (size <= 1e-14) {
        PetscInt N;
        VecGetSize(_var3rd[key],&N);
        _totErr += err/sqrt(N);
      }
      else { _totErr += err/(size); }
      VecDestroy(&errVec);
    }
    _totErr = _totErr * sqrt( (double) _errInds.size());
  }

  if (_normType.compare("L2_absolute")==0) { // weighted absolute error
    for(std::vector<int>::size_type i = 0; i != _errInds.size(); i++) {
      std::string key = _errInds[i];
      Vec errVec;
      PetscInt size;
      VecDuplicate(_var3rd[key],&errVec);
      ierr = VecWAXPY(errVec,-1.0,_var3rd[key],_var2nd[key]);CHKERRQ(ierr);
      VecNorm(errVec,NORM_2,&err);
      VecGetSize(_var3rd[key],&size);
      _totErr += err/sqrt(size);
      VecDestroy(&errVec);
    }
    _totErr = _totErr * sqrt( (double) _errInds.size());
  }

  if (_normType.compare("max")==0) { // max norm
    for(std::vector<int>::size_type i = 0; i != _errInds.size(); i++) {
      std::string key = _errInds[i];
      Vec errVec;
      PetscScalar size;
      VecDuplicate(_var3rd[key],&errVec);
      ierr = VecWAXPY(errVec,-1.0,_var3rd[key],_var2nd[key]);CHKERRQ(ierr);
      VecNorm(errVec,NORM_INFINITY,&err);
      _totErr = max(_totErr,err);
      VecDestroy(&errVec);
    }
  }


  /*for(std::vector<int>::size_type i = 0; i != _errInds.size(); i++) {
    std::string key = _errInds[i];

    // asbolute error based on weighted 2 norm
    Vec errVec;
    PetscScalar    size;
    VecDuplicate(_var3rd[key],&errVec);
    ierr = VecWAXPY(errVec,-1.0,_var3rd[key],_var2nd[key]);CHKERRQ(ierr);
    VecNorm(errVec,NORM_2,&err);
    VecNorm(_var3rd[key],NORM_2,&size);
    _totErr += err/(size+1.0);
    VecDestroy(&errVec);
  }*/



#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32_WBE::computeError in RK32_WBE.cpp.\n");
#endif
  return _totErr;
}




PetscErrorCode RK32_WBE::integrate(IntegratorContextImex *obj)
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
    for (map<string,Vec>::iterator it = _varEx.begin(); it!=_varEx.end(); it++ ) {
      _errInds.push_back(it->first);
    }
  }

  // check that errInds is valid
  for(std::vector<int>::size_type i = 0; i != _errInds.size(); i++) {
    std::string key = _errInds[i];
    if (_varEx.find(key) == _varEx.end()) {
      PetscPrintf(PETSC_COMM_WORLD,"ERROR: %s is not an element of explicitly integrated variable!\n",key.c_str());
    }
    assert(_varEx.find(key) != _varEx.end());
  }


  // set initial condition
  ierr = obj->d_dt(_currT,_varEx,_dvar);CHKERRQ(ierr);
  ierr = obj->timeMonitor(_currT,_stepCount,_varEx,_dvar,_varIm,stopIntegration);CHKERRQ(ierr);// write first step

  if (_finalT==_initT) { return ierr; }
  else if (_deltaT==0) { _deltaT = (_finalT-_initT)/_maxNumSteps; }
  if (_maxNumSteps == 0) { return ierr; }

  while (_stepCount<_maxNumSteps && _currT<_finalT) {

    _stepCount++;
    attemptCount = 0;
    while (attemptCount<100) { // repeat until time step is acceptable
      attemptCount++;
      if (attemptCount>=100) {PetscPrintf(PETSC_COMM_WORLD,"   WARNING: maximum number of attempts reached\n"); }
      //~ierr = PetscPrintf(PETSC_COMM_WORLD,"   attemptCount=%i\n",attemptCount);CHKERRQ(ierr);
      if (_currT+_deltaT>_finalT) { _deltaT=_finalT-_currT; }

      for (map<string,Vec>::iterator it = _varEx.begin(); it!=_varEx.end(); it++ ) {
        VecSet(_varHalfdT[it->first],0.0); VecSet(_dvarHalfdT[it->first],0.0);
        VecSet(_vardT[it->first],0.0);     VecSet(_dvardT[it->first],0.0);
        VecSet(_var2nd[it->first],0.0);    VecSet(_dvar2nd[it->first],0.0);
        VecSet(_var3rd[it->first],0.0);
      }

      // stage 1: integrate fields to _currT + 0.5*deltaT
      for (map<string,Vec>::iterator it = _varEx.begin(); it!=_varEx.end(); it++ ) {
        ierr = VecWAXPY(_varHalfdT[it->first],0.5*_deltaT,_dvar[it->first],_varEx[it->first]);CHKERRQ(ierr);
      }
      ierr = obj->d_dt(_currT+0.5*_deltaT,_varHalfdT,_dvarHalfdT);CHKERRQ(ierr);

      // stage 2: integrate fields to _currT + _deltaT
      for (map<string,Vec>::iterator it = _varEx.begin(); it!=_varEx.end(); it++ ) {
        ierr = VecWAXPY(_vardT[it->first],-_deltaT,_dvar[it->first],_varEx[it->first]);CHKERRQ(ierr);
        ierr = VecAXPY(_vardT[it->first],2*_deltaT,_dvarHalfdT[it->first]);CHKERRQ(ierr);
      }
      ierr = obj->d_dt(_currT+_deltaT,_vardT,_dvardT);CHKERRQ(ierr);

      // 2nd and 3rd order update
      for (map<string,Vec>::iterator it = _varEx.begin(); it!=_varEx.end(); it++ ) {
        ierr = VecWAXPY(_var2nd[it->first],0.5*_deltaT,_dvar[it->first],_varEx[it->first]);CHKERRQ(ierr);
        ierr = VecAXPY(_var2nd[it->first],0.5*_deltaT,_dvardT[it->first]);CHKERRQ(ierr);

        ierr = VecWAXPY(_var3rd[it->first],_deltaT/6.0,_dvar[it->first],_varEx[it->first]);CHKERRQ(ierr);
        ierr = VecAXPY(_var3rd[it->first],2*_deltaT/3.0,_dvarHalfdT[it->first]);CHKERRQ(ierr);
        ierr = VecAXPY(_var3rd[it->first],_deltaT/6.0,_dvardT[it->first]);CHKERRQ(ierr);
      }

      // calculate error
      _totErr = computeError();
      if (_totErr<_atol) { break; } // !!!orig
      _deltaT = computeStepSize(_totErr);
      if (_minDeltaT == _deltaT) { break; }

      _numRejectedSteps++;
    }
    _currT = _currT+_deltaT;

    // accept 3rd order solution as update
    for (map<string,Vec>::iterator it = _varEx.begin(); it!=_varEx.end(); it++ ) {
      VecSet(_varEx[it->first],0.0);
      ierr = VecCopy(_var3rd[it->first],_varEx[it->first]);CHKERRQ(ierr);
      VecSet(_dvar[it->first],0.0);
    }

    // update rates for explicit variables, and compute updated state for implicit variables
    ierr = obj->d_dt(_currT,_varEx,_dvar,_vardTIm,_varIm,_deltaT);CHKERRQ(ierr);

    // accept updated state for implicit variables
    for (map<string,Vec>::iterator it = _vardTIm.begin(); it!=_vardTIm.end(); it++ ) {
      VecCopy(_vardTIm[it->first],_varIm[it->first]);
    }

    if (_totErr!=0.0) { _deltaT = computeStepSize(_totErr); }
    _errA.push_front(_totErr); // record error for use when estimating time step

    ierr = obj->timeMonitor(_currT,_stepCount,_varEx,_dvar,_varIm,stopIntegration); CHKERRQ(ierr);
  }

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32_WBE::integrate in odeSolver.cpp.\n");
#endif
  return ierr;
}


//======================================================================
//======================================================================

RK43_WBE::RK43_WBE(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType)
: OdeSolverImex(maxNumSteps,finalT,deltaT,controlType),
  _kappa(0.9),_ord(4.0)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK43_WBE constructor in RK32_WBE.cpp.\n");
#endif
  double startTime = MPI_Wtime();

  _errA.resize(2);
  _errA.push_front(0);
  _errA.push_front(0);

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK43_WBE constructor in RK32_WBE.cpp.\n");
#endif
}

RK43_WBE::~RK43_WBE()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK43_WBE destructor in RK32_WBE.cpp.\n");
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
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK43_WBE destructor in RK32_WBE.cpp.\n");
#endif
}


PetscErrorCode RK43_WBE::setTimeRange(const PetscReal initT,const PetscReal finalT)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK43_WBE::setTimeRange in RK32_WBE.cpp.\n");
#endif
  double startTime = MPI_Wtime();

  _initT = initT;
  _currT = initT;
  _finalT = finalT;

  _runTime += MPI_Wtime() - startTime;
  return 0;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK43_WBE::setTimeRange in RK32_WBE.cpp.\n");
#endif
}

PetscErrorCode RK43_WBE::setStepSize(const PetscReal deltaT)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK43_WBE::setStepSize in RK32_WBE.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  _deltaT = deltaT;
  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK43_WBE::setStepSize in RK32_WBE.cpp.\n");
#endif
  return 0;
}


PetscErrorCode RK43_WBE::view()
{
  PetscErrorCode ierr = 0;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTime Integration summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   integration algorithm: IMEX runge-kutta (3,2)\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   control scheme: %s\n",_controlType.c_str());CHKERRQ(ierr);

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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);

  return 0;
}

PetscErrorCode RK43_WBE::setTolerance(const PetscReal tol)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK43_WBE::setTolerance in RK32_WBE.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  _atol = tol;

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK43_WBE::setTolerance in RK32_WBE.cpp.\n");
#endif
  return 0;
}

PetscErrorCode RK43_WBE::setInitialConds(std::map<string,Vec>& varEx,std::map<string,Vec>& varIm)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK43_WBE::setInitialConds in RK43_WBE.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  PetscErrorCode ierr = 0;

  // explicit part
  _varEx = varEx;
  for (map<string,Vec>::iterator it=varEx.begin(); it!=varEx.end(); it++ ) {
    Vec dvar;
    ierr = VecDuplicate(_varEx[it->first],&dvar); CHKERRQ(ierr);
    ierr = VecSet(dvar,0.0); CHKERRQ(ierr);
    _dvar[it->first] = dvar;

    //~ Vec f1;
    //~ ierr = VecDuplicate(_varEx[it->first],&f1); CHKERRQ(ierr);
    //~ ierr = VecSet(f1,0.0); CHKERRQ(ierr);
    //~ _f1[it->first] = f1;

    Vec f2;
    ierr = VecDuplicate(_varEx[it->first],&f2); CHKERRQ(ierr);
    ierr = VecSet(f2,0.0); CHKERRQ(ierr);
    _f2[it->first] = f2;

    Vec f3;
    ierr = VecDuplicate(_varEx[it->first],&f3); CHKERRQ(ierr);
    ierr = VecSet(f3,0.0); CHKERRQ(ierr);
    _f3[it->first] = f3;

    Vec f4;
    ierr = VecDuplicate(_varEx[it->first],&f4); CHKERRQ(ierr);
    ierr = VecSet(f4,0.0); CHKERRQ(ierr);
    _f4[it->first] = f4;

    Vec f5;
    ierr = VecDuplicate(_varEx[it->first],&f5); CHKERRQ(ierr);
    ierr = VecSet(f5,0.0); CHKERRQ(ierr);
    _f5[it->first] = f5;

    Vec f6;
    ierr = VecDuplicate(_varEx[it->first],&f6); CHKERRQ(ierr);
    ierr = VecSet(f6,0.0); CHKERRQ(ierr);
    _f6[it->first] = f6;


    //~ Vec k1;
    //~ ierr = VecDuplicate(_varEx[it->first],&k1); CHKERRQ(ierr);
    //~ ierr = VecSet(k1,0.0); CHKERRQ(ierr);
    //~ _k1[it->first] = k1;

    Vec k2;
    ierr = VecDuplicate(_varEx[it->first],&k2); CHKERRQ(ierr);
    ierr = VecSet(k2,0.0); CHKERRQ(ierr);
    _k2[it->first] = k2;

    Vec k3;
    ierr = VecDuplicate(_varEx[it->first],&k3); CHKERRQ(ierr);
    ierr = VecSet(k3,0.0); CHKERRQ(ierr);
    _k3[it->first] = k3;

    Vec k4;
    ierr = VecDuplicate(_varEx[it->first],&k4); CHKERRQ(ierr);
    ierr = VecSet(k4,0.0); CHKERRQ(ierr);
    _k4[it->first] = k4;

    Vec k5;
    ierr = VecDuplicate(_varEx[it->first],&k5); CHKERRQ(ierr);
    ierr = VecSet(k5,0.0); CHKERRQ(ierr);
    _k5[it->first] = k5;

    Vec k6;
    ierr = VecDuplicate(_varEx[it->first],&k6); CHKERRQ(ierr);
    ierr = VecSet(k6,0.0); CHKERRQ(ierr);
    _k6[it->first] = k6;


    Vec y3;
    ierr = VecDuplicate(_varEx[it->first],&y3); CHKERRQ(ierr);
    ierr = VecSet(y3,0.0); CHKERRQ(ierr);
    _y3[it->first] = y3;

    Vec y4;
    ierr = VecDuplicate(_varEx[it->first],&y4); CHKERRQ(ierr);
    ierr = VecSet(y4,0.0); CHKERRQ(ierr);
    _y4[it->first] = y4;
  }

  // implicit part, computed once per time step
  _varIm = varIm;
  for (map<string,Vec>::iterator it=_varIm.begin(); it!=_varIm.end(); it++ ) {
    Vec vardTIm;
    ierr = VecDuplicate(_varIm[it->first],&vardTIm); CHKERRQ(ierr);
    ierr = VecSet(vardTIm,0.0); CHKERRQ(ierr);
    _vardTIm[it->first] = vardTIm;
  }

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32_WBE::setInitialConds in RK32_WBE.cpp.\n");
#endif
  return ierr;
}

PetscErrorCode RK43_WBE::setErrInds(std::vector<string>& errInds)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK43_WBE::setErrInds in RK32_WBE.cpp.\n");
#endif
  _errInds = errInds;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK43_WBE::setTimeStepBounds in RK32_WBE.cpp.\n");
#endif
  return 0;
}

PetscErrorCode RK43_WBE::setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK43_WBE::setTimeStepBounds in RK32_WBE.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  _minDeltaT = minDeltaT;
  _maxDeltaT = maxDeltaT;
  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK43_WBE::setTimeStepBounds in RK32_WBE.cpp.\n");
#endif
  return 0;
}

PetscReal RK43_WBE::computeStepSize(const PetscReal _totErr)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK43_WBE::computeStepSize in RK32_WBE.cpp.\n");
#endif
  PetscReal stepRatio;

  if (_controlType.compare("P") == 0) {
    // if using integral feedback controller (I)
    PetscReal alpha = 1./(1.+_ord);
    stepRatio = _kappa*pow(_atol/_totErr,alpha);
  }
  else if (_controlType.compare("PID") == 0) {
    //if using proportional-integral-derivative feedback (PID)

    PetscReal alpha = 0.49/_ord;
    PetscReal beta  = 0.34/_ord;
    PetscReal gamma = 0.1/_ord;

    if (_stepCount < 3) {
      stepRatio = _kappa*pow(_atol/_totErr,1./(1.+_ord));
    }
    else {
      stepRatio = _kappa * pow(_atol/_totErr,alpha)
                         * pow(_errA[0]/_atol,beta)
                         * pow(_atol/_errA[1],gamma);
    }
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: timeControlType not understood\n");
    assert(0>1); // automatically fail
  }

  PetscReal deltaT = stepRatio*_deltaT;

  // respect bounds on min and max possible step size
  deltaT=min(_maxDeltaT,deltaT); // absolute max
  deltaT = min(_deltaT*5.0,deltaT); // cap growth rate of step size
  deltaT = max(_minDeltaT,deltaT);

  if (_minDeltaT == deltaT) {
    _numMinSteps++;
  }
  else if (_maxDeltaT == deltaT) {
    _numMaxSteps++;
  }
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK43_WBE::computeStepSize in RK32_WBE.cpp.\n");
#endif
  return deltaT;
}

PetscReal RK43_WBE::computeError()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK43_WBE::computeError in RK32_WBE.cpp.\n");
#endif
  PetscErrorCode ierr = 0;
  PetscReal      err,_totErr=0.0;

  if (_normType.compare("L2_relative")==0) { // relative error
    for(std::vector<int>::size_type i = 0; i != _errInds.size(); i++) {
      std::string key = _errInds[i];
      Vec errVec;
      PetscScalar size;
      VecDuplicate(_y4[key],&errVec);
      ierr = VecWAXPY(errVec,-1.0,_y4[key],_y3[key]);CHKERRQ(ierr);
      VecNorm(errVec,NORM_2,&err);
      VecNorm(_y4[key],NORM_2,&size);
      if (size <= 1e-14) {
        PetscInt N;
        VecGetSize(_y3[key],&N);
        _totErr += err/sqrt(N);
      }
      else { _totErr += err/(size); }
      VecDestroy(&errVec);
    }
    _totErr = _totErr * sqrt( (double) _errInds.size());
  }

  if (_normType.compare("L2_absolute")==0) { // weighted absolute error
    for(std::vector<int>::size_type i = 0; i != _errInds.size(); i++) {
      std::string key = _errInds[i];
      Vec errVec;
      PetscInt size;
      VecDuplicate(_y4[key],&errVec);
      ierr = VecWAXPY(errVec,-1.0,_y4[key],_y3[key]);CHKERRQ(ierr);
      VecNorm(errVec,NORM_2,&err);
      VecGetSize(_y4[key],&size);
      _totErr += err/sqrt(size);
      VecDestroy(&errVec);
    }
    _totErr = _totErr * sqrt( (double) _errInds.size());
  }

  if (_normType.compare("max")==0) { // max norm
    for(std::vector<int>::size_type i = 0; i != _errInds.size(); i++) {
      std::string key = _errInds[i];
      Vec errVec;
      PetscScalar size;
      VecDuplicate(_y4[key],&errVec);
      ierr = VecWAXPY(errVec,-1.0,_y4[key],_y3[key]);CHKERRQ(ierr);
      VecNorm(errVec,NORM_INFINITY,&err);
      _totErr = max(_totErr,err);
      VecDestroy(&errVec);
    }
  }


  /*for(std::vector<int>::size_type i = 0; i != _errInds.size(); i++) {
    std::string key = _errInds[i];

    // asbolute error based on weighted 2 norm
    Vec errVec;
    PetscScalar    size;
    VecDuplicate(_y4[key],&errVec);
    ierr = VecWAXPY(errVec,-1.0,_y4[key],_y3[key]);CHKERRQ(ierr);
    VecNorm(errVec,NORM_2,&err);
    VecNorm(_y4[key],NORM_2,&size);
    _totErr += err/(size+1.0);
    VecDestroy(&errVec);
  }*/

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK43_WBE::computeError in RK32_WBE.cpp.\n");
#endif
  return _totErr;
}




PetscErrorCode RK43_WBE::integrate(IntegratorContextImex *obj)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK43_WBE::integrate in odeSolver.cpp.\n");
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
    for (map<string,Vec>::iterator it = _varEx.begin(); it!=_varEx.end(); it++ ) {
      _errInds.push_back(it->first);
    }
  }

  // check that errInds is valid
  for(std::vector<int>::size_type i = 0; i != _errInds.size(); i++) {
    std::string key = _errInds[i];
    if (_varEx.find(key) == _varEx.end()) {
      PetscPrintf(PETSC_COMM_WORLD,"ERROR: %s is not an explicitly integrated variable!\n",key.c_str());
    }
    assert(_varEx.find(key) != _varEx.end());
  }


  // set initial condition
  ierr = obj->d_dt(_currT,_varEx,_dvar);CHKERRQ(ierr);
  _f1 = _dvar;
  ierr = obj->timeMonitor(_currT,_stepCount,_varEx,_dvar,_varIm,stopIntegration); CHKERRQ(ierr); // write first step

  if (_finalT==_initT) { return ierr; }
  if (_deltaT==0) { _deltaT = (_finalT-_initT)/_maxNumSteps; }
  if (_maxNumSteps == 0) { return ierr; }

  while (_stepCount<_maxNumSteps && _currT<_finalT) {
    _stepCount++;
    attemptCount = 0;
    while (attemptCount<100) { // repeat until time step is acceptable
      attemptCount++;
      if (attemptCount>=100) {PetscPrintf(PETSC_COMM_WORLD,"   WARNING: maximum number of attempts reached\n"); }

      if (_currT+_deltaT>_finalT) { _deltaT=_finalT-_currT; }

      for (map<string,Vec>::iterator it = _varEx.begin(); it!=_varEx.end(); it++ ) {
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
      for (map<string,Vec>::iterator it = _varEx.begin(); it!=_varEx.end(); it++ ) {
        ierr = VecWAXPY(_k2[it->first],a21*_deltaT,_f1[it->first],_varEx[it->first]); CHKERRQ(ierr);
      }
      ierr = obj->d_dt(_currT+c2*_deltaT,_k2,_f2);CHKERRQ(ierr); // compute f2

      // stage 3: compute k3
      for (map<string,Vec>::iterator it = _varEx.begin(); it!=_varEx.end(); it++ ) {
        ierr = VecWAXPY(_k3[it->first],a31*_deltaT,_f1[it->first],_varEx[it->first]); CHKERRQ(ierr);
        ierr = VecAXPY(_k3[it->first],a32*_deltaT,_f2[it->first]); CHKERRQ(ierr);
      }
      ierr = obj->d_dt(_currT+c3*_deltaT,_k3,_f3);CHKERRQ(ierr); // compute f3

      // stage 4
      for (map<string,Vec>::iterator it = _varEx.begin(); it!=_varEx.end(); it++ ) {
        ierr = VecWAXPY(_k4[it->first],a41*_deltaT,_f1[it->first],_varEx[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k4[it->first],a42*_deltaT,_f2[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k4[it->first],a43*_deltaT,_f3[it->first]); CHKERRQ(ierr);
      }
      ierr = obj->d_dt(_currT+c4*_deltaT,_k4,_f4);CHKERRQ(ierr); // compute f4

      // stage 5
      for (map<string,Vec>::iterator it = _varEx.begin(); it!=_varEx.end(); it++ ) {
        ierr = VecWAXPY(_k5[it->first],a51*_deltaT,_f1[it->first],_varEx[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k5[it->first],a52*_deltaT,_f2[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k5[it->first],a53*_deltaT,_f3[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k5[it->first],a54*_deltaT,_f4[it->first]); CHKERRQ(ierr);
      }
      ierr = obj->d_dt(_currT+c5*_deltaT,_k5,_f5);CHKERRQ(ierr); // compute f5

      // stage 6
      for (map<string,Vec>::iterator it = _varEx.begin(); it!=_varEx.end(); it++ ) {
        ierr = VecWAXPY(_k6[it->first],a61*_deltaT,_f1[it->first],_varEx[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k6[it->first],a62*_deltaT,_f2[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k6[it->first],a63*_deltaT,_f3[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k6[it->first],a64*_deltaT,_f4[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k6[it->first],a65*_deltaT,_f5[it->first]); CHKERRQ(ierr);
      }
      ierr = obj->d_dt(_currT+c6*_deltaT,_k6,_f6);CHKERRQ(ierr); // compute f6

      // 3rd and 4th order updates
      for (map<string,Vec>::iterator it = _varEx.begin(); it!=_varEx.end(); it++ ) {
        ierr = VecWAXPY(_y3[it->first],hb1*_deltaT,_f1[it->first],_varEx[it->first]); CHKERRQ(ierr);
        //~ ierr = VecAXPY(_y3[it->first],hb2*_deltaT,_f2[it->first]); CHKERRQ(ierr); // hb2 = 0
        ierr = VecAXPY(_y3[it->first],hb3*_deltaT,_f3[it->first]); CHKERRQ(ierr);
        ierr = VecAXPY(_y3[it->first],hb4*_deltaT,_f4[it->first]); CHKERRQ(ierr);
        ierr = VecAXPY(_y3[it->first],hb5*_deltaT,_f5[it->first]); CHKERRQ(ierr);
        ierr = VecAXPY(_y3[it->first],hb6*_deltaT,_f6[it->first]); CHKERRQ(ierr);


        ierr = VecWAXPY(_y4[it->first],b1*_deltaT,_f1[it->first],_varEx[it->first]); CHKERRQ(ierr);
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
    _currT = _currT+_deltaT;

    // accept 4th order solution as update
    for (map<string,Vec>::iterator it = _varEx.begin(); it!=_varEx.end(); it++ ) {
      ierr = VecCopy(_y4[it->first],_varEx[it->first]);CHKERRQ(ierr);
      VecSet(_dvar[it->first],0.0);
    }
    // update rates for explicit variables, and compute updated state for implicit variables
    ierr = obj->d_dt(_currT,_varEx,_dvar,_vardTIm,_varIm,_deltaT);CHKERRQ(ierr);

    // accept updated state for implicit variables
    for (map<string,Vec>::iterator it = _vardTIm.begin(); it!=_vardTIm.end(); it++ ) {
      VecCopy(_vardTIm[it->first],_varIm[it->first]);
    }
    ierr = obj->timeMonitor(_currT,_stepCount,_varEx,_dvar,_varIm,stopIntegration); CHKERRQ(ierr);
    if (stopIntegration > 0) { PetscPrintf(PETSC_COMM_WORLD,"RK43_WBE: Detected stop time integration request.\n"); break; }

    if (_totErr!=0.0) { _deltaT = computeStepSize(_totErr); }
    _errA.push_front(_totErr); // record error for use when estimating time step
  }

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK43_WBE::integrate in odeSolver.cpp.\n");
#endif
  return ierr;
}

