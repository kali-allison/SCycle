#include "odeSolverImex.hpp"

using namespace std;

OdeSolverImex::OdeSolverImex(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType)
: _initT(0),_finalT(finalT),_currT(0),_deltaT(deltaT),
  _maxNumSteps(maxNumSteps),_stepCount(0),
  _lenVar(0),_runTime(0),_controlType(controlType),
  _minDeltaT(0),_maxDeltaT(finalT),
  _atol(1e-9),_kappa(0.9),_ord(2.0),
  _numRejectedSteps(0),_numMinSteps(0),_numMaxSteps(0)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolverImex constructor in OdeSolverImex.cpp.\n");
#endif
  double startTime = MPI_Wtime();

  _absErr[0] = 0;_absErr[1] = 0;_absErr[2] = 0;

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolverImex constructor in OdeSolverImex.cpp.\n");
#endif
}

OdeSolverImex::~OdeSolverImex()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolverImex destructor in OdeSolverImex.cpp.\n");
#endif

  // because I don't allocate the contents of _var, I don't delete them in this class either

  // destruct temporary containers
  destroyVector(_dvar);
  destroyVector(_varHalfdT);
  destroyVector(_dvarHalfdT);
  destroyVector(_vardT);
  destroyVector(_dvardT);
  destroyVector(_var2nd);
  destroyVector(_dvar2nd);
  destroyVector(_var3rd);

  destroyVector(_varHalfdTIm);
  destroyVector(_vardTIm);
  destroyVector(_varIm_half);

#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolverImex destructor in OdeSolverImex.cpp.\n");
#endif
}


PetscErrorCode OdeSolverImex::setTimeRange(const PetscReal initT,const PetscReal finalT)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolverImex::setTimeRange in OdeSolverImex.cpp.\n");
#endif
  double startTime = MPI_Wtime();

  _initT = initT;
  _currT = initT;
  _finalT = finalT;

  _runTime += MPI_Wtime() - startTime;
  return 0;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolverImex::setTimeRange in OdeSolverImex.cpp.\n");
#endif
}

PetscErrorCode OdeSolverImex::setStepSize(const PetscReal deltaT)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolverImex::setStepSize in OdeSolverImex.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  _deltaT = deltaT;
  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolverImex::setStepSize in OdeSolverImex.cpp.\n");
#endif
  return 0;
}


PetscErrorCode OdeSolverImex::view()
{
  PetscErrorCode ierr = 0;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTime Integration summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   integration algorithm: IMEX runge-kutta (3,2)\n");CHKERRQ(ierr);
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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);

  return 0;
}

PetscErrorCode OdeSolverImex::setTolerance(const PetscReal tol)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolverImex::setTolerance in OdeSolverImex.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  _atol = tol;

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolverImex::setTolerance in OdeSolverImex.cpp.\n");
#endif
  return 0;
}

PetscErrorCode OdeSolverImex::setInitialConds(std::map<string,Vec>& varEx,std::map<string,Vec>& varIm)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolverImex::setInitialConds in OdeSolverImex.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  PetscErrorCode ierr = 0;

  // explicit part
  _var = varEx;
  for (map<string,Vec>::iterator it=_var.begin(); it!=_var.end(); it++ ) {
    Vec dvar;
    ierr = VecDuplicate(_var[it->first],&dvar); CHKERRQ(ierr);
    ierr = VecSet(dvar,0.0); CHKERRQ(ierr);
    _dvar[it->first] = dvar;

    Vec varHalfdT;
    ierr = VecDuplicate(_var[it->first],&varHalfdT); CHKERRQ(ierr);
    ierr = VecSet(varHalfdT,0.0); CHKERRQ(ierr);
    _varHalfdT[it->first] = varHalfdT;

    Vec dvarHalfdT;
    ierr = VecDuplicate(_var[it->first],&dvarHalfdT); CHKERRQ(ierr);
    ierr = VecSet(dvarHalfdT,0.0); CHKERRQ(ierr);
    _dvarHalfdT[it->first] = dvarHalfdT;

    Vec vardT;
    ierr = VecDuplicate(_var[it->first],&vardT); CHKERRQ(ierr);
    ierr = VecSet(vardT,0.0); CHKERRQ(ierr);
    _vardT[it->first] = vardT;

    Vec dvardT;
    ierr = VecDuplicate(_var[it->first],&dvardT); CHKERRQ(ierr);
    ierr = VecSet(dvardT,0.0); CHKERRQ(ierr);
    _dvardT[it->first] = dvardT;

    Vec var2nd;
    ierr = VecDuplicate(_var[it->first],&var2nd); CHKERRQ(ierr);
    ierr = VecSet(var2nd,0.0); CHKERRQ(ierr);
    _var2nd[it->first] = var2nd;

    Vec dvar2nd;
    ierr = VecDuplicate(_var[it->first],&dvar2nd); CHKERRQ(ierr);
    ierr = VecSet(dvar2nd,0.0); CHKERRQ(ierr);
    _dvar2nd[it->first] = dvar2nd;

    Vec var3rd;
    ierr = VecDuplicate(_var[it->first],&var3rd); CHKERRQ(ierr);
    ierr = VecSet(var3rd,0.0); CHKERRQ(ierr);
    _var3rd[it->first] = var3rd;
  }

  // implicit part
  _varIm = varIm;
  for (map<string,Vec>::iterator it=_varIm.begin(); it!=_varIm.end(); it++ ) {

    Vec varHalfdTIm;
    ierr = VecDuplicate(_varIm[it->first],&varHalfdTIm); CHKERRQ(ierr);
    ierr = VecSet(varHalfdTIm,0.0); CHKERRQ(ierr);
    _varHalfdTIm[it->first] = varHalfdTIm;

    Vec vardTIm;
    ierr = VecDuplicate(_varIm[it->first],&vardTIm); CHKERRQ(ierr);
    ierr = VecSet(vardTIm,0.0); CHKERRQ(ierr);
    _vardTIm[it->first] = vardTIm;

    Vec varIm_half;
    ierr = VecDuplicate(_varIm[it->first],&varIm_half); CHKERRQ(ierr);
    ierr = VecSet(varIm_half,0.0); CHKERRQ(ierr);
    _varIm_half[it->first] = varIm_half;
  }

  //~ Vec temp;
  //~ VecDuplicate(*varIm.begin(),&temp);
  //~ VecSet(temp,0.0);
  //~ _varHalfdTIm.push_back(temp);

  //~ Vec temp1;
  //~ VecDuplicate(*varIm.begin(),&temp1);
  //~ VecSet(temp1,0.0);
  //~ _vardTIm.push_back(temp1);

  //~ Vec temp2;
  //~ VecDuplicate(*varIm.begin(),&temp2);
  //~ VecSet(temp2,0.0);
  //~ _varIm_half.push_back(temp2);

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolverImex::setInitialConds in OdeSolverImex.cpp.\n");
#endif
  return ierr;
}

PetscErrorCode OdeSolverImex::setErrInds(std::vector<string>& errInds)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolverImex::setErrInds in OdeSolverImex.cpp.\n");
#endif
  _errInds = errInds;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolverImex::setTimeStepBounds in OdeSolverImex.cpp.\n");
#endif
  return 0;
}

PetscErrorCode OdeSolverImex::setTimeStepBounds(const PetscReal minDeltaT, const PetscReal maxDeltaT)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolverImex::setTimeStepBounds in OdeSolverImex.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  _minDeltaT = minDeltaT;
  _maxDeltaT = maxDeltaT;
  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolverImex::setTimeStepBounds in OdeSolverImex.cpp.\n");
#endif
  return 0;
}

PetscReal OdeSolverImex::computeStepSize(const PetscReal totErr)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolverImex::computeStepSize in OdeSolverImex.cpp.\n");
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
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolverImex::computeStepSize in OdeSolverImex.cpp.\n");
#endif
  return deltaT;
}

PetscReal OdeSolverImex::computeError()
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolverImex::computeError in OdeSolverImex.cpp.\n");
#endif
  PetscErrorCode ierr = 0;
  PetscReal      err,totErr=0.0;


  //~ for(std::vector<int>::size_type i = 0; i != _errInds.size(); i++) {
    //~ PetscInt ind = _errInds[i];

    //~ // error based on weighted 2 norm
    //~ Vec errVec;
    //~ PetscInt       size;
    //~ VecDuplicate(_var2nd[ind],&errVec);
    //~ ierr = VecWAXPY(errVec,-1.0,_var2nd[ind],_var3rd[ind]);CHKERRQ(ierr);
    //~ VecDot(errVec,errVec,&err);
    //~ VecGetSize(errVec,&size);
    //~ totErr += sqrt(err/size);
    //~ VecDestroy(&errVec);
  //~ }

  for(std::vector<int>::size_type i = 0; i != _errInds.size(); i++) {
    std::string key = _errInds[i];

    // error based on weighted 2 norm
    Vec errVec;
    PetscScalar    size;
    VecDuplicate(_var2nd[key],&errVec);
    ierr = VecWAXPY(errVec,-1.0,_var2nd[key],_var3rd[key]);CHKERRQ(ierr);
    VecNorm(errVec,NORM_2,&err);
    VecNorm(_var3rd[key],NORM_2,&size);
    totErr += err/(size+1.0);
    VecDestroy(&errVec);
  }


#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolverImex::computeError in OdeSolverImex.cpp.\n");
#endif
  return totErr;
}


PetscErrorCode OdeSolverImex::integrate(IntegratorContextImex *obj)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolverImex::integrate in OdeSolverImex.cpp.\n");
#endif
  double startTime = MPI_Wtime();

  PetscErrorCode ierr=0;
  PetscReal      totErr=0.0;
  PetscInt       attemptCount = 0;

  // build default errInds if it hasn't been defined already
  if (_errInds.size()==0) {
    for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
      _errInds.push_back(it->first);
    }
  }

  if (_finalT==_initT) { return ierr; }
  else if (_deltaT==0) { _deltaT = (_finalT-_initT)/_maxNumSteps; }

  if (_maxNumSteps == 0) { return ierr; }


  // set initial condition
  ierr = obj->d_dt(_currT,_var,_dvar);CHKERRQ(ierr);
  ierr = obj->timeMonitor(_currT,_stepCount,_var,_dvar,_varIm);CHKERRQ(ierr); // write initial conditions
  while (_stepCount<_maxNumSteps && _currT<_finalT) {

    _stepCount++;
    attemptCount = 0;
    while (attemptCount<100) { // repeat until time step is acceptable
      attemptCount++;
      if (attemptCount>=100) {PetscPrintf(PETSC_COMM_WORLD,"   WARNING: maximum number of attempts reached\n"); }
      //~ierr = PetscPrintf(PETSC_COMM_WORLD,"   attemptCount=%i\n",attemptCount);CHKERRQ(ierr);
      if (_currT+_deltaT>_finalT) { _deltaT=_finalT-_currT; }

      // set fields to 0
      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        VecSet(_varHalfdT[it->first],0.0); VecSet(_dvarHalfdT[it->first],0.0);
        VecSet(_vardT[it->first],0.0);     VecSet(_dvardT[it->first],0.0);
        VecSet(_var2nd[it->first],0.0);    VecSet(_dvar2nd[it->first],0.0);
        VecSet(_var3rd[it->first],0.0);
      }


      // stage 1: integrate fields to _currT + 0.5*deltaT
      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        ierr = VecWAXPY(_varHalfdT[it->first],0.5*_deltaT,_dvar[it->first],_var[it->first]);CHKERRQ(ierr);
      }
      ierr = obj->d_dt(_currT+0.5*_deltaT,_varHalfdT,_dvarHalfdT,
               _varHalfdTIm,_varIm,0.5*_deltaT);CHKERRQ(ierr);


      // stage 2: integrate fields to _currT + _deltaT
      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        ierr = VecWAXPY(_vardT[it->first],-_deltaT,_dvar[it->first],_var[it->first]);CHKERRQ(ierr);
        ierr = VecAXPY(_vardT[it->first],2*_deltaT,_dvarHalfdT[it->first]);CHKERRQ(ierr);
      }
      ierr = obj->d_dt(_currT+_deltaT,_vardT,_dvardT,_vardTIm,_varHalfdTIm,0.5*_deltaT);CHKERRQ(ierr);

      // 2nd and 3rd order update
      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        ierr = VecWAXPY(_var2nd[it->first],0.5*_deltaT,_dvar[it->first],_var[it->first]);CHKERRQ(ierr);
        ierr = VecAXPY(_var2nd[it->first],0.5*_deltaT,_dvardT[it->first]);CHKERRQ(ierr);

        ierr = VecWAXPY(_var3rd[it->first],_deltaT/6.0,_dvar[it->first],_var[it->first]);CHKERRQ(ierr);
        ierr = VecAXPY(_var3rd[it->first],2*_deltaT/3.0,_dvarHalfdT[it->first]);CHKERRQ(ierr);
        ierr = VecAXPY(_var3rd[it->first],_deltaT/6.0,_dvardT[it->first]);CHKERRQ(ierr);
      }

      // calculate error
      totErr = computeError();
      #if ODEPRINT > 0
        ierr = PetscPrintf(PETSC_COMM_WORLD,"    totErr = %.15e\n",totErr);
      #endif
      if (totErr<_atol) { break; } // !!!orig
      _deltaT = computeStepSize(totErr);
      if (_minDeltaT == _deltaT) { break; }

      _numRejectedSteps++;
    }
    _currT = _currT+_deltaT;

    // accept 3rd order solution as update
    for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
      VecSet(_var[it->first],0.0);
      ierr = VecCopy(_var3rd[it->first],_var[it->first]);CHKERRQ(ierr);
      VecSet(_dvar[it->first],0.0);
    }
    for (map<string,Vec>::iterator it = _vardTIm.begin(); it!=_vardTIm.end(); it++ ) {
      VecCopy(_vardTIm[it->first],_varIm[it->first]);
      VecCopy(_varHalfdTIm[it->first],_varIm_half[it->first]);
    }
    ierr = obj->d_dt(_currT,_var,_dvar);CHKERRQ(ierr);

    if (totErr!=0.0) {
      _deltaT = computeStepSize(totErr);
    }

    ierr = obj->timeMonitor(_currT,_stepCount,_var,_dvar,_varIm);CHKERRQ(ierr);
  }

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolverImex::integrate in OdeSolverImex.cpp.\n");
#endif
  return ierr;
}

