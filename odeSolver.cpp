#include "odeSolver.hpp"

using namespace std;

OdeSolver::OdeSolver(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType)
: _initT(0),_finalT(finalT),_currT(0),_deltaT(deltaT),
  _maxNumSteps(maxNumSteps),_stepCount(0),
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
  _lenVar = var.size();

  //~ _dvar.reserve(_lenVar);
  //~ for (int ind=0;ind<_lenVar;ind++) {
    //~ ierr = VecDuplicate(_var[ind],&_dvar[ind]);CHKERRQ(ierr);
    //~ ierr = VecSet(_dvar[ind],0.0);CHKERRQ(ierr);
  //~ }

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

  if (_finalT==_initT) { return ierr; }
  else if (_deltaT==0) { _deltaT = (_finalT-_initT)/_maxNumSteps; }

  // set initial condition
  ierr = obj->d_dt(_currT,_var,_dvar);CHKERRQ(ierr);
  ierr = obj->debug(_currT,_stepCount,_var,_dvar,"FE");CHKERRQ(ierr);
  ierr = obj->timeMonitor(_currT,_stepCount,_var,_dvar);CHKERRQ(ierr); // write first step

  while (_stepCount<_maxNumSteps && _currT<_finalT) {

    //~ ierr = obj->d_dt(_currT,_var.begin(),_dvar.begin());CHKERRQ(ierr);
    //~ ierr = obj->debug(_currT,_stepCount,_var.begin(),_dvar.begin(),"FE");CHKERRQ(ierr);
    ierr = obj->d_dt(_currT,_var,_dvar);CHKERRQ(ierr);
    ierr = obj->debug(_currT,_stepCount,_var,_dvar,"FE");CHKERRQ(ierr);
    //~ for (int varInd=0;varInd<_lenVar;varInd++) {
      //~ ierr = VecAXPY(_var[varInd],_deltaT,_dvar[varInd]);CHKERRQ(ierr); // var = var + deltaT*dvar
    //~ }
    for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
      ierr = VecAXPY(_var[it->first],_deltaT,_dvar[it->first]);CHKERRQ(ierr); // var = var + deltaT*dvar
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
  destroyVector(_dvar);
  destroyVector(_varHalfdT);
  destroyVector(_dvarHalfdT);
  destroyVector(_vardT);
  destroyVector(_dvardT);
  destroyVector(_var2nd);
  destroyVector(_dvar2nd);
  destroyVector(_var3rd);


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

PetscErrorCode RK32::setInitialConds(std::map<string,Vec>& var)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32::setInitialConds in odeSolver.cpp.\n");
#endif
  double startTime = MPI_Wtime();
  PetscErrorCode ierr = 0;

  //~ _var = var;
  //~ _lenVar = var.size();

  /*
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
  }
  */

  _var = var;
  for (map<string,Vec>::iterator it=var.begin(); it!=var.end(); it++ ) {
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

  _runTime += MPI_Wtime() - startTime;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32::setInitialConds in odeSolver.cpp.\n");
#endif
  return ierr;
}

PetscErrorCode RK32::setErrInds(std::vector<string>& errInds)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32::setErrInds in odeSolver.cpp.\n");
#endif
  _errInds = errInds;
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32::setTimeStepBounds in odeSolver.cpp.\n");
#endif
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
  PetscReal      err,totErr=0.0;


  //~ // relative error
  //~ for(std::vector<int>::size_type i = 0; i != _errInds.size(); i++) {
    //~ PetscInt ind = _errInds[i];

    //~ // error based on weighted 2 norm
    //~ Vec errVec;
    //~ PetscScalar    size;
    //~ VecDuplicate(_var2nd[ind],&errVec);
    //~ ierr = VecWAXPY(errVec,-1.0,_var2nd[ind],_var3rd[ind]);CHKERRQ(ierr);
    //~ VecNorm(errVec,NORM_2,&err);
    //~ VecNorm(_var3rd[ind],NORM_2,&size);
    //~ totErr += err/(size+1.0);
    //~ VecDestroy(&errVec);
  //~ }

  // relative error

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

  //~ // use quasi-static stress to determine rate
  //~ Vec errVec;
  //~ PetscScalar    size;
  //~ VecDuplicate(_var2nd["tau"],&errVec);
  //~ ierr = VecWAXPY(errVec,-1.0,_vardT["tau"],_varHalfdT["tau"]);CHKERRQ(ierr);
  //~ VecNorm(errVec,NORM_2,&err);
  //~ VecNorm(_var3rd["tau"],NORM_2,&size);
  //~ totErr += err/(size+1.0);
  //~ VecDestroy(&errVec);


#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Ending RK32::computeError in odeSolver.cpp.\n");
#endif
  return totErr;
}


PetscErrorCode RK32::integrate(IntegratorContextEx *obj)
{
#if VERBOSE > 1
  PetscPrintf(PETSC_COMM_WORLD,"Starting RK32::integrate in odeSolver.cpp.\n");
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

  // check that errInds is valid
  for(std::vector<int>::size_type i = 0; i != _errInds.size(); i++) {
    std::string key = _errInds[i];
    if (_var.find(key) == _var.end()) {
      PetscPrintf(PETSC_COMM_WORLD,"ERROR: %s is not an element of explicitly integrated variable!\n",key.c_str());
    }
    assert(_var.find(key) != _var.end());
  }


  // set initial condition
  ierr = obj->d_dt(_currT,_var,_dvar);CHKERRQ(ierr);
  ierr = obj->debug(_currT,_stepCount,_var,_dvar,"IC");CHKERRQ(ierr);
  ierr = obj->timeMonitor(_currT,_stepCount,_var,_dvar);CHKERRQ(ierr); // write first step

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
      ierr = obj->d_dt(_currT+0.5*_deltaT,_varHalfdT,_dvarHalfdT);CHKERRQ(ierr);
      ierr = obj->debug(_currT+0.5*_deltaT,_stepCount,_varHalfdT,_dvarHalfdT,"t+dt/2");CHKERRQ(ierr);

      // stage 2: integrate fields to _currT + _deltaT
      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        ierr = VecWAXPY(_vardT[it->first],-_deltaT,_dvar[it->first],_var[it->first]);CHKERRQ(ierr);
        ierr = VecAXPY(_vardT[it->first],2*_deltaT,_dvarHalfdT[it->first]);CHKERRQ(ierr);
      }
      ierr = obj->d_dt(_currT+_deltaT,_vardT,_dvardT);CHKERRQ(ierr);
      ierr = obj->debug(_currT+_deltaT,_stepCount,_vardT,_dvardT,"t+dt");CHKERRQ(ierr);

      // 2nd and 3rd order update
      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        ierr = VecWAXPY(_var2nd[it->first],0.5*_deltaT,_dvar[it->first],_var[it->first]);CHKERRQ(ierr);
        ierr = VecAXPY(_var2nd[it->first],0.5*_deltaT,_dvardT[it->first]);CHKERRQ(ierr);

        ierr = VecWAXPY(_var3rd[it->first],_deltaT/6.0,_dvar[it->first],_var[it->first]);CHKERRQ(ierr);
        ierr = VecAXPY(_var3rd[it->first],2*_deltaT/3.0,_dvarHalfdT[it->first]);CHKERRQ(ierr);
        ierr = VecAXPY(_var3rd[it->first],_deltaT/6.0,_dvardT[it->first]);CHKERRQ(ierr);
      }
      ierr = obj->debug(_currT+_deltaT,_stepCount,_var2nd,_dvardT,"2nd");CHKERRQ(ierr);
      ierr = obj->debug(_currT+_deltaT,_stepCount,_var3rd,_dvardT,"3rd");CHKERRQ(ierr);


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
    ierr = obj->d_dt(_currT,_var,_dvar);CHKERRQ(ierr);
    ierr = obj->debug(_currT,_stepCount,_var,_dvar,"F");CHKERRQ(ierr);

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


