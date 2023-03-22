#include "odeSolver.hpp"

using namespace std;

// constructor for OdeSolver
OdeSolver::OdeSolver(PetscInt maxNumSteps, PetscReal finalT,PetscReal deltaT,string controlType)
: _initT(0),_finalT(finalT),_currT(0),_deltaT(deltaT),_newDeltaT(deltaT),
  _maxNumSteps(maxNumSteps),_stepCount(0),_runTime(0),
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
  return 0;
}


// set initial, current, final time and calculates runtime since a point in time
PetscErrorCode OdeSolver::setTimeRange(const PetscReal initT, const PetscReal finalT)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolver::setTimeRange in odeSolver.cpp.\n");
  #endif

  // return an elapsed time on the calling processor
  double startTime = MPI_Wtime();
  _initT = initT;
  _currT = initT;
  _finalT = finalT;
  _runTime += MPI_Wtime() - startTime;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolver::setTimeRange in odeSolver.cpp.\n");
  #endif

  return 0;
}


// set step size deltaT, computes runtime since startTime
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


// set the type of norm to be calculated, calculates runtime since startTime
PetscErrorCode OdeSolver::setToleranceType(const string normType)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting OdeSolver::setToleranceType in odeSolver.cpp.\n");
  #endif

  double startTime = MPI_Wtime();
  _normType = normType;
  assert(_normType.compare("L2_relative")==0 ||
    _normType.compare("L2_absolute")==0 ||
    _normType.compare("max_relative")==0 );

  _runTime += MPI_Wtime() - startTime;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolver::setToleranceType in odeSolver.cpp.\n");
  #endif

  return 0;
}



//================= FEuler child class functions =======================

// constructor, initializes same object as OdeSolver
FEuler::FEuler(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType)
  : OdeSolver(maxNumSteps,finalT,deltaT,controlType)
{}


//destructor, free memory
FEuler::~FEuler() {
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting FEuler::destructor in odeSolver.cpp.\n");
  #endif

  // destruct temporary container
  destroyVector(_dvar);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending FEuler::destructor in odeSolver.cpp.\n");
  #endif
}


// print out information about method
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

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending FEuler::view in odeSolver.cpp.\n");
  #endif

  return 0;
}


// set initial condition on _var and _dvar, computes runtime for this step
PetscErrorCode FEuler::setInitialConds(map<string,Vec>& var)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting FEuler::setInitialConds in odeSolver.cpp.\n");
  #endif

  double startTime = MPI_Wtime();
  PetscErrorCode ierr = 0;
  _var = var; // shallow copy

  for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++) {
    Vec dvar;
    ierr = VecDuplicate(_var[it->first],&dvar); CHKERRQ(ierr);
    ierr = VecSet(dvar,0.0); CHKERRQ(ierr);
    _dvar[it->first] = dvar;
  }

  _runTime += MPI_Wtime() - startTime;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending FEuler::setInitialConds in odeSolver.cpp.\n");
  #endif

  return ierr;
}


// perform explicit time stepping, calling d_dt method defined in strikeSlip_linearElastic_qd, which is a derived class from IntegratorContextEx
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
  //ierr = obj->timeMonitor(_currT,_deltaT,_stepCount,stopIntegration); CHKERRQ(ierr); // write first step

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

PetscErrorCode FEuler::writeCheckpoint(PetscViewer &viewer)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting FEuler::writeCheckpoint in odeSolver.cpp.\n");
  #endif

  // not needed, but just in case: _deltaT, _totErr

  PetscErrorCode ierr = 0;

  // initiate Vec to serve as underlying data set for step count and deltaT to be written out as attributes
  Vec temp;
  VecCreateMPI(PETSC_COMM_WORLD, 1, 1, &temp);
  VecSetBlockSize(temp, 1);
  PetscObjectSetName((PetscObject) temp, "odeSolver_chkpt_data");
  VecSet(temp,0.);

  ierr = PetscViewerHDF5PushGroup(viewer, "/odeSolver");        CHKERRQ(ierr);
  ierr = VecView(temp, viewer);                                         CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer, "odeSolver_chkpt_data", "stepCount", PETSC_INT, &_stepCount); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer, "odeSolver_chkpt_data", "deltaT", PETSC_SCALAR, &_deltaT); CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer); CHKERRQ(ierr);
  VecDestroy(&temp);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending FEuler::writeCheckpoint in odeSolver.cpp.\n");
  #endif

  return ierr;
}

PetscErrorCode FEuler::loadCheckpoint(const std::string inputDir)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting FEuler::loadCheckpoint in odeSolver.cpp.\n");
  #endif
  PetscErrorCode ierr;

  string fileName = inputDir + "checkpoint.h5";

  // load saved checkpoint data
  PetscViewer viewer;

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/odeSolver");                                                  CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "odeSolver_chkpt_data", "stepCount", PETSC_INT, NULL, &_stepCount); CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "odeSolver_chkpt_data", "deltaT", PETSC_SCALAR, NULL, &_deltaT); CHKERRQ(ierr);

  ierr = PetscViewerHDF5PopGroup(viewer);                                                              CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending FEuler::loadCheckpoint in odeSolver.cpp.\n");
  #endif

  return ierr;
}


//================= RK32 child class functions =========================

// constructor, initializes OdeSolver object and more parameters
RK32::RK32(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType)
  : OdeSolver(maxNumSteps,finalT,deltaT,controlType),
    _minDeltaT(0),_maxDeltaT(finalT),
    _totTol(1e-9),_kappa(0.9),_ord(3.0),
    _numRejectedSteps(0),_numMinSteps(0),_numMaxSteps(0)
{

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting RK32::constructor in odeSolver.cpp.\n");
  #endif

  double startTime = MPI_Wtime();

  // initialize place-holder values for errA, which holds errors from past 2 time steps
  _errA[0] = 0.;
  _errA[1] = 0.;

  _runTime += MPI_Wtime() - startTime;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolver::constructor in odeSolver.cpp.\n");
  #endif
}


// destructor, frees memory
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


// print out various set up and run time information
PetscErrorCode RK32::view()
{
  PetscErrorCode ierr = 0;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTime Integration summary:\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   integration algorithm: runge-kutta (3,2)\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   control scheme: %s\n",_controlType.c_str());CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   norm type used to measure error: %s\n",_normType.c_str());CHKERRQ(ierr);
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


// sets tolerance levels
PetscErrorCode RK32::setTolerance(const PetscReal tol)
{
  _totTol = tol;
  return 0;
}


// set initial condition on _var and _dvar
PetscErrorCode RK32::setInitialConds(map<string,Vec>& var)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting RK32::setInitialConds in odeSolver.cpp.\n");
  #endif

  double startTime = MPI_Wtime();
  PetscErrorCode ierr = 0;
  _var = var; // shallow copy

  // initialize RK vectors to zero
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

    Vec y2;
    ierr = VecDuplicate(_var[it->first],&y2); CHKERRQ(ierr);
    ierr = VecSet(y2,0.0); CHKERRQ(ierr);
    _y2[it->first] = y2;

    Vec y3;
    ierr = VecDuplicate(_var[it->first],&y3); CHKERRQ(ierr);
    ierr = VecSet(y3,0.0); CHKERRQ(ierr);
    _y3[it->first] = y3;
  }

  _runTime += MPI_Wtime() - startTime;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending RK32::setInitialConds in odeSolver.cpp.\n");
  #endif

  return ierr;
}


// set value of _errInds
PetscErrorCode RK32::setErrInds(vector<string>& errInds) {
  _errInds = errInds;
  return 0;
}


// set value of _errInds and _scale
PetscErrorCode RK32::setErrInds(vector<string>& errInds, vector<double> scale)
{
  _errInds = errInds;
  _scale = scale;
  return 0;
}


// set _minDeltaT and _maxDeltaT
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


// compute time step size
PetscReal RK32::computeStepSize(const PetscReal totErr)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting RK32::computeStepSize in odeSolver.cpp.\n");
  #endif

  PetscReal stepRatio;

  // if using integral feedback controller (I)
  if (_controlType.compare("P") == 0) {
    PetscReal alpha = 1./(1.+_ord);
    stepRatio = _kappa*pow(_totTol/totErr,alpha);
  }

  // if using proportional-integral-derivative feedback (PID)
  else if (_controlType.compare("PID") == 0) {
    PetscReal alpha = 0.49/_ord;
    PetscReal beta  = 0.34/_ord;
    PetscReal gamma = 0.1/_ord;

    // only do this when we're at the first simulation with no history of _errA
    if (_stepCount < 4) {
      stepRatio = _kappa*pow(_totTol/totErr,1./(1.+_ord));
    }
    else {
      stepRatio = _kappa * pow(_totTol/totErr,alpha)
                         * pow(_errA[0]/_totTol,beta)
                         * pow(_totTol/_errA[1],gamma);
    }
  }
  else { // unknown control type
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: timeControlType not understood\n");
    assert(0); // automatically fail
  }

  PetscReal deltaT = stepRatio*_deltaT;

  // respect bounds on min and max possible step size
  deltaT = min(_deltaT*5.0,deltaT); // cap growth rate of step size
  deltaT = min(_maxDeltaT,deltaT); // absolute max
  deltaT = max(_minDeltaT,deltaT);

  if (_minDeltaT == deltaT) { _numMinSteps++; }
  else if (_maxDeltaT == deltaT) { _numMaxSteps++; }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending RK32::computeStepSize in odeSolver.cpp.\n");
  #endif

  return deltaT;
}


// compute the L2 error (absolute/relative)
PetscReal RK32::computeError()
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting RK32::computeError in odeSolver.cpp.\n");
  #endif

  PetscErrorCode ierr = 0;
  PetscScalar err = 0, _totErr = 0;

  // if using absolute error for control
  // error: the absolute L2 error, weighted by N and a user-inputted scale factor
  // tolerance: the absolute tolerance
  if (_normType.compare("L2_absolute")==0) {
    for(vector<int>::size_type i = 0; i != _errInds.size(); i++) {
      string key = _errInds[i];
      Vec errVec;
      VecDuplicate(_y3[key],&errVec);
      VecSet(errVec,0.0);
      ierr = VecWAXPY(errVec,-1.0,_y3[key],_y2[key]); CHKERRQ(ierr);
      VecNorm(errVec,NORM_2,&err);
      VecDestroy(&errVec);

      PetscInt N = 0;
      VecGetSize(_y3[key],&N);
      _totErr += err / (sqrt(N) * _scale[i]);
    }
  }

  // if using relative error for control
  // error: the absolute L2 error, scaled by the L2 norm of the solution and a user-inputted scale factor
  // tolerance: the relative tolerance
  if (_normType.compare("L2_relative")==0) {
    for(vector<int>::size_type i = 0; i != _errInds.size(); i++) {
      string key = _errInds[i];
      Vec errVec;
      VecDuplicate(_y3[key],&errVec);
      VecSet(errVec,0.0);
      ierr = VecWAXPY(errVec,-1.0,_y3[key],_y2[key]); CHKERRQ(ierr);
      VecNorm(errVec,NORM_2,&err);
      VecDestroy(&errVec);

      PetscReal s = 0;
      VecNorm(_y3[key],NORM_2,&s);
      _totErr += err / (s * _scale[i]);
    }
  }

  // if using the maximum relative error for control
  if (_normType.compare("max_relative")==0) {
    for(vector<int>::size_type i = 0; i != _errInds.size(); i++) {
      string key = _errInds[i];
      Vec errVec;
      VecDuplicate(_y3[key],&errVec);
      VecSet(errVec,0.0);
      ierr = VecWAXPY(errVec,-1.0,_y4[key],_y3[key]); CHKERRQ(ierr);
      VecAbs(errVec);
      VecPointwiseDivide(errVec,errVec,_y4[key]);
      VecMax(errVec,NULL,&err);
      VecDestroy(&errVec);
      assert(!std::isinf(err));
      _totErr += err / (_scale[i]);
    }
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending RK32::computeError in odeSolver.cpp.\n");
  #endif

  return _totErr;
}


// perform RK32 explicit time stepping
PetscErrorCode RK32::integrate(IntegratorContextEx *obj)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting RK32::integrate in odeSolver.cpp.\n");
  #endif

  double         startTime = MPI_Wtime();
  PetscErrorCode ierr = 0;
  PetscScalar    _totErr = 0;
  PetscInt       attemptCount = 0;
  int            stopIntegration = 0;

  // build default errInds if it hasn't been defined already
  if (_errInds.size()==0) {
    for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
      _errInds.push_back(it->first);
    }
  }

  // check that errInds is valid
  for(vector<int>::size_type i = 0; i != _errInds.size(); i++) {
    string key = _errInds[i];
    if (_var.find(key) == _var.end()) {
      PetscPrintf(PETSC_COMM_WORLD,"RK32 ERROR: %s is not an element of explicitly integrated variable!\n",key.c_str());
    }
    assert(_var.find(key) != _var.end());
  }

  // set up scaling for elements in errInds
  if (_scale.size() == 0) { // if 0 entries, set all to 1
    for(vector<int>::size_type i = 0; i != _errInds.size(); i++) {
      _scale.push_back(1.0);
    }
  }
  assert(_scale.size() == _errInds.size());

  if (_finalT==_initT) { return ierr; }
  else if (_deltaT==0) { _deltaT = (_finalT-_initT)/_maxNumSteps; }
  if (_maxNumSteps == 0) { return ierr; }

  // set initial condition
  ierr = obj->d_dt(_currT,_var,_dvar);CHKERRQ(ierr);
  //~ ierr = obj->timeMonitor(_currT,_deltaT,_stepCount,stopIntegration); CHKERRQ(ierr);

  // perform time stepping routine and calling d_dt
  while (_stepCount < _maxNumSteps && _currT < _finalT) {
    _stepCount++;
    attemptCount = 0;

    while (attemptCount<100) { // repeat until time step is acceptable
      attemptCount++;
      if (attemptCount>=100) {PetscPrintf(PETSC_COMM_WORLD,"   RK32 WARNING: maximum number of attempts reached\n"); }
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
      if (_totErr <= _totTol) { break; }
      _deltaT = computeStepSize(_totErr);
      if (_minDeltaT == _deltaT) { break; }

      _numRejectedSteps++;
    }

    // accept 3rd order solution as update
    _currT = _currT+_deltaT;
    for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
      ierr = VecCopy(_y3[it->first],_var[it->first]);CHKERRQ(ierr);
      VecSet(_dvar[it->first],0.0);
    }
    ierr = obj->d_dt(_currT,_var,_dvar);CHKERRQ(ierr);

    // compute new deltaT for next time step
    // but timeMonitor before updating to newDeltaT, to keep output consistent while allowing for checkpointing
    if (_totErr!=0.0) { _newDeltaT = computeStepSize(_totErr); }
    _errA[1] = _errA[0]; // record error for use when estimating time step
    _errA[0] = _totErr;


    ierr = obj->timeMonitor(_currT,_deltaT,_stepCount,stopIntegration); CHKERRQ(ierr);
    if (stopIntegration > 0) { PetscPrintf(PETSC_COMM_WORLD,"RK32: Detected stop time integration request.\n"); break; }

    // now update deltaT
    _deltaT = _newDeltaT;
  }

  _runTime += MPI_Wtime() - startTime;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending RK32::integrate in odeSolver.cpp.\n");
  #endif

  return ierr;
}

PetscErrorCode RK32::loadCheckpoint(const std::string inputDir)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting RK43::loadCheckpoint in odeSolver.cpp.\n");
  #endif
  PetscErrorCode ierr;

  // needed errA[0], errA[1], _stepCount
  // not needed, but just in case: _deltaT, _totErr

  string fileName = inputDir + "checkpoint.h5";

  // load saved checkpoint data
  PetscViewer viewer;

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/time1D");                                                  CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "odeSolver_chkpt_data", "stepCount", PETSC_INT, NULL, &_stepCount); CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "odeSolver_chkpt_data", "errA0", PETSC_SCALAR, NULL, &_errA[0]); CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "odeSolver_chkpt_data", "errA1", PETSC_SCALAR, NULL, &_errA[1]); CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "odeSolver_chkpt_data", "totErr", PETSC_SCALAR, NULL, &_totErr); CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "odeSolver_chkpt_data", "deltaT", PETSC_SCALAR, NULL, &_deltaT); CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "odeSolver_chkpt_data", "deltaT", PETSC_SCALAR, NULL, &_newDeltaT); CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                                                              CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending RK32::loadCheckpoint in odeSolver.cpp.\n");
  #endif

  return ierr;
}

PetscErrorCode RK32::writeCheckpoint(PetscViewer &viewer)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting RK32::writeCheckpoint in odeSolver.cpp.\n");
  #endif
  PetscErrorCode ierr;

  // needed errA[0], errA[1], _stepCount
  // not needed, but just in case: _deltaT, _totErr

  // initiate Vec to serve as underlying data set for step count and deltaT to be written out as attributes
  Vec temp;
  VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 1, &temp);
  VecSetBlockSize(temp, 1);
  PetscObjectSetName((PetscObject) temp, "odeSolver_chkpt_data");
  VecSet(temp,0.);

  ierr = PetscViewerHDF5PushGroup(viewer, "/odeSolver");        CHKERRQ(ierr);
  ierr = VecView(temp, viewer);                                         CHKERRQ(ierr);
    ierr = PetscViewerHDF5WriteAttribute(viewer, "odeSolver_chkpt_data", "stepCount", PETSC_INT, &_stepCount); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer, "odeSolver_chkpt_data", "errA0", PETSC_SCALAR, &_errA[0]);    CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer, "odeSolver_chkpt_data", "errA1", PETSC_SCALAR, &_errA[1]);    CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer, "odeSolver_chkpt_data", "deltaT", PETSC_SCALAR, &_newDeltaT); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer, "odeSolver_chkpt_data", "totErr", PETSC_SCALAR, &_totErr);    CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer); CHKERRQ(ierr);
  VecDestroy(&temp);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending RK32::writeCheckpoint in odeSolver.cpp.\n");
  #endif

  return ierr;
}

//======================================================================
//                  RK43 child class
//======================================================================

// constructor, initializes same object as RK32
RK43::RK43(PetscInt maxNumSteps,PetscReal finalT,PetscReal deltaT,string controlType)
  : OdeSolver(maxNumSteps,finalT,deltaT,controlType),
  _minDeltaT(0),_maxDeltaT(finalT),
  _totTol(1e-9),_kappa(0.9),_ord(4.0),
  _numRejectedSteps(0),_numMinSteps(0),_numMaxSteps(0)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting RK43::constructor in odeSolver.cpp.\n");
  #endif

  double startTime = MPI_Wtime();

  // initialize place-holder values for errA, which holds errors from past 2 time steps
  _errA[0] = 0.;
  _errA[1] = 0.;

  _runTime += MPI_Wtime() - startTime;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending OdeSolver::constructor in odeSolver.cpp.\n");
  #endif
}


// destructor, frees intermediate vectors used in RK43 method
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


// print out various information about the method
PetscErrorCode RK43::view()
{
  PetscErrorCode ierr = 0;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTime Integration summary:\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   integration algorithm: runge-kutta (4,3)\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   control scheme: %s\n",_controlType.c_str());CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   norm type used to measure error: %s\n",_normType.c_str());CHKERRQ(ierr);
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


// set tolerance levels
PetscErrorCode RK43::setTolerance(const PetscReal tol)
{
  _totTol = tol;
  return 0;
}


// set initial conditions on _var, _dvar, and intermediate vectors used in RK43
PetscErrorCode RK43::setInitialConds(map<string,Vec>& var)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting RK43::setInitialConds in odeSolver.cpp.\n");
  #endif

  double startTime = MPI_Wtime();
  PetscErrorCode ierr = 0;
  _var = var;

  // initialize _dvar and various RK43 intermediate vectors to zero
  for (map<string,Vec>::iterator it=var.begin(); it!=var.end(); it++ ) {
    Vec dvar;
    ierr = VecDuplicate(_var[it->first],&dvar); CHKERRQ(ierr);
    ierr = VecSet(dvar,0.0); CHKERRQ(ierr);
    _dvar[it->first] = dvar;

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


// set error indices
PetscErrorCode RK43::setErrInds(vector<string>& errInds) {
  _errInds = errInds;
  return 0;
}


// set error indices and scale
PetscErrorCode RK43::setErrInds(vector<string>& errInds, vector<double> scale)
{
  _errInds = errInds;
  _scale = scale;
  return 0;
}


// set _minDeltaT and _maxDeltaT
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


// compute the time stepping size, depending on control type
PetscReal RK43::computeStepSize(const PetscReal totErr)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting RK43::computeStepSize in odeSolver.cpp.\n");
  #endif

  PetscReal stepRatio;

  // if using integral feedback controller (I)
  if (_controlType == "P") {
    PetscReal alpha = 1./(1.+_ord);
    stepRatio = _kappa*pow(_totTol/totErr,alpha);
  }

  //if using proportional-integral-derivative feedback (PID)
  else if (_controlType == "PID") {
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
    assert(0 > 1);
  }

  PetscReal deltaT = stepRatio*_deltaT;

  // respect bounds on min and max possible step size
  deltaT = min(_deltaT*5.0,deltaT); // cap growth rate of step size
  deltaT= min(_maxDeltaT,deltaT); // absolute max
  deltaT = max(_minDeltaT,deltaT);

  if (_minDeltaT == deltaT) {
    _numMinSteps++;
  }
  else if (_maxDeltaT == deltaT) {
    _numMaxSteps++;
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending RK43::computeStepSize in odeSolver.cpp.\n");
  #endif

  return deltaT;
}


// compute the L2 error (absolute/relative), returns _totErr
PetscReal RK43::computeError()
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting RK43::computeError in odeSolver.cpp.\n");
  #endif

  PetscErrorCode ierr = 0;
  PetscScalar err = 0, _totErr = 0;

  // if using absolute error for control
  // error: the absolute L2 error, weighted by N and a user-inputted scale factor
  // tolerance: the absolute tolerance
  if (_normType.compare("L2_absolute")==0) {
    for(vector<int>::size_type i = 0; i != _errInds.size(); i++) {
      string key = _errInds[i];
      Vec errVec;
      VecDuplicate(_y3[key],&errVec);
      VecSet(errVec,0.0);
      ierr = VecWAXPY(errVec,-1.0,_y4[key],_y3[key]); CHKERRQ(ierr);
      VecNorm(errVec,NORM_2,&err);
      VecDestroy(&errVec);

      PetscInt N = 0;
      VecGetSize(_y4[key],&N);
      _totErr += err / (sqrt(N) * _scale[i]);
    }
  }

  // if using relative error for control
  // error: the absolute L2 error, scaled by the L2 norm of the solution and a user-inputted scale factor
  // tolerance: the relative tolerance
  if (_normType.compare("L2_relative")==0) {
    for(vector<int>::size_type i = 0; i != _errInds.size(); i++) {
      string key = _errInds[i];
      Vec errVec;
      VecDuplicate(_y3[key],&errVec);
      VecSet(errVec,0.0);
      ierr = VecWAXPY(errVec,-1.0,_y4[key],_y3[key]); CHKERRQ(ierr);
      VecNorm(errVec,NORM_2,&err);
      VecDestroy(&errVec);
      PetscReal s = 0;
      VecNorm(_y4[key],NORM_2,&s);
      _totErr += err / (s * _scale[i]);
    }
  }

  // if using the maximum relative error for control
  if (_normType.compare("max_relative")==0) {
    for(vector<int>::size_type i = 0; i != _errInds.size(); i++) {
      string key = _errInds[i];
      Vec errVec;
      VecDuplicate(_y3[key],&errVec);
      VecSet(errVec,0.0);
      ierr = VecWAXPY(errVec,-1.0,_y4[key],_y3[key]); CHKERRQ(ierr);
      VecAbs(errVec);
      VecPointwiseDivide(errVec,errVec,_y4[key]);
      VecMax(errVec,NULL,&err);
      VecDestroy(&errVec);
      assert(!std::isinf(err));
      _totErr += err / (_scale[i]);
    }
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending RK43::computeError in odeSolver.cpp.\n");
  #endif

  return _totErr;
}


// perform explicit RK4 time stepping, calling d_dt method
PetscErrorCode RK43::integrate(IntegratorContextEx *obj)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting RK43::integrate in odeSolver.cpp.\n");
  #endif

  double startTime = MPI_Wtime();
  PetscErrorCode  ierr = 0;
  PetscScalar    _totErr = 0;
  PetscInt       attemptCount = 0;
  int            stopIntegration = 0;

  // coefficients
  PetscScalar c2 = 1./2.;
  PetscScalar c3 = 83./250.;
  PetscScalar c4 = 31./50.;
  PetscScalar c5 = 17./20.;
  PetscScalar c6 = 1.;

  PetscScalar b1 = 82889./524892.;
  PetscScalar b3 = 15625./83664.;
  PetscScalar b4 = 69875./102672.;
  PetscScalar b5 = -2260./8211.;
  PetscScalar b6 = 1./4.;


  PetscScalar hb1 = 4586570599./29645900160.;
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
  for(vector<int>::size_type i = 0; i != _errInds.size(); i++) {
    string key = _errInds[i];
    if (_var.find(key) == _var.end()) {
      PetscPrintf(PETSC_COMM_WORLD,"RK43 ERROR: %s is not an element of explicitly integrated variable!\n",key.c_str());
    }
    assert(_var.find(key) != _var.end());
  }

  // set up scaling for elements in errInds
  if (_scale.size() == 0) { // if 0 entries, set all to 1
    for(vector<int>::size_type i = 0; i != _errInds.size(); i++) {
      _scale.push_back(1.0);
    }
  }
  assert(_scale.size() == _errInds.size());

  if (_finalT == _initT) { return ierr; }
  if (_deltaT == 0) { _deltaT = (_finalT - _initT) / _maxNumSteps; }
  if (_maxNumSteps == 0) { return ierr; }

  // set initial condition
  ierr = obj->d_dt(_currT,_var,_dvar);CHKERRQ(ierr);
  //~ ierr = obj->timeMonitor(_currT,_deltaT,_stepCount,stopIntegration); CHKERRQ(ierr);

  // perform time stepping
  while (_stepCount < _maxNumSteps && _currT < _finalT) {
    _stepCount++;
    attemptCount = 0;
    while (attemptCount < 100) {
      attemptCount++;
      if (attemptCount >= 100) { PetscPrintf(PETSC_COMM_WORLD,"   RK43 WARNING: maximum number of attempts reached\n"); }

      if (_currT + _deltaT > _finalT) { _deltaT = _finalT - _currT; }

      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        VecSet(_k2[it->first],0.0);
        VecSet(_k3[it->first],0.0); VecSet(_k4[it->first],0.0);
        VecSet(_k5[it->first],0.0); VecSet(_k6[it->first],0.0);
        VecSet(_f2[it->first],0.0);
        VecSet(_f3[it->first],0.0); VecSet(_f4[it->first],0.0);
        VecSet(_f5[it->first],0.0); VecSet(_f6[it->first],0.0);
      }

      // stage 1: k1 = var, compute f1 = f(k1) = dvar
      _f1 = _dvar;

      // stage 2: compute k2
      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        ierr = VecWAXPY(_k2[it->first],a21*_deltaT,_f1[it->first],_var[it->first]); CHKERRQ(ierr);
      }
      //~ PetscPrintf(PETSC_COMM_WORLD,"\n\nstage 2\n");
      ierr = obj->d_dt(_currT+c2*_deltaT,_k2,_f2);CHKERRQ(ierr);

      // stage 3: compute k3
      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        ierr = VecWAXPY(_k3[it->first],a31*_deltaT,_f1[it->first],_var[it->first]); CHKERRQ(ierr);
        ierr = VecAXPY(_k3[it->first],a32*_deltaT,_f2[it->first]); CHKERRQ(ierr);
      }
      //~ PetscPrintf(PETSC_COMM_WORLD,"\n\nstage 3\n");
      ierr = obj->d_dt(_currT+c3*_deltaT,_k3,_f3);CHKERRQ(ierr);

      // stage 4
      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        ierr = VecWAXPY(_k4[it->first],a41*_deltaT,_f1[it->first],_var[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k4[it->first],a42*_deltaT,_f2[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k4[it->first],a43*_deltaT,_f3[it->first]); CHKERRQ(ierr);
      }
      //~ PetscPrintf(PETSC_COMM_WORLD,"\n\nstage 4\n");
      ierr = obj->d_dt(_currT+c4*_deltaT,_k4,_f4);CHKERRQ(ierr);

      // stage 5
      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        ierr = VecWAXPY(_k5[it->first],a51*_deltaT,_f1[it->first],_var[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k5[it->first],a52*_deltaT,_f2[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k5[it->first],a53*_deltaT,_f3[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k5[it->first],a54*_deltaT,_f4[it->first]); CHKERRQ(ierr);
      }
      //~ PetscPrintf(PETSC_COMM_WORLD,"\n\nstage 5\n");
      ierr = obj->d_dt(_currT+c5*_deltaT,_k5,_f5);CHKERRQ(ierr);

      // stage 6
      for (map<string,Vec>::iterator it = _var.begin(); it!=_var.end(); it++ ) {
        ierr = VecWAXPY(_k6[it->first],a61*_deltaT,_f1[it->first],_var[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k6[it->first],a62*_deltaT,_f2[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k6[it->first],a63*_deltaT,_f3[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k6[it->first],a64*_deltaT,_f4[it->first]); CHKERRQ(ierr);
        ierr =  VecAXPY(_k6[it->first],a65*_deltaT,_f5[it->first]); CHKERRQ(ierr);
      }
      //~ PetscPrintf(PETSC_COMM_WORLD,"\n\nstage 6\n");
      ierr = obj->d_dt(_currT+c6*_deltaT,_k6,_f6);CHKERRQ(ierr);

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
      if (_totErr<_totTol) { break; } // accept step
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
    //~ PetscPrintf(PETSC_COMM_WORLD,"\n\nFinal stage\n");
    ierr = obj->d_dt(_currT,_var,_dvar);CHKERRQ(ierr);

    // compute new deltaT for next time step
    // but call timeMonitor before updating to newDeltaT, to keep output
    // consistent while allowing for checkpointing
    if (_totErr!=0.0) { _newDeltaT = computeStepSize(_totErr); }
    _errA[1] = _errA[0]; // record error for use when estimating time step
    _errA[0] = _totErr;

    ierr = obj->timeMonitor(_currT,_deltaT,_stepCount,stopIntegration); CHKERRQ(ierr);
    if (stopIntegration > 0) { PetscPrintf(PETSC_COMM_WORLD,"RK43: Detected stop time integration request.\n"); break; }

    // now update deltaT
    _deltaT = _newDeltaT;

  }

  _runTime += MPI_Wtime() - startTime;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending RK43::integrate in odeSolver.cpp.\n");
  #endif

  return ierr;
}

PetscErrorCode RK43::loadCheckpoint(const std::string inputDir)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting RK43::loadCheckpoint in odeSolver.cpp.\n");
  #endif
  PetscErrorCode ierr;

  // needed errA[0], errA[1], _stepCount
  // not needed, but just in case: _deltaT, _totErr

  string fileName = inputDir + "checkpoint.h5";

  // load saved checkpoint data
  PetscViewer viewer;

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/odeSolver");                                                  CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "odeSolver_chkpt_data", "stepCount", PETSC_INT, NULL, &_stepCount); CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "odeSolver_chkpt_data", "errA0", PETSC_SCALAR, NULL, &_errA[0]); CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "odeSolver_chkpt_data", "errA1", PETSC_SCALAR, NULL, &_errA[1]); CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "odeSolver_chkpt_data", "totErr", PETSC_SCALAR, NULL, &_totErr); CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "odeSolver_chkpt_data", "deltaT", PETSC_SCALAR, NULL, &_deltaT); CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "odeSolver_chkpt_data", "deltaT", PETSC_SCALAR, NULL, &_newDeltaT); CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                                                              CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending RK43::loadCheckpoint in odeSolver.cpp.\n");
  #endif

  return ierr;
}

PetscErrorCode RK43::writeCheckpoint(PetscViewer &viewer)
{
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting RK43::writeCheckpoint in odeSolver.cpp.\n");
  #endif
  PetscErrorCode ierr;

  // needed errA[0], errA[1], _stepCount
  // not needed, but just in case: _deltaT, _totErr

  // initiate Vec to serve as underlying data set for step count and deltaT to be written out as attributes
  Vec temp;
  VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 1, &temp);
  VecSetBlockSize(temp, 1);
  PetscObjectSetName((PetscObject) temp, "odeSolver_chkpt_data");
  VecSet(temp,_stepCount);

  ierr = PetscViewerHDF5PushGroup(viewer, "/odeSolver");                CHKERRQ(ierr);
  ierr = VecView(temp, viewer);                                         CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer, "odeSolver_chkpt_data", "stepCount", PETSC_INT, &_stepCount); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer, "odeSolver_chkpt_data", "errA0", PETSC_SCALAR, &_errA[0]);    CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer, "odeSolver_chkpt_data", "errA1", PETSC_SCALAR, &_errA[1]);    CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer, "odeSolver_chkpt_data", "deltaT", PETSC_SCALAR, &_newDeltaT); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer, "odeSolver_chkpt_data", "totErr", PETSC_SCALAR, &_totErr);    CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer); CHKERRQ(ierr);
  VecDestroy(&temp);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending RK43::writeCheckpoint in odeSolver.cpp.\n");
  #endif

  return ierr;
}
