#include "mediator.hpp"

#define FILENAME "mediator.cpp"

using namespace std;


Mediator::Mediator(Domain&D)
: _delim(D._delim),_isMMS(D._isMMS),
  _bcLTauQS(0),
  _outputDir(D._outputDir),
  _vL(D._vL),
  _thermalCoupling("no"),_heatEquationType("transient"),
  _hydraulicCoupling("no"),_hydraulicTimeIntType("explicit"),
  _timeIntegrator(D._timeIntegrator),
  _stride1D(D._stride1D),_stride2D(D._stride2D),_maxStepCount(D._maxStepCount),
  _initTime(D._initTime),_currTime(_initTime),_maxTime(D._maxTime),
  _minDeltaT(D._minDeltaT),_maxDeltaT(D._maxDeltaT),
  _stepCount(0),_atol(D._atol),_initDeltaT(D._initDeltaT),_timeIntInds(D._timeIntInds),
  _integrateTime(0),_writeTime(0),_linSolveTime(0),_factorTime(0),_startTime(MPI_Wtime()),
  _miscTime(0),
  _quadEx(NULL),_quadImex(NULL),
  _fault(NULL),_momBal(NULL),_he(D)
{
  #if VERBOSE > 1
    std::string funcName = "Mediator::Mediator()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  _startTime = MPI_Wtime();

  loadSettings(D._file);
  checkInput();

  // set up time integration member
  if (_timeIntegrator.compare("FEuler")==0) {
    _quadEx = new FEuler(_maxStepCount,_maxTime,_initDeltaT,D._timeControlType);
  }
  else if (_timeIntegrator.compare("RK32")==0) {
    _quadEx = new RK32(_maxStepCount,_maxTime,_initDeltaT,D._timeControlType);
  }
  else if (_timeIntegrator.compare("IMEX")==0) {
    _quadImex = new OdeSolverImex(_maxStepCount,_maxTime,_initDeltaT,D._timeControlType);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: timeIntegrator type not understood\n");
    assert(0); // automatically fail
  }

  if (_hydraulicCoupling.compare("coupled")==0 || _hydraulicCoupling.compare("uncoupled")==0) {
    _fault = new SymmFault_Hydr(D,_he);
  }
  else { _fault = new SymmFault(D,_he); }

  // initiate momentum balance equation
  if (D._bulkDeformationType.compare("linearElastic")==0) { _momBal = new LinearElastic(D,_fault->_tauQSP); }
  else if (D._bulkDeformationType.compare("powerLaw")==0) { _momBal = new PowerLaw(D,_he,_fault->_tauQSP); }

  writeContext();

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


Mediator::~Mediator()
{
  #if VERBOSE > 1
    std::string funcName = "Mediator::~Mediator()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  map<string,Vec>::iterator it;
  for (it = _varEx.begin(); it!=_varEx.end(); it++ ) {
    VecDestroy(&it->second);
  }
  for (it = _varIm.begin(); it!=_varIm.end(); it++ ) {
    VecDestroy(&it->second);
  }

  delete _quadImex; _quadImex = NULL;
  delete _quadEx;   _quadEx = NULL;
  delete _momBal;   _momBal = NULL;
  delete _fault;    _fault = NULL;

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// loads settings from the input text file
PetscErrorCode Mediator::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;
#if VERBOSE > 1
    std::string funcName = "HeatEquation::loadSettings()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  PetscMPIInt rank,size;
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  ifstream infile( file );
  string line,var;
  size_t pos = 0;
  while (getline(infile, line))
  {
    istringstream iss(line);
    pos = line.find(_delim); // find position of the delimiter
    var = line.substr(0,pos);

    if (var.compare("thermalCoupling")==0) {
      _thermalCoupling = line.substr(pos+_delim.length(),line.npos).c_str();
    }
    else if (var.compare("hydraulicCoupling")==0) {
      _hydraulicCoupling = line.substr(pos+_delim.length(),line.npos).c_str();
    }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// Check that required fields have been set by the input file
PetscErrorCode Mediator::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::checkInput";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  assert(_thermalCoupling.compare("coupled")==0 ||
      _thermalCoupling.compare("uncoupled")==0 ||
      _thermalCoupling.compare("no")==0 );

  assert(_timeIntegrator.compare("FEuler")==0 ||
      _timeIntegrator.compare("RK32")==0 ||
      _timeIntegrator.compare("IMEX")==0 );

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif
  return ierr;
}

// initiate variables to be integrated in time
PetscErrorCode Mediator::initiateIntegrand()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mediator::initiateIntegrand()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  _momBal->initiateIntegrand(_initTime,_varEx,_varIm);
  _fault->initiateIntegrand(_initTime,_varEx,_varIm);

  if (_thermalCoupling.compare("coupled")==0 || _thermalCoupling.compare("uncoupled")==0) {
    _he.initiateIntegrand(_initTime,_varEx,_varIm);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

// monitoring function for explicit integration
PetscErrorCode Mediator::timeMonitor(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;
  _stepCount = stepCount;
  _currTime = time;
  #if VERBOSE > 1
    std::string funcName = "Mediator::timeMonitor for explicit";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
double startTime = MPI_Wtime();

  if ( stepCount % _stride1D == 0) {
    ierr = _momBal->writeStep1D(stepCount,time); CHKERRQ(ierr);
    ierr = _fault->writeStep(_stepCount,time); CHKERRQ(ierr);
  }

  if ( stepCount % _stride2D == 0) {
    ierr = _momBal->writeStep2D(stepCount,time);CHKERRQ(ierr);
  }

  if (stepCount % 50 == 0) {
    PetscScalar maxTimeStep_tot, maxDeltaT_momBal = 0.0;
    _momBal->computeMaxTimeStep(maxDeltaT_momBal);
    maxTimeStep_tot = min(_maxDeltaT,maxDeltaT_momBal);
    if (_timeIntegrator.compare("IMEX")==0) {
        _quadImex->setTimeStepBounds(_minDeltaT,maxTimeStep_tot);CHKERRQ(ierr);
    }
    else { _quadEx->setTimeStepBounds(_minDeltaT,maxTimeStep_tot);CHKERRQ(ierr); }
  }

_writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 0
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %.15e\n",stepCount,_currTime);CHKERRQ(ierr);
  #endif
  return ierr;
}

// monitoring function for IMEX integration
PetscErrorCode Mediator::timeMonitor(const PetscScalar time,const PetscInt stepCount,
      const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,const map<string,Vec>& varIm)
{
  PetscErrorCode ierr = 0;
  _stepCount = stepCount;
  _currTime = time;
  #if VERBOSE > 1
    std::string funcName = "Mediator::timeMonitor for IMEX";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
double startTime = MPI_Wtime();
  timeMonitor(time,stepCount,varEx,dvarEx);

  if ( stepCount % _stride1D == 0) {
    ierr = _he.writeStep1D(_stepCount,time); CHKERRQ(ierr);
  }

  if ( stepCount % _stride2D == 0) {
    ierr = _he.writeStep2D(_stepCount,time);CHKERRQ(ierr);
  }

_writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 0
    //~ ierr = PetscPrintf(PETSC_COMM_WORLD,"%i %.15e\n",stepCount,_currTime);CHKERRQ(ierr);
  #endif
  return ierr;
}


PetscErrorCode Mediator::view()
{
  PetscErrorCode ierr = 0;

  double totRunTime = MPI_Wtime() - _startTime;

  if (_timeIntegrator.compare("IMEX")==0) { ierr = _quadImex->view(); _he.view(); }
  if (_timeIntegrator.compare("RK32")==0) { ierr = _quadEx->view(); }

  _momBal->view(_integrateTime);
  _fault->view(_integrateTime);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Mediator Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in integration (s): %g\n",_integrateTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent writing output: %g\n",_writeTime/totRunTime*100.);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode Mediator::writeContext()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mediator::writeContext";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  _momBal->writeContext();
  _he.writeContext();
  _fault->writeContext();

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode Mediator::integrate()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mediator::integrate";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime = MPI_Wtime();

  initiateIntegrand();
  _stepCount = 0;

  if (_timeIntegrator.compare("IMEX")==0) {
    _quadImex->setTolerance(_atol);CHKERRQ(ierr);
    _quadImex->setTimeStepBounds(_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
    ierr = _quadImex->setTimeRange(_initTime,_maxTime);
    ierr = _quadImex->setInitialConds(_varEx,_varIm);CHKERRQ(ierr);

    // control which fields are used to select step size
    ierr = _quadImex->setErrInds(_timeIntInds);

    ierr = _quadImex->integrate(this);CHKERRQ(ierr);
  }
  else {
    _quadEx->setTolerance(_atol);CHKERRQ(ierr);
    _quadEx->setTimeStepBounds(_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
    ierr = _quadEx->setTimeRange(_initTime,_maxTime);
    ierr = _quadEx->setInitialConds(_varEx);CHKERRQ(ierr);

    // control which fields are used to select step size
    ierr = _quadEx->setErrInds(_timeIntInds);

    ierr = _quadEx->integrate(this);CHKERRQ(ierr);
  }


  _integrateTime += MPI_Wtime() - startTime;
  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// explicit time stepping
PetscErrorCode Mediator::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;

  // update fields based on varEx, varIm
  _momBal->updateFields(time,varEx,_varIm);
  _fault->updateFields(time,varEx,_varIm);

  ierr = _momBal->d_dt(time,varEx,dvarEx); CHKERRQ(ierr);

  // update fields on fault
  ierr = _fault->setTauQS(_momBal->_sxy,_momBal->_sxz); CHKERRQ(ierr);

  if (!_bcLTauQS) {
    ierr = _fault->d_dt(time,varEx,dvarEx); // sets rates for slip and state
  }
  else {
    VecSet(dvarEx.find("psi")->second,0.0); // dstate psi
    VecSet(dvarEx.find("slip")->second,0.0); // slip vel
  }
  //~ ierr = _momBal->computeTotalStrainRates(time,varEx,dvarEx); CHKERRQ(ierr);

  return ierr;
}


// implicit/explicit time stepping
PetscErrorCode Mediator::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx,
      map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "Mediator::d_dt";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // update fields based on varEx, varIm
  _momBal->updateFields(time,varEx,varImo);
  _fault->updateFields(time,varEx,varImo);

  ierr = d_dt(time,varEx,dvarEx);CHKERRQ(ierr);

  if (varIm.find("Temp") != varIm.end()) {
    Vec sdev = NULL;
    _momBal->getSigmaDev(sdev);
    ierr = _he.be(time,dvarEx.find("slip")->second,_fault->_tauQSP,
      dvarEx.find("gVxy")->second,dvarEx.find("gVxz")->second,
      sdev,varIm.find("Temp")->second,varImo.find("Temp")->second,dt);CHKERRQ(ierr);
    // arguments: time, slipVel, txy, sigmadev, dgxy, dgxz, T, dTdt
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// Outputs data at each time step.
PetscErrorCode Mediator::debug(const PetscReal time,const PetscInt stepCount,
                         const map<string,Vec>& varEx,const map<string,Vec>& dvarEx,const char *stage)
{
  PetscErrorCode ierr = 0;

#if ODEPRINT > 0
  PetscInt       Istart,Iend;
  PetscScalar    bcRval,uVal,psiVal,velVal,dQVal,tauQS;


#endif
  return ierr;
}

PetscErrorCode Mediator::measureMMSError()
{
  PetscErrorCode ierr = 0;

  //~ _momBal->measureMMSError(_currTime);
  // if (_thermalCoupling.compare("coupled")==0 || _thermalCoupling.compare("uncoupled")==0) {
  //   ierr = _he.measureMMSError(_currTime);
  // }

  _fault->measureMMSError(_currTime);


  return ierr;
}



