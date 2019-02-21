#include "strikeSlip_linearElastic_qd.hpp"

#define FILENAME "strikeSlip_linearElastic_qd.cpp"

using namespace std;

StrikeSlip_LinearElastic_qd::StrikeSlip_LinearElastic_qd(Domain&D)
: _D(&D),_delim(D._delim),_isMMS(D._isMMS),
  _outputDir(D._outputDir),_inputDir(D._inputDir),_loadICs(D._loadICs),
  _vL(1e-9),
  _thermalCoupling("no"),_heatEquationType("transient"),
  _hydraulicCoupling("no"),_hydraulicTimeIntType("explicit"),
  _guessSteadyStateICs(0.),_forcingType("no"),_faultTypeScale(2.0),
  _timeIntegrator("RK43"),_timeControlType("PID"),
  _stride1D(1),_stride2D(1),_maxStepCount(1e8), _ckpt(0), _ckptNumber(0),
  _interval(500),_initTime(0),_currTime(0),_maxTime(1e15),
  _minDeltaT(-1),_maxDeltaT(1e10),
  _stepCount(0),_timeStepTol(1e-8),_initDeltaT(1e-3),_normType("L2_absolute"),
  _integrateTime(0),_writeTime(0),_linSolveTime(0),_factorTime(0),_startTime(MPI_Wtime()),_totalRunTime(0),
  _miscTime(0),_timeV1D(NULL),_dtimeV1D(NULL),_timeV2D(NULL),_forcingVal(0),
  _bcRType("remoteLoading"),_bcTType("freeSurface"),_bcLType("symmFault"),_bcBType("freeSurface"),
  _quadEx(NULL),_quadImex(NULL),
  _fault(NULL),_material(NULL),_he(NULL),_p(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::StrikeSlip_LinearElastic_qd()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings(D._file);
  checkInput();
  parseBCs();
v
  // heat equation
  if (_thermalCoupling.compare("no") != 0) {
    _he = new HeatEquation(D);
  }

  // fault
  _body2fault = &(D._scatters["body2L"]);
  _fault = new Fault_qd(D,D._scatters["body2L"],_faultTypeScale);
  if (_thermalCoupling.compare("no")!=0 && _stateLaw.compare("flashHeating") == 0) {
    _fault->setThermalFields(_he->_Tamb,_he->_k,_he->_c);
  }

  // pressure diffusion equation
  if (_hydraulicCoupling.compare("no") != 0) {
    _p = new PressureEq(D);
  }
  else if (_hydraulicCoupling.compare("coupled")==0) {
    _fault->setSNEff(_p->_p);
  }

  // initiate momentum balance equation
  if (_guessSteadyStateICs) {
    _material = new LinearElastic(D,_mat_bcRType,_mat_bcTType,"Neumann",_mat_bcBType);
  }
  else {
    _material = new LinearElastic(D,_mat_bcRType,_mat_bcTType,_mat_bcLType,_mat_bcBType);
  }

  // body forcing term for ice stream
  _forcingTerm = NULL; _forcingTermPlain = NULL;
  if (_forcingType.compare("iceStream")==0) {
    constructIceStreamForcingTerm();
  }

  computeMinTimeStep(); // compute min allowed time step for adaptive time stepping method

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}


// destructor
StrikeSlip_LinearElastic_qd::~StrikeSlip_LinearElastic_qd()
{
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::~StrikeSlip_LinearElastic_qd()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  {
    map<string,Vec>::iterator it;
    for (it = _varEx.begin(); it!=_varEx.end(); it++ ) {
      VecDestroy(&it->second);
    }
    for (it = _varIm.begin(); it!=_varIm.end(); it++ ) {
      VecDestroy(&it->second);
    }
  }

  {  // destroy viewers for steady state iteration
    map<string,pair<PetscViewer,string>>::iterator it;
    for (it = _viewers.begin(); it!=_viewers.end(); it++ ) {
      PetscViewerDestroy(& (_viewers[it->first].first) );
    }
  }

  PetscViewerDestroy(&_timeV1D);
  PetscViewerDestroy(&_dtimeV1D);
  PetscViewerDestroy(&_timeV2D);

  delete _quadImex;    _quadImex = NULL;
  delete _quadEx;      _quadEx = NULL;
  delete _material;    _material = NULL;
  delete _fault;       _fault = NULL;
  delete _he;          _he = NULL;
  delete _p;           _p = NULL;

  VecDestroy(&_forcingTerm);
  VecDestroy(&_forcingTermPlain);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// loads settings from the input text file
PetscErrorCode StrikeSlip_LinearElastic_qd::loadSettings(const char *file)
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

  // load checkpoint number
  loadValueFromCheckpoint(_outputDir, "_ckptNumber", _ckptNumber);

  ifstream infile( file );
  string line, var, rhs, rhsFull;
  size_t pos = 0;
  
  while (getline(infile, line)) {
    istringstream iss(line);
    pos = line.find(_delim); // find position of the delimiter
    var = line.substr(0,pos);
    rhs = "";
    if (line.length() > (pos + _delim.length())) {
      rhs = line.substr(pos+_delim.length(),line.npos);
    }
    rhsFull = rhs; // everything after _delim

    // interpret everything after the appearance of a space on the line as a comment
    pos = rhs.find(" ");
    rhs = rhs.substr(0,pos);

    if (var.compare("thermalCoupling")==0) { _thermalCoupling = rhs.c_str(); }
    else if (var.compare("hydraulicCoupling")==0) { _hydraulicCoupling = rhs.c_str(); }
    else if (var.compare("stateLaw")==0) { _stateLaw = rhs.c_str(); }
    else if (var.compare("guessSteadyStateICs")==0) { _guessSteadyStateICs = atoi( rhs.c_str() ); }
    else if (var.compare("forcingType")==0) { _forcingType = rhs.c_str(); }

    // time integration properties
    else if (var.compare("timeIntegrator")==0) { _timeIntegrator = rhs; }
    else if (var.compare("timeControlType")==0) { _timeControlType = rhs; }
    else if (var.compare("stride1D")==0){ _stride1D = atoi(rhs.c_str()); }
    else if (var.compare("stride2D")==0){ _stride2D = atoi(rhs.c_str()); }
    else if (var.compare("ckpt") == 0) { _ckpt = atoi(rhs.c_str()); }
    else if (var.compare("interval")==0) { _interval = atoi(rhs.c_str()); }

    /* if checkpoint number > 0 (i.e. there has been a checkpoint before), then set _maxStepCount to _interval */
    else if (var.compare("maxStepCount")==0) {
      if (_ckptNumber > 0) {
	_maxStepCount = _interval;
      }
      else {
	_maxStepCount = atoi(rhs.c_str());
      }
    }
    // if checkpoint number > 0, load _initTime from checkpoint file
    else if (var.compare("initTime")==0) {
      if (_ckptNumber > 0) {
	loadValueFromCheckpoint(_outputDir, "_currTime", _initTime);
      }
      else {
	_initTime = atof( rhs.c_str() );
      }
    }
    
    else if (var.compare("maxTime")==0) { _maxTime = atof( rhs.c_str() ); }
    else if (var.compare("minDeltaT")==0) { _minDeltaT = atof( rhs.c_str() ); }
    else if (var.compare("maxDeltaT")==0) {_maxDeltaT = atof( rhs.c_str() ); }
    else if (var.compare("initDeltaT")==0) { _initDeltaT = atof( rhs.c_str() ); }
    else if (var.compare("timeStepTol")==0) { _timeStepTol = atof( rhs.c_str() ); }
    else if (var.compare("timeIntInds")==0) { loadVectorFromInputFile(rhsFull,_timeIntInds); }
    else if (var.compare("scale")==0) { loadVectorFromInputFile(rhsFull,_scale); }
    else if (var.compare("normType")==0) { _normType = rhs.c_str(); }
    else if (var.compare("vL")==0) { _vL = atof( rhs.c_str() ); }
    else if (var.compare("bodyForce")==0) { _forcingVal = atof( rhs.c_str() ); }

    // boundary condition types for momentum balance equation
    else if (var.compare("momBal_bcR_qd")==0) { _bcRType = rhs.c_str(); }
    else if (var.compare("momBal_bcT_qd")==0) { _bcTType = rhs.c_str(); }
    else if (var.compare("momBal_bcL_qd")==0) { _bcLType = rhs.c_str(); }
    else if (var.compare("momBal_bcB_qd")==0) { _bcBType = rhs.c_str(); }
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  return ierr;
}


// Check that required fields have been set by the input file
PetscErrorCode StrikeSlip_LinearElastic_qd::checkInput()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "PowerLaw::checkInput";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  assert(_guessSteadyStateICs == 0 || _guessSteadyStateICs == 1);
  if (_loadICs) { assert(_guessSteadyStateICs == 0); }

  assert(_thermalCoupling.compare("coupled")==0 ||
    _thermalCoupling.compare("uncoupled")==0 ||
    _thermalCoupling.compare("no")==0 );

  assert(_hydraulicCoupling.compare("coupled")==0 ||
    _hydraulicCoupling.compare("uncoupled")==0 ||
    _hydraulicCoupling.compare("no")==0 );

  assert(_forcingType.compare("iceStream")==0 || _forcingType.compare("no")==0 );

  assert(_timeIntegrator.compare("FEuler")==0 ||
    _timeIntegrator.compare("RK32")==0 ||
    _timeIntegrator.compare("RK43")==0 ||
    _timeIntegrator.compare("RK32_WBE")==0 ||
    _timeIntegrator.compare("RK43_WBE")==0 );

  assert(_timeControlType.compare("P")==0 ||
    _timeControlType.compare("PI")==0 ||
    _timeControlType.compare("PID")==0 );

  if (_initDeltaT<_minDeltaT || _initDeltaT < 1e-14) {
    _initDeltaT = _minDeltaT;
  }

  assert(_ckpt >= 0);
  assert(_interval > 0);
  assert(_maxStepCount >= 0);
  assert(_initTime >= 0);
  assert(_maxTime >= 0 && _maxTime>=_initTime);
  assert(_timeStepTol >= 1e-14);
  assert(_maxDeltaT >= 1e-14  &&  _maxDeltaT >= _minDeltaT);
  assert(_initDeltaT>0 && _initDeltaT>=_minDeltaT && _initDeltaT<=_maxDeltaT);
  
  // check boundary condition types for momentum balance equation
  assert(_bcRType.compare("freeSurface")==0 || _bcRType.compare("remoteLoading")==0);
  assert(_bcTType.compare("freeSurface")==0 || _bcTType.compare("remoteLoading")==0);
  assert(_bcLType.compare("symmFault")==0   || _bcLType.compare("rigidFault")==0 );
  assert(_bcBType.compare("freeSurface")==0 || _bcBType.compare("remoteLoading")==0);

  if (_stateLaw.compare("flashHeating")==0) {
    assert(_thermalCoupling.compare("no")!=0);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  return ierr;
}


// compute recommended smallest time step based on grid spacing and shear wave speed
// Note: defaults to user specified value
// recommended minDeltaT <= min(dy/cs, dz/cs)
PetscErrorCode StrikeSlip_LinearElastic_qd::computeMinTimeStep()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::computeTimeStep";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // compute grid spacing in y and z
  Vec dy, dz;
  VecDuplicate(_D->_y,&dy);
  VecDuplicate(_D->_y,&dz);
  if (_D->_sbpType.compare("mfc_coordTrans")==0){
    Mat J,Jinv,qy,rz,yq,zr;
    ierr = _material->_sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    MatGetDiagonal(yq, dy);
    VecScale(dy,1.0/(_D->_Ny-1));
    MatGetDiagonal(zr, dz);
    VecScale(dz,1.0/(_D->_Nz-1));
  }
  else {
    VecSet(dy,_D->_Ly/(_D->_Ny-1.0));
    VecSet(dz,_D->_Lz/(_D->_Nz-1.0));
  }

  // compute time for shear wave to travel one dy or dz
  Vec ts_dy,ts_dz;
  VecDuplicate(_D->_y,&ts_dy);
  VecDuplicate(_D->_z,&ts_dz);
  VecPointwiseDivide(ts_dy,dy,_material->_cs);
  VecPointwiseDivide(ts_dz,dz,_material->_cs);

  PetscScalar min_ts_dy, min_ts_dz;
  VecMin(ts_dy,NULL,&min_ts_dy);
  VecMin(ts_dz,NULL,&min_ts_dz);

  // clean up memory usage
  VecDestroy(&dy);
  VecDestroy(&dz);
  VecDestroy(&ts_dy);
  VecDestroy(&ts_dz);

  // smallest reasonable time step
  PetscScalar min_deltaT = min(min_ts_dy,min_ts_dz);

  // provide if not user specified
  if (_minDeltaT == -1) {
    _minDeltaT = min_deltaT;
  } 
  else if (_minDeltaT > min_deltaT) {
    PetscPrintf(PETSC_COMM_WORLD,"Warning: minimum requested time step (minDeltaT) is larger than recommended.");
    PetscPrintf(PETSC_COMM_WORLD," Requested: %e s, Recommended (min(dy/cs,dz/cs)): %e s\n",_minDeltaT,min_deltaT);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// parse boundary conditions
PetscErrorCode StrikeSlip_LinearElastic_qd::parseBCs()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::parseBCs()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_bcRType.compare("symmFault")==0 || _bcRType.compare("rigidFault")==0 || _bcRType.compare("remoteLoading")==0) {
    _mat_bcRType = "Dirichlet";
  }
  else if (_bcRType.compare("freeSurface")==0 || _bcRType.compare("outGoingCharacteristics")==0) {
    _mat_bcRType = "Neumann";
  }

  if (_bcTType.compare("symmFault")==0 || _bcTType.compare("rigidFault")==0 || _bcTType.compare("remoteLoading")==0) {
    _mat_bcTType = "Dirichlet";
  }
  else if (_bcTType.compare("freeSurface")==0 || _bcTType.compare("outGoingCharacteristics")==0) {
    _mat_bcTType = "Neumann";
  }

  if (_bcLType.compare("symmFault")==0 || _bcLType.compare("rigidFault")==0 || _bcLType.compare("remoteLoading")==0) {
    _mat_bcLType = "Dirichlet";
  }
  else if (_bcLType.compare("freeSurface")==0 || _bcLType.compare("outGoingCharacteristics")==0) {
    _mat_bcLType = "Neumann";
  }

  if (_bcBType.compare("symmFault")==0 || _bcBType.compare("rigidFault")==0 || _bcBType.compare("remoteLoading")==0) {
    _mat_bcBType = "Dirichlet";
  }
  else if (_bcBType.compare("freeSurface")==0 || _bcBType.compare("outGoingCharacteristics")==0) {
    _mat_bcBType = "Neumann";
  }

  // determine if material is symmetric about the fault, or if one side is rigid
  _faultTypeScale = 2.0;
  if (_bcLType.compare("rigidFault")==0 ) {
    _faultTypeScale = 1.0;
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// initiate variables to be integrated in time
PetscErrorCode StrikeSlip_LinearElastic_qd::initiateIntegrand()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::initiateIntegrand()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  Mat A;
  _material->_sbp->getA(A);
  _material->setupKSP(_material->_sbp,_material->_ksp,_material->_pc,A);

  // if to perform MMS test, also set MMS initial condition
  if (_isMMS) {
    _material->setMMSInitialConditions(_initTime);
  }

  Vec slip;
  VecDuplicate(_material->_bcL,&slip);
  VecCopy(_material->_bcL,slip);
  VecScale(slip,_faultTypeScale);
  ierr = loadVecFromInputFile(slip,_inputDir,"slip"); CHKERRQ(ierr);
  _varEx["slip"] = slip;

  if (_guessSteadyStateICs) {
    solveSS();
  }

  _fault->initiateIntegrand(_initTime,_varEx);

  if (_thermalCoupling.compare("no")!=0 ) {
     _he->initiateIntegrand(_initTime,_varEx,_varIm);
     _fault->updateTemperature(_he->_T);
  }

  if (_hydraulicCoupling.compare("no")!=0 ) {
     _p->initiateIntegrand(_initTime,_varEx,_varIm);
  }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// monitoring function for ode solvers
PetscErrorCode StrikeSlip_LinearElastic_qd::timeMonitor(const PetscScalar time,const PetscScalar deltaT, const PetscInt stepCount, int& stopIntegration)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::timeMonitor";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  
  double startTime = MPI_Wtime();
  _stepCount = stepCount;
  _deltaT = deltaT;
  _currTime = time;

  /* write solution to file (1D) when _currTime == _maxTime (simulation ends),
   * or if _stride1D > 0 and stepCount is an integer multiple of _stride1D */

  /* TODO: inspect the writeStep functions called by each object and change
   * the output file names depending on the checkpoint reached */

  if (_currTime == _maxTime || (_stride1D > 0 && stepCount % _stride1D == 0)) {
    ierr = writeStep1D(stepCount,time,_outputDir); CHKERRQ(ierr);
    ierr = _material->writeStep1D(_stepCount,time,_outputDir); CHKERRQ(ierr);
    ierr = _fault->writeStep(_stepCount,time,_outputDir); CHKERRQ(ierr);

    // check if there is hydraulic coupling and write result
    if (_hydraulicCoupling.compare("no")!=0) {
      ierr = _p->writeStep(_stepCount,time,_outputDir); CHKERRQ(ierr);
    }

    // check if there is thermal coupling and write result
    if (_thermalCoupling.compare("no")!=0) {
      ierr =  _he->writeStep1D(_stepCount,time,_outputDir); CHKERRQ(ierr);
    }
  }

  /* write solution to file (2D) when _currTime == _maxTime (simulation ends),
   * or if _stride2D > 0 and stepCount is an integer multiple of _stride2D */

  /* TODO: inspect the writeStep functions called by each object and change
   * the output file names depending on the checkpoint reached */
  
  if (_currTime == _maxTime || (_stride2D > 0 && stepCount % _stride2D == 0)) {
    ierr = writeStep2D(stepCount,time,_outputDir); CHKERRQ(ierr);
    ierr = _material->writeStep2D(_stepCount,time,_outputDir);CHKERRQ(ierr);

    // check if there is thermal coupling and write result
    if (_thermalCoupling.compare("no")!=0) {
      ierr =  _he->writeStep2D(_stepCount,time,_outputDir);CHKERRQ(ierr);
    }
  }

  #if VERBOSE > 0
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%i: t = %.15e s, dt = %.5e \n",stepCount,_currTime,_deltaT);CHKERRQ(ierr);
  #endif
  
  _writeTime += MPI_Wtime() - startTime;
  
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// write out time and _deltaT at each time step
PetscErrorCode StrikeSlip_LinearElastic_qd::writeStep1D(const PetscInt stepCount, const PetscScalar time, const std::string outputDir)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::writeStep1D";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // TODO: change this to depend on checkpointing
  // _timeV1D is a PetscViewer
  if (_timeV1D==NULL) {
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(outputDir+"med_time1D.txt").c_str(),&_timeV1D);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_timeV1D, "%.15e\n",time);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(outputDir+"med_dt1D.txt").c_str(),&_dtimeV1D);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_dtimeV1D, "%.15e\n",_deltaT);CHKERRQ(ierr);
  }
  else {
    ierr = PetscViewerASCIIPrintf(_timeV1D, "%.15e\n",time);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_dtimeV1D, "%.15e\n",_deltaT);CHKERRQ(ierr);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// write out time at each time step
PetscErrorCode StrikeSlip_LinearElastic_qd::writeStep2D(const PetscInt stepCount, const PetscScalar time,const std::string outputDir)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::writeStep1D";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_timeV2D==NULL) {
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,(outputDir+"med_time2D.txt").c_str(),&_timeV2D);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(_timeV2D, "%.15e\n",time);CHKERRQ(ierr);
  }
  else {
    ierr = PetscViewerASCIIPrintf(_timeV2D, "%.15e\n",time);CHKERRQ(ierr);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// append new data from start to end of of current checkpoint to existing data files
PetscErrorCode StrikeSlip_LinearElastic_qd::saveCheckpoint(const PetscInt start, const PetscInt interval, const string outputDir) {
  PetscErrorCode ierr = 0;

#if VERBOSE > 1
  string funcName = "StrikeSlip_LinearElastic_qd::saveCheckpoint";
  PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
#endif

  // check if current checkpoint files has finished writing

  // appends data from current checkpoint files to original data files
  
  return ierr;
}


// delete previous checkpoint file
PetscErrorCode StrikeSlip_LinearElastic_qd::deleteCheckpoint(const PetscInt start, const PetscInt interval, const string outputDir) {
  PetscErrorCode ierr = 0;

#if VERBOSE > 1
  string funcName = "StrikeSlip_LinearElastic_qd::writeNewCheckpoint";
  PetscPrintf(PETSC_COMM_WORLD, "Starting %s in %s\n", funcName.c_str(), FILENAME);
#endif

  // deletes the previous checkpoint files

  // increments the start variable

  return ierr;
}
  

// print fields to screen
PetscErrorCode StrikeSlip_LinearElastic_qd::view()
{
  PetscErrorCode ierr = 0;

  double totRunTime = MPI_Wtime() - _startTime;

  _material->view(_integrateTime);
  _fault->view(_integrateTime);

  // set up view for various settings
  if ((_timeIntegrator.compare("RK32")==0 || _timeIntegrator.compare("RK43")==0) && _quadEx!=NULL) {
    ierr = _quadEx->view();
  }

  if ((_timeIntegrator.compare("RK32_WBE")==0 || _timeIntegrator.compare("RK43_WBE")==0) && _quadImex!=NULL) {
    ierr = _quadImex->view();
  }

  if (_hydraulicCoupling.compare("no")!=0) {
    _p->view(_integrateTime);
  }

  if (_thermalCoupling.compare("no")!=0) {
    _he->view();
  }

  ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"StrikeSlip_LinearElastic_qd Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in integration (s): %g\n",_integrateTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   total run time (s): %g\n",totRunTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent writing output: %g\n",(_writeTime/_integrateTime)*100.);CHKERRQ(ierr);

  // // output SBP matrices
  // ierr = _material->_sbp->writeOps(_outputDir + "ops_u_"); CHKERRQ(ierr);
  return ierr;
}


// write out context parameters that don't change in time, and that can't be put into a .txt file, e.g. shear modulus, shear wave speed, density
PetscErrorCode StrikeSlip_LinearElastic_qd::writeContext()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::writeContext";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // output scalar fields, only from the first processor in the PetscViewer
  std::string str = _outputDir + "mediator_context.txt";
  PetscViewer    viewer;
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());
  ierr = PetscViewerASCIIPrintf(viewer,"thermalCoupling = %s\n",_thermalCoupling.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"hydraulicCoupling = %s\n",_hydraulicCoupling.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"forcingType = %s\n",_forcingType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"vL = %g\n",_vL);CHKERRQ(ierr);

  // time integration settings
  ierr = PetscViewerASCIIPrintf(viewer,"timeIntegrator = %s\n",_timeIntegrator.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"timeControlType = %s\n",_timeControlType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride1D = %i\n",_stride1D);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride2D = %i\n",_stride1D);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"maxStepCount = %i\n",_maxStepCount);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"initTime = %.15e # (s)\n",_initTime);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"maxTime = %.15e # (s)\n",_maxTime);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"minDeltaT = %.15e # (s)\n",_minDeltaT);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"maxDeltaT = %.15e # (s)\n",_maxDeltaT);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"initDeltaT = %.15e # (s)\n",_initDeltaT);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"timeStepTol = %g\n",_timeStepTol);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"timeIntInds = %s\n",vector2str(_timeIntInds).c_str());CHKERRQ(ierr);
  if (_scale.size() > 0) {
    ierr = PetscViewerASCIIPrintf(viewer,"scale = %s\n",vector2str(_scale).c_str());CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"normType = %s\n",_normType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

  // boundary conditions for momentum balance equation
  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcR = %s\n",_bcRType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcT = %s\n",_bcTType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcL = %s\n",_bcLType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"momBal_bcB = %s\n",_bcBType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"faultTypeScale = %g\n",_faultTypeScale);CHKERRQ(ierr);

  // free memory
  PetscViewerDestroy(&viewer);

  _material->writeContext(_outputDir);
  _fault->writeContext(_outputDir);

  if (_thermalCoupling.compare("no")!=0) {
    _he->writeContext(_outputDir);
  }

  if (_hydraulicCoupling.compare("no")!=0) {
    _p->writeContext(_outputDir);
  }

  if (_forcingType.compare("iceStream")==0) {
    ierr = writeVec(_forcingTermPlain,_outputDir + "momBal_forcingTerm"); CHKERRQ(ierr);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


//======================================================================
// Adaptive time stepping functions
//======================================================================

PetscErrorCode StrikeSlip_LinearElastic_qd::integrate()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::integrate";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime = MPI_Wtime();

  initiateIntegrand(); // put initial conditions into var for integration
  _stepCount = 0;

  // initialize time integrator
  if (_timeIntegrator.compare("FEuler")==0) {
    _quadEx = new FEuler(_maxStepCount,_maxTime,_initDeltaT,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK32")==0) {
    _quadEx = new RK32(_maxStepCount,_maxTime,_initDeltaT,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK43")==0) {
    _quadEx = new RK43(_maxStepCount,_maxTime,_initDeltaT,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK32_WBE")==0) {
    _quadImex = new RK32_WBE(_maxStepCount,_maxTime,_initDeltaT,_timeControlType);
  }
  else if (_timeIntegrator.compare("RK43_WBE")==0) {
    _quadImex = new RK43_WBE(_maxStepCount,_maxTime,_initDeltaT,_timeControlType);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: timeIntegrator type not understood\n");
    assert(0); // automatically fail
  }

  // with backward Euler, implicit time stepping
  if (_timeIntegrator.compare("RK32_WBE")==0 || _timeIntegrator.compare("RK43_WBE")==0) {
    _quadImex->setTolerance(_timeStepTol);CHKERRQ(ierr);
    _quadImex->setTimeStepBounds(_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
    ierr = _quadImex->setTimeRange(_initTime,_maxTime);
    ierr = _quadImex->setToleranceType(_normType); CHKERRQ(ierr);
    ierr = _quadImex->setInitialConds(_varEx,_varIm);CHKERRQ(ierr);
    // control which fields are used to select step size
    ierr = _quadImex->setErrInds(_timeIntInds,_scale);
    // performs integration according to odeSolver class
    ierr = _quadImex->integrate(this);CHKERRQ(ierr);
  }

  // explicit time stepping
  else {
    _quadEx->setTolerance(_timeStepTol);CHKERRQ(ierr);
    _quadEx->setTimeStepBounds(_minDeltaT,_maxDeltaT);CHKERRQ(ierr);
    ierr = _quadEx->setTimeRange(_initTime,_maxTime);
    ierr = _quadEx->setToleranceType(_normType); CHKERRQ(ierr);
    ierr = _quadEx->setInitialConds(_varEx);CHKERRQ(ierr);
    // control which fields are used to select step size
    ierr = _quadEx->setErrInds(_timeIntInds,_scale);
    // performs integration according to odeSolver class
    ierr = _quadEx->integrate(this);CHKERRQ(ierr);
  }

  // calculate time used in integration
  _integrateTime = MPI_Wtime() - startTime;

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// purely explicit time stepping
// note that the heat equation never appears here because it is only ever solved implicitly
PetscErrorCode StrikeSlip_LinearElastic_qd::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;

  // update for momBal; var holds slip, bcL is displacement at y=0+
  if (_bcLType.compare("symmFault")==0 || _bcLType.compare("rigidFault")==0) {
    ierr = VecCopy(varEx.find("slip")->second,_material->_bcL);CHKERRQ(ierr);
    ierr = VecScale(_material->_bcL,1.0/_faultTypeScale);CHKERRQ(ierr);
  }
  if (_bcRType.compare("remoteLoading")==0) {
    ierr = VecSet(_material->_bcR,_vL*time/_faultTypeScale);CHKERRQ(ierr);
    ierr = VecAXPY(_material->_bcR,1.0,_material->_bcRShift);CHKERRQ(ierr);
  }

  _fault->updateFields(time,varEx);

  if ((varEx.find("pressure") != varEx.end() || varEx.find("permeability") != varEx.end()) && _hydraulicCoupling.compare("no")!=0){
    _p->updateFields(time,varEx);
  }
  if (_hydraulicCoupling.compare("coupled")==0 && varEx.find("pressure") != varEx.end()) {
    _fault->setSNEff(varEx.find("pressure")->second);
  }

  // compute rates
  ierr = solveMomentumBalance(time,varEx,dvarEx); CHKERRQ(ierr);

  // update fields on fault from other classes
  Vec sxy,sxz,sdev;
  ierr = _material->getStresses(sxy,sxz,sdev);
  ierr = VecScatterBegin(*_body2fault, sxy, _fault->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, sxy, _fault->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  // rates for fault
  ierr = _fault->d_dt(time,varEx,dvarEx); // sets rates for slip and state

  if ((varEx.find("pressure") != varEx.end() || varEx.find("permeability") != varEx.end() ) && _hydraulicCoupling.compare("no")!=0 ){
    _p->d_dt(time,varEx,dvarEx);
  }

  return ierr;
}


// implicit/explicit time stepping
PetscErrorCode StrikeSlip_LinearElastic_qd::d_dt(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx, map<string,Vec>& varIm,const map<string,Vec>& varImo,const PetscScalar dt)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::d_dt";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // update state of each class from integrated variables varEx and varImo

  // update for momBal; var holds slip, bcL is displacement at y=0+
  if (_bcLType.compare("symmFault")==0 || _bcLType.compare("rigidFault")==0) {
    ierr = VecCopy(varEx.find("slip")->second,_material->_bcL);CHKERRQ(ierr);
    ierr = VecScale(_material->_bcL,1.0/_faultTypeScale);CHKERRQ(ierr);
  }
  if (_bcRType.compare("remoteLoading")==0) {
    ierr = VecSet(_material->_bcR,_vL*time/_faultTypeScale);CHKERRQ(ierr);
    ierr = VecAXPY(_material->_bcR,1.0,_material->_bcRShift);CHKERRQ(ierr);
  }

  _fault->updateFields(time,varEx);

  if ( _hydraulicCoupling.compare("no")!=0 ) {
    _p->updateFields(time,varEx,varImo);
  }
  if (varImo.find("Temp") != varImo.end() && _thermalCoupling.compare("coupled")==0) {
    _fault->updateTemperature(varImo.find("Temp")->second);
  }

  // update effective normal stress in fault using pore pressure
  if (_hydraulicCoupling.compare("coupled")==0) {
    _fault->setSNEff(_p->_p);
  }

  // compute rates
  ierr = solveMomentumBalance(time,varEx,dvarEx); CHKERRQ(ierr);

  // update shear stress on fault from momentum balance computation
  Vec sxy,sxz,sdev;
  ierr = _material->getStresses(sxy,sxz,sdev);
  ierr = VecScatterBegin(*_body2fault, sxy, _fault->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, sxy, _fault->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  // rates for fault
  ierr = _fault->d_dt(time,varEx,dvarEx); // sets rates for slip and state

  if ( _hydraulicCoupling.compare("no")!=0 ) {
    _p->d_dt(time,varEx,dvarEx,varIm,varImo,dt);
  }

  // heat equation
  // solve heat equation implicitly
  if (varIm.find("Temp") != varIm.end()) {
    Vec V = dvarEx.find("slip")->second;
    Vec tau = _fault->_tauP;
    Vec gVxy_t = NULL;
    Vec gVxz_t = NULL;
    Vec Told = varImo.find("Temp")->second;
    // arguments: time, slipVel, txy, sigmadev, dgxy, dgxz, T, old T, dt
    ierr = _he->be(time,V,tau,NULL,gVxy_t,gVxz_t,varIm["Temp"],Told,dt); CHKERRQ(ierr);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// momentum balance equation and constitutive laws portion of d_dt
PetscErrorCode StrikeSlip_LinearElastic_qd::solveMomentumBalance(const PetscScalar time,const map<string,Vec>& varEx,map<string,Vec>& dvarEx)
{
  PetscErrorCode ierr = 0;

  // update rhs
  _material->setRHS();
  if (_isMMS) {
    _material->setMMSBoundaryConditions(time);
    _material->addRHS_MMSSource(time,_material->_rhs);
  }

  // add source term for driving the ice stream to rhs Vec
  if (_forcingType.compare("iceStream")==0) {
    VecAXPY(_material->_rhs,1.0,_forcingTerm);
  }

  // compute displacement and stresses
  _material->computeU();
  _material->computeStresses();

  return ierr;
}


// guess at the steady-state solution
PetscErrorCode StrikeSlip_LinearElastic_qd::solveSS()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::solveSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // estimate steady-state conditions for fault, material based on strain rate
  _fault->guessSS(_vL); // sets: slipVel, psi, tau

  // writes tau and psi into file in output directory
  ierr = io_initiateWriteAppend(_viewers, "tau", _fault->_tauP, _outputDir + "SS_tau"); CHKERRQ(ierr);
  ierr = io_initiateWriteAppend(_viewers, "psi", _fault->_psi, _outputDir + "SS_tau"); CHKERRQ(ierr);

  // compute compute u that satisfies tau at left boundary
  VecCopy(_fault->_tauP,_material->_bcL);
  _material->setRHS();
  _material->computeU();
  _material->computeStresses();

  // update fault to contain correct stresses
  Vec sxy,sxz,sdev;
  ierr = _material->getStresses(sxy,sxz,sdev);

  // write slip and stress into file in output directory
  ierr = io_initiateWriteAppend(_viewers, "SS_uSS0", _material->_u, _outputDir + "SS_uSS0"); CHKERRQ(ierr);
  ierr = io_initiateWriteAppend(_viewers, "SS_sxySS", sxy, _outputDir + "SS_sxySS"); CHKERRQ(ierr);

  // scatter body fields to fault vector
  ierr = VecScatterBegin(*_body2fault, sxy, _fault->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, sxy, _fault->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  // update boundary conditions, stresses
  solveSSb();
  _material->changeBCTypes(_mat_bcRType,_mat_bcTType,_mat_bcLType,_mat_bcBType);

  // steady state temperature
  if (_thermalCoupling.compare("no")!=0) {
    ierr = writeVec(_he->_Tamb,_outputDir + "SS_T0"); CHKERRQ(ierr);
    _material->getStresses(sxy,sxz,sdev);
    Vec T;
    VecDuplicate(sxy,&T);
    _he->computeSteadyStateTemp(_currTime,_fault->_slipVel,_fault->_tauP,NULL,NULL,NULL,T);
    ierr = writeVec(_he->_T,_outputDir + "SS_TSS"); CHKERRQ(ierr);

    // free memory
    VecDestroy(&T);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// update the boundary conditions based on new steady state u
PetscErrorCode StrikeSlip_LinearElastic_qd::solveSSb()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::solveSSb";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // adjust u so it has no negative values
  PetscScalar minVal = 0;
  VecMin(_material->_u,NULL,&minVal);
  if (minVal < 0) {
    minVal = abs(minVal);
    Vec temp;
    VecDuplicate(_material->_u,&temp);
    VecSet(temp,minVal);
    VecAXPY(_material->_u,1.,temp);
    // free memory
    VecDestroy(&temp);
  }

   ierr = io_initiateWriteAppend(_viewers, "SS_uSS1", _material->_u, _outputDir + "SS_uSS1"); CHKERRQ(ierr);

  // extract R boundary from u, to set _material->bcR
  VecScatterBegin(_D->_scatters["body2R"], _material->_u, _material->_bcRShift, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(_D->_scatters["body2R"], _material->_u, _material->_bcRShift, INSERT_VALUES, SCATTER_FORWARD);
  VecCopy(_material->_bcRShift,_material->_bcR);

  // extract L boundary from u to set slip, possibly _material->_bcL
  Vec uL;
  VecDuplicate(_material->_bcL,&uL);
  VecScatterBegin(_D->_scatters["body2L"], _material->_u, uL, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(_D->_scatters["body2L"], _material->_u, uL, INSERT_VALUES, SCATTER_FORWARD);

  // write result into file in output directory
  ierr = io_initiateWriteAppend(_viewers, "SS_uL", uL, _outputDir + "SS_uL"); CHKERRQ(ierr);
  ierr = io_initiateWriteAppend(_viewers, "SS_bcRShift", _material->_bcRShift, _outputDir + "SS_bcRShift"); CHKERRQ(ierr);

  VecCopy(uL,_varEx["slip"]);
  VecScale(_varEx["slip"],2.0);
  VecCopy(uL,_material->_bcL);

  // free memory
  VecDestroy(&uL);

  ierr = io_initiateWriteAppend(_viewers, "SS_slip0", _varEx["slip"], _outputDir + "SS_slip0"); CHKERRQ(ierr);

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// constructs the body forcing term for an ice stream
// includes allocation of memory for this forcing term
PetscErrorCode StrikeSlip_LinearElastic_qd::constructIceStreamForcingTerm()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "StrikeSlip_LinearElastic_qd::constructIceStreamForcingTerm";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  // matrix to map the value for the forcing term, which lives on the fault, to all other processors
  Mat MapV;
  MatCreate(PETSC_COMM_WORLD,&MapV);
  MatSetSizes(MapV,PETSC_DECIDE,PETSC_DECIDE,_D->_Ny*_D->_Nz,_D->_Nz);
  MatSetFromOptions(MapV);
  MatMPIAIJSetPreallocation(MapV,_D->_Ny*_D->_Nz,NULL,_D->_Ny*_D->_Nz,NULL);
  MatSeqAIJSetPreallocation(MapV,_D->_Ny*_D->_Nz,NULL);
  MatSetUp(MapV);

  PetscScalar v=1.0;
  PetscInt Ii=0,Istart=0,Iend=0,Jj=0;
  MatGetOwnershipRange(MapV,&Istart,&Iend);
  for (Ii = Istart; Ii < Iend; Ii++) {
    Jj = Ii % _D->_Nz;
    MatSetValues(MapV,1,&Ii,1,&Jj,&v,INSERT_VALUES);
  }
  MatAssemblyBegin(MapV,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(MapV,MAT_FINAL_ASSEMBLY);

  // // compute forcing term for momentum balance equation
  // forcing = (1/Ly) * (tau_ss + eta_rad*V_ss)
  // Vec tauSS = NULL,radDamp=NULL,V=NULL;
  // VecDuplicate(_fault->_eta_rad,&V); VecSet(V,_vL);
  // VecDuplicate(_fault->_eta_rad,&radDamp); VecPointwiseMult(radDamp,_fault->_eta_rad,V);
  // _fault->computeTauRS(tauSS,_vL);
  // VecAXPY(tauSS,1.0,radDamp);
  // VecScale(tauSS,-1./_D->_Ly);

  // VecDuplicate(_material->_u,&_forcingTerm); VecSet(_forcingTerm,0.0);
  // MatMult(MapV,tauSS,_forcingTerm);

  // // free memory
  // MatDestroy(&MapV);
  // VecDestroy(&tauSS);
  // VecDestroy(&radDamp);

  // // compute forcing term for momentum balance equation
  // forcing = - tau_ss / Ly
  // Vec tauSS = NULL;
  // _fault->computeTauRS(tauSS,_vL);
  // VecScale(tauSS,-1./_D->_Ly);

  // VecDuplicate(_material->_u,&_forcingTerm); VecSet(_forcingTerm,0.0);
  // MatMult(MapV,tauSS,_forcingTerm);

  // // free memory
  // MatDestroy(&MapV);
  // VecDestroy(&tauSS);

  // compute forcing term using scalar input
  VecDuplicate(_material->_u,&_forcingTerm);
  VecSet(_forcingTerm,_forcingVal);
  VecDuplicate(_material->_u,&_forcingTermPlain);
  VecCopy(_forcingTerm,_forcingTermPlain);

  // alternatively, load forcing term from user input
  ierr = loadVecFromInputFile(_forcingTerm,_inputDir,"iceForcingTerm"); CHKERRQ(ierr);

  // multiply forcing term H*J if using a curvilinear grid (the H matrix and the Jacobian)
  if (_material->_sbpType.compare("mfc_coordTrans")==0) {
    Vec temp1;
    Mat J,Jinv,qy,rz,yq,zr,H;
    VecDuplicate(_forcingTerm,&temp1);
    ierr = _material->_sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    ierr = MatMult(J,_forcingTerm,temp1); CHKERRQ(ierr);
    _material->_sbp->getH(H);
    ierr = MatMult(H,temp1,_forcingTerm); CHKERRQ(ierr);
    // free memory
    VecDestroy(&temp1);
  }

  // multiply forcing term by H if grid is regular
  else{
    Vec temp1;
    Mat H;
    VecDuplicate(_forcingTerm,&temp1);
    _material->_sbp->getH(H);
    ierr = MatMult(H,_forcingTerm,temp1); CHKERRQ(ierr);
    VecCopy(temp1,_forcingTerm);
    // free memory
    VecDestroy(&temp1);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  return ierr;
}


// measure MMS error for various outputs
PetscErrorCode StrikeSlip_LinearElastic_qd::measureMMSError()
{
  PetscErrorCode ierr = 0;

  _material->measureMMSError(_currTime);
  _he->measureMMSError(_currTime);
  _p->measureMMSError(_currTime);

  return ierr;
}
