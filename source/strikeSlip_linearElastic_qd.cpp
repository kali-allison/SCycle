#include "strikeSlip_linearElastic_qd.hpp"

#define FILENAME "strikeSlip_linearElastic_qd.cpp"

using namespace std;

StrikeSlip_LinearElastic_qd::StrikeSlip_LinearElastic_qd(Domain &D)
  : _D(&D),_delim(D._delim),_isMMS(D._isMMS),_inputDir(D._inputDir),
  _outputDir(D._outputDir),_vL(1e-9),
  _thermalCoupling("no"),_heatEquationType("transient"),
  _hydraulicCoupling("no"),_hydraulicTimeIntType("explicit"),
  _guessSteadyStateICs(0),_computeSSMomBal(0),_forcingType("no"),_faultTypeScale(2.0),
  _evolveTemperature(0),_computeSSHeatEq(0),
  _timeIntegrator("RK43"),_timeControlType("PID"),
  _stride1D(1),_stride2D(1),_strideChkpt(1e4),
  _maxStepCount(1e8),_initTime(0),_currTime(0),_maxTime(1e15),
  _minDeltaT(-1),_maxDeltaT(1e10),
  _time1DVec(NULL), _dtime1DVec(NULL),_time2DVec(NULL), _dtime2DVec(NULL),
  _stepCount(0),_timeStepTol(1e-8),_initDeltaT(1e-3),_normType("L2_absolute"),
  _chkptTimeStep1D(0), _chkptTimeStep2D(0),
  _JjSSVec(NULL),
  _integrateTime(0),_writeTime(0),_linSolveTime(0),_factorTime(0),
  _startTime(MPI_Wtime()),_totalRunTime(0),
  _miscTime(0),_viewer_context(NULL),_viewer1D(NULL),_viewer2D(NULL),_viewerSS(NULL),_viewer_chkpt(NULL),
  _forcingVal(0),
  _bcRType("remoteLoading"),_bcTType("freeSurface"),_bcLType("symmFault"),_bcBType("freeSurface"),
  _quadEx(NULL),_quadImex(NULL),_fault(NULL),_material(NULL),_he(NULL),_p(NULL)
{
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::StrikeSlip_LinearElastic_qd()";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  loadSettings(D._file);
  checkInput();
  parseBCs();
  allocateFields();

  if (_D->_restartFromChkpt) {
    loadCheckpoint();
    _guessSteadyStateICs = 0;
  }

  // heat equation
  if (_thermalCoupling != "no") { _he = new HeatEquation(D); }

  // fault
  _body2fault = &(D._scatters["body2L"]); // pull out fault component of 2D fields
  _fault = new Fault_qd(D,D._scatters["body2L"],_faultTypeScale);
  if (_thermalCoupling != "no" && _stateLaw == "flashHeating") {
    _fault->setThermalFields(_he->_Tamb,_he->_k,_he->_c);
  }

  // pressure diffusion equation
  if (_hydraulicCoupling != "no") { _p = new PressureEq(D); }
  else if (_hydraulicCoupling == "coupled") { _fault->setSNEff(_p->_p); }

  // initiate momentum balance equation
  if (_guessSteadyStateICs == 1 && _computeSSMomBal==1 && _forcingType != "iceStream") {
    _material = new LinearElastic(D,_mat_bcRType,_mat_bcTType,"Neumann",_mat_bcBType);
  }
  else if (_guessSteadyStateICs == 1 && _computeSSMomBal==1 && _forcingType == "iceStream") {
    // treat fault as Dirichlet boundary condition if including body force for ice stream
    _material = new LinearElastic(D,_mat_bcRType,_mat_bcTType,"Dirichlet",_mat_bcBType);
  }
  else {
    _material = new LinearElastic(D,_mat_bcRType,_mat_bcTType,_mat_bcLType,_mat_bcBType);
  }

  // body forcing term for ice stream
  _forcingTerm = NULL;
  _forcingTermPlain = NULL;
  if (_forcingType.compare("iceStream")==0) { constructIceStreamForcingTerm(); }

  // compute min allowed time step for adaptive time stepping method
  computeMinTimeStep();

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

  PetscViewerDestroy(&_viewer1D);
  PetscViewerDestroy(&_viewer2D);
  PetscViewerDestroy(&_viewer_context);
  PetscViewerDestroy(&_viewerSS);
  PetscViewerDestroy(&_viewer_chkpt);

  delete _quadImex;    _quadImex = NULL;
  delete _quadEx;      _quadEx = NULL;
  delete _material;    _material = NULL;
  delete _fault;       _fault = NULL;
  delete _he;          _he = NULL;
  delete _p;           _p = NULL;

  VecDestroy(&_forcingTerm);
  VecDestroy(&_forcingTermPlain);
  VecDestroy(&_time1DVec);
  VecDestroy(&_dtime1DVec);
  VecDestroy(&_time2DVec);
  VecDestroy(&_dtime2DVec);
  VecDestroy(&_JjSSVec);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
}

// loads settings from the input text file
PetscErrorCode StrikeSlip_LinearElastic_qd::loadSettings(const char *file)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::loadSettings()";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  PetscMPIInt rank,size;
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

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
    else if (var.compare("computeSSMomBal")==0) { _computeSSMomBal = atoi( rhs.c_str() ); }
    else if (var.compare("forcingType")==0) { _forcingType = rhs.c_str(); }

    else if (var.compare("evolveTemperature")==0) { _evolveTemperature = (int) atoi( rhs.c_str() ); }
    else if (var.compare("computeSSHeatEq")==0) { _computeSSHeatEq = (int) atoi( rhs.c_str() ); }

    // time integration properties
    else if (var.compare("timeIntegrator")==0) { _timeIntegrator = rhs; }
    else if (var.compare("timeControlType")==0) { _timeControlType = rhs; }
    else if (var.compare("stride1D")==0){ _stride1D = (int)atof(rhs.c_str()); }
    else if (var.compare("stride2D")==0){ _stride2D = (int)atof(rhs.c_str()); }
    else if (var.compare("strideChkpt")==0){ _strideChkpt = (int)atof(rhs.c_str()); }
    else if (var.compare("maxStepCount")==0) { _maxStepCount = (int)atof( rhs.c_str() ); }
    else if (var.compare("initTime")==0) { _initTime = atof( rhs.c_str() ); }
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
    std::string funcName = "StrikeSlip_LinearElastic_qd::checkInput";
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  assert(_guessSteadyStateICs == 0 || _guessSteadyStateICs == 1);

  assert(_thermalCoupling == "coupled" ||
      _thermalCoupling == "uncoupled" ||
      _thermalCoupling == "no" );

  assert(_hydraulicCoupling == "coupled" ||
      _hydraulicCoupling == "uncoupled" ||
      _hydraulicCoupling == "no" );

  assert(_forcingType == "iceStream" || _forcingType == "no" );

  assert(_timeIntegrator == "FEuler" ||
      _timeIntegrator == "RK32" ||
      _timeIntegrator == "RK43" ||
      _timeIntegrator == "RK32_WBE" ||
      _timeIntegrator == "RK43_WBE" );

  assert(_timeControlType == "P" ||
         _timeControlType == "PI" ||
         _timeControlType == "PID" );

  if (_initDeltaT < _minDeltaT || _initDeltaT < 1e-14) {
    _initDeltaT = _minDeltaT;
  }

  assert(_initTime >= 0);
  assert(_maxTime >= 0 && _maxTime>=_initTime);
  assert(_timeStepTol >= 1e-14);
  assert(_maxDeltaT >= 1e-14  &&  _maxDeltaT >= _minDeltaT);
  assert(_initDeltaT > 0 && _initDeltaT >= _minDeltaT && _initDeltaT <= _maxDeltaT);

  // check boundary condition types for momentum balance equation
  assert(_bcRType == "freeSurface" || _bcRType == "remoteLoading");
  assert(_bcTType == "freeSurface" || _bcTType == "remoteLoading");
  assert(_bcLType == "symmFault"   || _bcLType == "rigidFault" );
  assert(_bcBType == "freeSurface" || _bcBType == "remoteLoading");

  if (_stateLaw.compare("flashHeating")==0) {
    assert(_thermalCoupling != "no");
  }

  if (_thermalCoupling != "no" && (_timeIntegrator != "RK32_WBE" && _timeIntegrator != "RK43_WBE")) {
    assert(0);
  }

  #if VERBOSE > 1
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
    CHKERRQ(ierr);
  #endif

  return ierr;
}

// allocate space for member fields
PetscErrorCode StrikeSlip_LinearElastic_qd::allocateFields()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "StrikeSlip_LinearElastic_qd::allocateFields";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // initiate Vecs to hold current time and time step
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 1, &_time1DVec); CHKERRQ(ierr);
  ierr = VecSetBlockSize(_time1DVec, 1); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _time1DVec, "time1D"); CHKERRQ(ierr);
  ierr = VecSet(_time1DVec,_initTime); CHKERRQ(ierr);

  ierr = VecDuplicate(_time1DVec,&_dtime1DVec); CHKERRQ(ierr);
  PetscObjectSetName((PetscObject) _dtime1DVec, "dtime1D"); CHKERRQ(ierr);
  VecSet(_dtime1DVec,_deltaT); CHKERRQ(ierr);

  ierr = VecDuplicate(_time1DVec,&_time2DVec); CHKERRQ(ierr);
  PetscObjectSetName((PetscObject) _time2DVec, "time2D"); CHKERRQ(ierr);
  VecSet(_time2DVec,_initTime); CHKERRQ(ierr);

  ierr = VecDuplicate(_time1DVec,&_dtime2DVec); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) _dtime2DVec, "dtime2D"); CHKERRQ(ierr);
  ierr = VecSet(_dtime2DVec,_deltaT); CHKERRQ(ierr);

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
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

  if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
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

  Vec slip;
  VecDuplicate(_material->_bcL,&slip);
  VecCopy(_material->_bcL,slip);
  if (_bcLType.compare("symmFault")==0) {
    VecScale(slip,_faultTypeScale);
  }
  if (!_D->_restartFromChkpt) {
    ierr = loadVecFromInputFile(slip,_inputDir,"slip"); CHKERRQ(ierr);
  }
  _varEx["slip"] = slip;

  if (_guessSteadyStateICs == 1) { solveSS(); }

  { // set up KSP context for time integration
    Mat A;
    _material->_sbp->getA(A);
    _material->setupKSP(_material->_ksp,_material->_pc,A,_material->_linSolverTrans);
  }

  if (_isMMS) { _material->setMMSInitialConditions(_initTime); }


  _fault->initiateIntegrand(_initTime,_varEx);

  if (_evolveTemperature == 1) {
     _he->initiateIntegrand(_initTime,_varEx,_varIm);
     _fault->updateTemperature(_he->_T);
  }

  if (_hydraulicCoupling != "no") { _p->initiateIntegrand(_initTime,_varEx,_varIm); }

  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


// monitoring function for ode solvers
PetscErrorCode StrikeSlip_LinearElastic_qd::timeMonitor(PetscScalar time, PetscScalar deltaT, PetscInt stepCount, int& stopIntegration)
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

  if ( (_stride1D > 0 && _currTime == _maxTime) || (_stride1D > 0 && stepCount % _stride1D == 0)) {
    ierr = writeStep1D(_stepCount, _currTime, _deltaT); CHKERRQ(ierr); // this initializes _viewer1D and must go first
    ierr = _material->writeStep1D(_viewer1D); CHKERRQ(ierr);
    ierr = _fault->writeStep(_viewer1D); CHKERRQ(ierr);
    if (_hydraulicCoupling.compare("no")!=0) { _p->writeStep(_viewer1D); }
    if (_thermalCoupling.compare("no")!=0) { _he->writeStep1D(_viewer1D); }
  }

  if ( (_stride2D > 0 && _currTime == _maxTime) || (_stride2D > 0 && stepCount % _stride2D == 0)) {
    ierr = writeStep2D(_stepCount, _currTime, _deltaT); CHKERRQ(ierr); // this initializes _viewer2D and must go first
    ierr = _material->writeStep2D(_viewer2D); CHKERRQ(ierr);
    if (_thermalCoupling.compare("no")!=0) { _he->writeStep2D(_viewer2D); }
  }

  if ( _D->_saveChkpts== 1 && ((_strideChkpt > 0 && stepCount % _strideChkpt == 0) || (_currTime == _maxTime)) ) {
    ierr = writeCheckpoint();                                           CHKERRQ(ierr);
    ierr = _D->writeCheckpoint(_viewer_chkpt);                          CHKERRQ(ierr);
    ierr = _material->writeCheckpoint(_viewer_chkpt);                   CHKERRQ(ierr);
    ierr = _fault->writeCheckpoint(_viewer_chkpt);                      CHKERRQ(ierr);
    if (_quadEx != NULL) { ierr = _quadEx->writeCheckpoint(_viewer_chkpt); CHKERRQ(ierr); }
    if (_quadImex != NULL) { ierr = _quadImex->writeCheckpoint(_viewer_chkpt); CHKERRQ(ierr); }
    if (_hydraulicCoupling.compare("no")!=0) { ierr = _p->writeCheckpoint(_viewer_chkpt);  CHKERRQ(ierr); }
    if (_thermalCoupling.compare("no")!=0) { ierr = _he->writeCheckpoint(_viewer_chkpt); CHKERRQ(ierr); }
    ierr = PetscViewerFlush(_viewer_chkpt); CHKERRQ(ierr);
  }

  _writeTime += MPI_Wtime() - startTime;
  #if VERBOSE > 0
    PetscScalar maxVel = 0, maxTau = 0;
    VecMax(_fault->_slipVel,NULL,&maxVel);
    VecMax(_fault->_strength,NULL,&maxTau);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%i: t = %.15e s, dt = %.5e, maxVel = %.5e, maxTau = %.5e\n",stepCount,_currTime,_deltaT,maxVel,maxTau);CHKERRQ(ierr);
    //~ ierr = PetscPrintf(PETSC_COMM_WORLD,"%i: t = %.15e s, dt = %.5e, KSP its tot = %i, KSP its step = %i\n",stepCount,_currTime,_deltaT,_material->_myKspCtx._myKspItNumTot,_material->_myKspCtx._myKspItNumStep);CHKERRQ(ierr);
  #endif
  #if VERBOSE > 1
    PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  _material->_myKspCtx._myKspItNumStep = 0; // reset count to prepare for next step
  return ierr;
}


// write out time and _deltaT at each time step
PetscErrorCode StrikeSlip_LinearElastic_qd::writeStep1D(PetscInt stepCount, PetscScalar time, PetscScalar deltaT)
{
  PetscErrorCode ierr = 0;

  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::writeStep1D";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // update Vecs to reflect current time and time step
  VecSet(_time1DVec,time);
  VecSet(_dtime1DVec,deltaT);

  if (_viewer1D == NULL ) {
    // initiate viewer
    string outFileName = _outputDir + "data_1D.h5";
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), _D->_outputFileMode, &_viewer1D);CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetBaseDimension2(_viewer1D, PETSC_TRUE);CHKERRQ(ierr);

    ierr = PetscViewerHDF5PushGroup(_viewer1D, "/time");                CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewer1D);                  CHKERRQ(ierr);
    if (_D->_restartFromChkpt) {
      ierr = PetscViewerHDF5SetTimestep(_viewer1D, _D->_prevChkptTimeStep1D +1); CHKERRQ(ierr);
    }

    ierr = VecView(_time1DVec, _viewer1D);                              CHKERRQ(ierr);
    ierr = VecView(_dtime1DVec, _viewer1D);                             CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopTimestepping(_viewer1D);                   CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(_viewer1D);                          CHKERRQ(ierr);
  }
  else{
    ierr = PetscViewerHDF5PushGroup(_viewer1D, "/time");                CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewer1D);                  CHKERRQ(ierr);
    ierr = PetscViewerHDF5IncrementTimestep(_viewer1D);                 CHKERRQ(ierr);
    ierr = VecView(_time1DVec, _viewer1D);                              CHKERRQ(ierr);
    ierr = VecView(_dtime1DVec, _viewer1D);                             CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopTimestepping(_viewer1D);                   CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(_viewer1D);                          CHKERRQ(ierr);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// write out time at each time step
PetscErrorCode StrikeSlip_LinearElastic_qd::writeStep2D(PetscInt stepCount, PetscScalar time, PetscScalar deltaT)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::writeStep2D";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // update Vecs to reflect current time and time step
  VecSet(_time2DVec,time);
  VecSet(_dtime2DVec,deltaT);

  if (_viewer2D == NULL ) {
    // initiate viewer
    string outFileName = _outputDir + "data_2D.h5";
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), _D->_outputFileMode, &_viewer2D);CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetBaseDimension2(_viewer2D, PETSC_TRUE);CHKERRQ(ierr);

    ierr = PetscViewerHDF5PushGroup(_viewer2D, "/time");                CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewer2D);                  CHKERRQ(ierr);
    if (_D->_restartFromChkpt) {
      ierr = PetscViewerHDF5SetTimestep(_viewer2D, _D->_prevChkptTimeStep2D + 1); CHKERRQ(ierr);
    }

    ierr = VecView(_time2DVec, _viewer2D);                              CHKERRQ(ierr);
    ierr = VecView(_dtime2DVec, _viewer2D);                             CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopTimestepping(_viewer2D);                   CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(_viewer2D);                          CHKERRQ(ierr);
  }
  else{
    ierr = PetscViewerHDF5PushGroup(_viewer2D, "/time");                CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewer2D);                  CHKERRQ(ierr);
    ierr = PetscViewerHDF5IncrementTimestep(_viewer2D);                 CHKERRQ(ierr);
    ierr = VecView(_time2DVec, _viewer2D);                              CHKERRQ(ierr);
    ierr = VecView(_dtime2DVec, _viewer2D);                             CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopTimestepping(_viewer2D);                   CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(_viewer2D);                          CHKERRQ(ierr);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode StrikeSlip_LinearElastic_qd::writeSS(const int Ii)
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::writeSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

bool needToDestroyJjSSVec = 0;
  if (_JjSSVec == NULL) {
    // initiate Vec to hold index Jj
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 1, &_JjSSVec);
    VecSetBlockSize(_JjSSVec, 1);
    PetscObjectSetName((PetscObject) _JjSSVec, "SS_index");
    VecSet(_JjSSVec,Ii);
    needToDestroyJjSSVec = 1;
  }
  else {
    VecSet(_JjSSVec,Ii);
  }

  ierr = VecSet(_JjSSVec,Ii);                                           CHKERRQ(ierr);

  if (_viewerSS == NULL) {
    // set up viewer for output of steady-state data
    string outFileName = _outputDir + "data_steadyState.h5";
    ierr = PetscViewerCreate(PETSC_COMM_WORLD, &_viewerSS);             CHKERRQ(ierr);
    ierr = PetscViewerSetType(_viewerSS, PETSCVIEWERBINARY);            CHKERRQ(ierr);
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), FILE_MODE_APPEND, &_viewerSS);CHKERRQ(ierr);

    ierr = PetscViewerHDF5PushGroup(_viewerSS, "/steadyState");         CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewerSS);                  CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetTimestep(_viewerSS, Ii);            CHKERRQ(ierr);
  }
  else {
    ierr = PetscViewerHDF5PushGroup(_viewerSS, "/steadyState");         CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewerSS);                  CHKERRQ(ierr);
    ierr = PetscViewerHDF5IncrementTimestep(_viewerSS);                 CHKERRQ(ierr);
  }

  ierr = VecView(_JjSSVec, _viewerSS);                                  CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewerSS, "SS_index", "SS_index", PETSC_INT, &Ii); CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopTimestepping(_viewerSS);                     CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(_viewerSS);                            CHKERRQ(ierr);

  ierr = _material->writeStep1D(_viewerSS);                             CHKERRQ(ierr);
  ierr = _fault->writeStep(_viewerSS);                                  CHKERRQ(ierr);
  if (_hydraulicCoupling.compare("no")!=0) { ierr = _p->writeStep(_viewerSS); CHKERRQ(ierr);}
  if (_he!=NULL) { ierr =  _he->writeStep1D(_viewerSS); CHKERRQ(ierr); }

  ierr = _material->writeStep2D(_viewerSS);                             CHKERRQ(ierr);
  if (_he!=NULL) { ierr =  _he->writeStep2D(_viewerSS); CHKERRQ(ierr); }

  if (needToDestroyJjSSVec == 1) {VecDestroy(&_JjSSVec);}


/*
  bool needToDestroyJjSSVec = 0;
  if (_JjSSVec == NULL) {
    // initiate Vec to hold index Jj
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 1, &_JjSSVec);
    VecSetBlockSize(_JjSSVec, 1);
    PetscObjectSetName((PetscObject) _JjSSVec, "index");
    VecSet(_JjSSVec,Ii);
    needToDestroyJjSSVec = 1;
  }
  else {
    VecSet(_JjSSVec,Ii);
  }

  if (_viewerSS == NULL) {
    // set up viewer for output of steady-state data
    string outFileName = _outputDir + "data_steadyState.h5";
    ierr = PetscViewerCreate(PETSC_COMM_WORLD, &_viewerSS);             CHKERRQ(ierr);
    ierr = PetscViewerSetType(_viewerSS, PETSCVIEWERBINARY);            CHKERRQ(ierr);
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), FILE_MODE_WRITE, &_viewerSS);CHKERRQ(ierr);

    ierr = PetscViewerHDF5PushGroup(_viewerSS, "/steadyState");         CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewerSS);                  CHKERRQ(ierr);
    ierr = VecView(_JjSSVec, _viewerSS);                                CHKERRQ(ierr);

    ierr = _material->writeStep1D(_viewerSS);                           CHKERRQ(ierr);
    ierr = _fault->writeStep(_viewerSS);                                CHKERRQ(ierr);
    if (_thermalCoupling.compare("no")!=0) { ierr = _he->writeStep1D(_viewerSS); CHKERRQ(ierr); }

    ierr = _material->writeStep2D(_viewerSS);                           CHKERRQ(ierr);
    if (_thermalCoupling.compare("no")!=0) { ierr =  _he->writeStep2D(_viewerSS); CHKERRQ(ierr); }


  }
  else {
    ierr = PetscViewerHDF5PushGroup(_viewerSS, "/steadyState");     CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushTimestepping(_viewerSS);                  CHKERRQ(ierr);
    ierr = PetscViewerHDF5IncrementTimestep(_viewerSS);                 CHKERRQ(ierr);

    ierr = VecView(_JjSSVec, _viewerSS);                                CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopTimestepping(_viewerSS);                   CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(_viewerSS);                          CHKERRQ(ierr);

    ierr = _material->writeStep1D(_viewerSS);                           CHKERRQ(ierr);
    ierr = _fault->writeStep(_viewerSS);                                CHKERRQ(ierr);
    if (_hydraulicCoupling.compare("no")!=0) { _p->writeStep(_viewerSS); }
    if (_thermalCoupling.compare("no")!=0) { ierr =  _he->writeStep1D(_viewerSS); CHKERRQ(ierr); }

    ierr = _material->writeStep2D(_viewerSS);                           CHKERRQ(ierr);
    if (_thermalCoupling.compare("no")!=0) { ierr =  _he->writeStep2D(_viewerSS); CHKERRQ(ierr); }
  }

  if (needToDestroyJjSSVec == 1) {VecDestroy(&_JjSSVec);}
*/

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}

PetscErrorCode StrikeSlip_LinearElastic_qd::writeCheckpoint()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::writeCheckpoint";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  if (_viewer_chkpt == NULL ) {
    // initiate viewer
    string outFileName = _outputDir + "checkpoint.h5";
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), FILE_MODE_WRITE, &_viewer_chkpt);CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetBaseDimension2(_viewer_chkpt, PETSC_TRUE);CHKERRQ(ierr);
  }

  if (_viewer1D != NULL) {
    ierr = PetscViewerHDF5PushTimestepping(_viewer1D);                  CHKERRQ(ierr);
    ierr = PetscViewerHDF5GetTimestep(_viewer1D,&_chkptTimeStep1D);     CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopTimestepping(_viewer1D);                   CHKERRQ(ierr);

  }
  if (_viewer2D != NULL) {
    ierr = PetscViewerHDF5PushTimestepping(_viewer2D);                  CHKERRQ(ierr);
    ierr = PetscViewerHDF5GetTimestep(_viewer2D,&_chkptTimeStep2D);     CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopTimestepping(_viewer2D);                   CHKERRQ(ierr);
  }

  ierr = PetscViewerFileSetMode(_viewer_chkpt,FILE_MODE_WRITE);         CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(_viewer_chkpt, "/time1D");            CHKERRQ(ierr);
  ierr = VecView(_time1DVec, _viewer_chkpt);                            CHKERRQ(ierr);
  ierr = VecView(_dtime1DVec, _viewer_chkpt);                           CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time1D", "chkptTimeStep", PETSC_INT, &_chkptTimeStep1D); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time1D", "stepCount", PETSC_INT, &_stepCount); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time1D", "currTime", PETSC_SCALAR, &_currTime); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time1D", "deltaT", PETSC_SCALAR, &_deltaT); CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(_viewer_chkpt);                        CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(_viewer_chkpt, "/time2D");          CHKERRQ(ierr);
  ierr = VecView(_time2DVec, _viewer_chkpt);                          CHKERRQ(ierr);
  ierr = VecView(_dtime2DVec, _viewer_chkpt);                         CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time2D", "chkptTimeStep", PETSC_INT, &_chkptTimeStep2D); CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(_viewer_chkpt, "time2D", "currTime", PETSC_SCALAR, &_currTime); CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(_viewer_chkpt);                      CHKERRQ(ierr);



  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


PetscErrorCode StrikeSlip_LinearElastic_qd::loadCheckpoint()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    std::string funcName = "StrikeSlip_LinearElastic_qd::loadCheckpoint";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  string fileName = _outputDir + "checkpoint.h5";

    // load saved checkpoint data
  PetscViewer viewer;

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.c_str(), FILE_MODE_READ, &viewer);CHKERRQ(ierr);

  ierr = PetscViewerFileSetMode(viewer,FILE_MODE_READ);                 CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/time1D");                   CHKERRQ(ierr);
  ierr = VecLoad(_time1DVec, viewer);                                   CHKERRQ(ierr);
  ierr = VecLoad(_dtime1DVec, viewer);                                  CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "time1D", "currTime", PETSC_SCALAR, NULL, &_currTime); CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "time1D", "deltaT", PETSC_SCALAR, NULL, &_deltaT); CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, "time1D", "stepCount", PETSC_INT, NULL, &_stepCount); CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/time2D");                   CHKERRQ(ierr);
  ierr = VecLoad(_time2DVec, viewer);                                   CHKERRQ(ierr);
  ierr = VecLoad(_dtime2DVec, viewer);                                  CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);                               CHKERRQ(ierr);

  PetscViewerDestroy(&viewer);

  _initTime = _currTime;
  _initDeltaT = _deltaT;
  _maxStepCount = _maxStepCount + _stepCount;

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  return ierr;
}


// print fields to screen
PetscErrorCode StrikeSlip_LinearElastic_qd::view()
{
  PetscErrorCode ierr = 0;

  double totRunTime = MPI_Wtime() - _startTime;

  _material->view(_integrateTime);
  _fault->view(_integrateTime);
  if ((_timeIntegrator.compare("RK32")==0 || _timeIntegrator.compare("RK43")==0) && _quadEx!=NULL) {
    ierr = _quadEx->view();
  }
  if ((_timeIntegrator.compare("RK32_WBE")==0 || _timeIntegrator.compare("RK43_WBE")==0) && _quadImex!=NULL) {
    ierr = _quadImex->view();
  }
  if (_hydraulicCoupling.compare("no")!=0) { _p->view(_integrateTime); }
  if (_thermalCoupling.compare("no")!=0) { _he->view(); }

  // get number of processors
  PetscMPIInt size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"-------------------------------\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"StrikeSlip_LinearElastic_qd Runtime Summary:\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   number of processors: %i\n",size);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent in integration (s): %g\n",_integrateTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   time spent writing output (s): %g\n",_writeTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   total run time (s): %g\n",totRunTime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"   %% integration time spent writing output: %g\n",(_writeTime/_integrateTime)*100.);CHKERRQ(ierr);

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
  // write this out every checkpoint, since the values of some variables here change
  string str = _outputDir + "mediator.txt";
  PetscViewer viewer;
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, str.c_str());
  ierr = PetscViewerASCIIPrintf(viewer,"thermalCoupling = %s\n",_thermalCoupling.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"hydraulicCoupling = %s\n",_hydraulicCoupling.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"forcingType = %s\n",_forcingType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"vL = %g\n",_vL);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"computeSSMomBal = %i\n",_computeSSMomBal);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"evolveTemperature = %i\n",_evolveTemperature);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"computeSSHeatEq = %i\n",_computeSSHeatEq);CHKERRQ(ierr);

  // time integration settings
  ierr = PetscViewerASCIIPrintf(viewer,"timeIntegrator = %s\n",_timeIntegrator.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"timeControlType = %s\n",_timeControlType.c_str());CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride1D = %i\n",_stride1D);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"stride2D = %i\n",_stride2D);CHKERRQ(ierr);
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

  // write non-ascii context
  string outFileName = _outputDir + "data_context.h5";
  ierr = PetscViewerCreate(PETSC_COMM_WORLD, &_viewer_context); CHKERRQ(ierr);
  ierr = PetscViewerSetType(_viewer_context, PETSCVIEWERBINARY); CHKERRQ(ierr);
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, outFileName.c_str(), FILE_MODE_WRITE, &_viewer_context);CHKERRQ(ierr);

  _D->write(_viewer_context);
  _fault->writeContext(_outputDir, _viewer_context);
  _material->writeContext(_outputDir, _viewer_context);


  if (_thermalCoupling!="no") { _he->writeContext(_outputDir, _viewer_context); }
  if (_hydraulicCoupling!="no") { _p->writeContext(_outputDir, _viewer_context); }
  if (_forcingType=="iceStream") {
    ierr = PetscViewerHDF5PushGroup(_viewer_context, "/momBal");                 CHKERRQ(ierr);
    ierr = VecView(_forcingTermPlain, _viewer_context);                          CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(_viewer_context);                             CHKERRQ(ierr);
  }

  #if VERBOSE > 1
     PetscPrintf(PETSC_COMM_WORLD,"Ending %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  return ierr;
}


//======================================================================
// Adaptive time stepping functions
//======================================================================


// perform all integration and time stepping
PetscErrorCode StrikeSlip_LinearElastic_qd::integrate()
{
  PetscErrorCode ierr = 0;
  #if VERBOSE > 1
    string funcName = "StrikeSlip_LinearElastic_qd::integrate";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif
  double startTime = MPI_Wtime();

  // put initial conditions into var for integration
  initiateIntegrand();

  // initialize time integrator
  if (_timeIntegrator == "FEuler") {
    _quadEx = new FEuler(_maxStepCount,_maxTime,_initDeltaT,_timeControlType);
  }
  else if (_timeIntegrator == "RK32") {
    _quadEx = new RK32(_maxStepCount,_maxTime,_initDeltaT,_timeControlType);
  }
  else if (_timeIntegrator == "RK43") {
    _quadEx = new RK43(_maxStepCount,_maxTime,_initDeltaT,_timeControlType);
  }
  else if (_timeIntegrator == "RK32_WBE") {
    _quadImex = new RK32_WBE(_maxStepCount,_maxTime,_initDeltaT,_timeControlType);
  }
  else if (_timeIntegrator == "RK43_WBE") {
    _quadImex = new RK43_WBE(_maxStepCount,_maxTime,_initDeltaT,_timeControlType);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: timeIntegrator type not understood\n");
    assert(0); // automatically fail
  }

  // with backward Euler, implicit time stepping
  if (_timeIntegrator == "RK32_WBE" || _timeIntegrator == "RK43_WBE") {
    ierr = _quadImex->setTolerance(_timeStepTol);                       CHKERRQ(ierr);
    ierr = _quadImex->setTimeStepBounds(_minDeltaT,_maxDeltaT);         CHKERRQ(ierr);
    ierr = _quadImex->setTimeRange(_initTime,_maxTime);                 CHKERRQ(ierr);
    ierr = _quadImex->setToleranceType(_normType);                      CHKERRQ(ierr);
    ierr = _quadImex->setInitialConds(_varEx,_varIm);                   CHKERRQ(ierr);
    ierr = _quadImex->setErrInds(_timeIntInds,_scale);                  CHKERRQ(ierr);

    if (_D->_restartFromChkpt) { ierr = _quadImex->loadCheckpoint(_outputDir); CHKERRQ(ierr); }

    ierr = _quadImex->integrate(this);CHKERRQ(ierr);
  }

  // explicit time stepping
  else {
    ierr = _quadEx->setTolerance(_timeStepTol);                         CHKERRQ(ierr);
    ierr = _quadEx->setTimeStepBounds(_minDeltaT,_maxDeltaT);           CHKERRQ(ierr);
    ierr = _quadEx->setTimeRange(_initTime,_maxTime);                   CHKERRQ(ierr);
    ierr = _quadEx->setToleranceType(_normType);                        CHKERRQ(ierr);
    ierr = _quadEx->setInitialConds(_varEx);                            CHKERRQ(ierr);
    ierr = _quadEx->setErrInds(_timeIntInds,_scale);                    CHKERRQ(ierr);

    if (_D->_restartFromChkpt) { ierr = _quadEx->loadCheckpoint(_outputDir); CHKERRQ(ierr); }

    ierr = _quadEx->integrate(this);                                    CHKERRQ(ierr);
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

  // 1. update state of each class from integrated variables varEx

  // update for momBal; var holds slip, bcL is displacement at y=0+
  if (_bcLType=="symmFault" || _bcLType=="rigidFault") {
    ierr = VecCopy(varEx.find("slip")->second,_material->_bcL);CHKERRQ(ierr);
    ierr = VecScale(_material->_bcL,1.0/_faultTypeScale);CHKERRQ(ierr);
  }
  if (_bcRType=="remoteLoading") {
    ierr = VecSet(_material->_bcR,_vL*time/_faultTypeScale);CHKERRQ(ierr);
    ierr = VecAXPY(_material->_bcR,1.0,_material->_bcRShift);CHKERRQ(ierr);
  }
  //~ if (_bcTType=="remoteLoading") {
    //~ ierr = VecSet(_material->_bcT,_vL*time/_faultTypeScale);CHKERRQ(ierr);
    //~ ierr = VecAXPY(_material->_bcT,1.0,_material->_bcTShift);CHKERRQ(ierr);
  //~ }
  //~ if (_bcBType=="remoteLoading") {
    //~ ierr = VecSet(_material->_bcB,_vL*time/_faultTypeScale);CHKERRQ(ierr);
    //~ ierr = VecAXPY(_material->_bcB,1.0,_material->_bcBShift);CHKERRQ(ierr);
  //~ }

  _fault->updateFields(time,varEx);

  if ((varEx.find("pressure") != varEx.end() || varEx.find("permeability") != varEx.end()) && _hydraulicCoupling.compare("no")!=0){
    _p->updateFields(time,varEx);
  }
  if (_hydraulicCoupling=="coupled" && varEx.find("pressure") != varEx.end()) {
    _fault->setSNEff(varEx.find("pressure")->second);
  }

  // 2. compute rates
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
    string funcName = "StrikeSlip_LinearElastic_qd::d_dt";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // 1. update state of each class from integrated variables varEx and varImo

  // update for momBal; var holds slip, bcL is displacement at y=0+
  if (_bcLType=="symmFault" || _bcLType=="rigidFault") {
    ierr = VecCopy(varEx.find("slip")->second,_material->_bcL);CHKERRQ(ierr);
    ierr = VecScale(_material->_bcL,1.0/_faultTypeScale);CHKERRQ(ierr);
  }
  if (_bcRType=="remoteLoading") {
    ierr = VecSet(_material->_bcR,_vL*time/_faultTypeScale);CHKERRQ(ierr);
    ierr = VecAXPY(_material->_bcR,1.0,_material->_bcRShift);CHKERRQ(ierr);
  }
  //~ if (_bcTType=="remoteLoading") {
    //~ ierr = VecSet(_material->_bcT,_vL*time/_faultTypeScale);CHKERRQ(ierr);
    //~ ierr = VecAXPY(_material->_bcT,1.0,_material->_bcTShift);CHKERRQ(ierr);
  //~ }
  //~ if (_bcBType=="remoteLoading") {
    //~ ierr = VecSet(_material->_bcB,_vL*time/_faultTypeScale);CHKERRQ(ierr);
    //~ ierr = VecAXPY(_material->_bcB,1.0,_material->_bcBShift);CHKERRQ(ierr);
  //~ }

  _fault->updateFields(time,varEx);

  if ( _hydraulicCoupling!="no" ) {
    _p->updateFields(time,varEx,varImo);
  }
  if (varImo.find("Temp") != varImo.end() && _thermalCoupling == "coupled") {
    _fault->updateTemperature(varImo.find("Temp")->second);
  }

  // update effective normal stress in fault using pore pressure
  if (_hydraulicCoupling.compare("coupled")==0) {
    _fault->setSNEff(_p->_p);
  }

  // 2. compute explicit rates
  ierr = solveMomentumBalance(time,varEx,dvarEx); CHKERRQ(ierr);

  // update shear stress on fault from momentum balance computation
  Vec sxy,sxz,sdev;
  ierr = _material->getStresses(sxy,sxz,sdev);
  ierr = VecScatterBegin(*_body2fault, sxy, _fault->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(*_body2fault, sxy, _fault->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  // rates for fault
  ierr = _fault->d_dt(time,varEx,dvarEx); // sets rates for slip and state

  if ( _hydraulicCoupling != "no" ) {
    _p->d_dt(time,varEx,dvarEx,varIm,varImo,dt);
  }

  // 3. Implicit time step
  // heat equation
  // solve heat equation implicitly
  if (varIm.find("Temp") != varIm.end()) {
    Vec V = dvarEx.find("slip")->second;
    Vec tau = _fault->_tauP;
    Vec Told = varImo.find("Temp")->second;
    // arguments: time, slipVel, txy, sigmadev, dgxy, dgxz, T, old T, dt
    ierr = _he->be(time,V,tau,NULL,NULL,varIm["Temp"],Told,dt); CHKERRQ(ierr);
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
  if (_isMMS) { _material->setMMSBoundaryConditions(time); }
  _material->setRHS();
  if (_isMMS) { _material->addRHS_MMSSource(time,_material->_rhs); }

  // add source term for driving the ice stream to rhs Vec
  if (_forcingType.compare("iceStream")==0) { VecAXPY(_material->_rhs,-1.0,_forcingTerm); }

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
    string funcName = "StrikeSlip_LinearElastic_qd::solveSS";
    PetscPrintf(PETSC_COMM_WORLD,"Starting %s in %s\n",funcName.c_str(),FILENAME);
  #endif

  // initiate Vecs to hold index Jj
  VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 1, &_JjSSVec);
  VecSetBlockSize(_JjSSVec, 1);
  PetscObjectSetName((PetscObject) _JjSSVec, "SS_index");
  VecSet(_JjSSVec,0);

  // estimate steady-state conditions for fault, material based on strain rate
  _fault->guessSS(_vL); // sets: slipVel, psi, tau
  loadVecFromInputFile(_fault->_tauP,_inputDir,"tauSS"); // if provided, set tau from file instead

  // output initial conditions, mostly for debugging purposes
  writeSS(0);

  // steady state momentum balance equation
  if (_computeSSMomBal == 1) {
    // set up KSP for steady-state solution
    Mat A;
    _material->_sbp->getA(A);
    _material->setupKSP(_material->_ksp,_material->_pc,A,_material->_linSolverSS);

    // compute compute u that satisfies tau at left boundary
    ierr = VecSet(_material->_bcR,0.0); CHKERRQ(ierr);
    ierr = VecSet(_material->_bcT,0.0); CHKERRQ(ierr);
    ierr = VecSet(_material->_bcB,0.0); CHKERRQ(ierr);
    if (_forcingType != "iceStream") {
      VecCopy(_fault->_tauP,_material->_bcL);
    }
    else {
      ierr = VecSet(_material->_bcL,0.0); CHKERRQ(ierr);
    }

    _material->setRHS();
    _material->computeU();
    _material->computeStresses();

    // update fault to contain correct stresses
    Vec sxy,sxz,sdev;
    ierr = _material->getStresses(sxy,sxz,sdev);

    // scatter body fields to fault vector
    ierr = VecScatterBegin(*_body2fault, sxy, _fault->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(*_body2fault, sxy, _fault->_tauQSP, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);


    // update boundary conditions, stresses
    solveSSb();
    _material->changeBCTypes(_mat_bcRType,_mat_bcTType,_mat_bcLType,_mat_bcBType);

    // free memory for KSP
    KSPDestroy(&_material->_ksp);
  }

  // steady state temperature
  if (_computeSSHeatEq == 1) {
    Vec T;
    VecDuplicate(_material->_sxy,&T);
    _he->computeSteadyStateTemp(_currTime,_fault->_slipVel,_fault->_tauP,NULL,NULL,T);
    VecDestroy(&T);
  }

  // output final steady state results
  writeSS(1);



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
    string funcName = "StrikeSlip_LinearElastic_qd::solveSSb";
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
    VecDestroy(&temp);
  }

  if (_bcRType=="remoteLoading") {
    PetscPrintf(PETSC_COMM_WORLD,"bcR is remote loading\n");
    // extract R boundary from u, to set _material->bcR
    VecScatterBegin(_D->_scatters["body2R"], _material->_u, _material->_bcRShift, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(_D->_scatters["body2R"], _material->_u, _material->_bcRShift, INSERT_VALUES, SCATTER_FORWARD);
    VecCopy(_material->_bcRShift,_material->_bcR);
  }
  // change boundary condition types
  if (_bcTType=="remoteLoading") {
    PetscPrintf(PETSC_COMM_WORLD,"bcT is remote loading\n");
    // extract R boundary from u, to set _material->bcR
    VecScatterBegin(_D->_scatters["body2T"], _material->_u, _material->_bcTShift, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(_D->_scatters["body2T"], _material->_u, _material->_bcTShift, INSERT_VALUES, SCATTER_FORWARD);
    VecCopy(_material->_bcTShift,_material->_bcT);
  }
  if (_bcBType=="remoteLoading") {
    PetscPrintf(PETSC_COMM_WORLD,"bcB is remote loading\n");
    // extract R boundary from u, to set _material->bcR
    VecScatterBegin(_D->_scatters["body2B"], _material->_u, _material->_bcBShift, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(_D->_scatters["body2B"], _material->_u, _material->_bcBShift, INSERT_VALUES, SCATTER_FORWARD);
    VecCopy(_material->_bcBShift,_material->_bcB);
  }

  // extract L boundary from u to set slip and _material->_bcL
  Vec uL;
  VecDuplicate(_material->_bcL,&uL);
  VecScatterBegin(_D->_scatters["body2L"], _material->_u, uL, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(_D->_scatters["body2L"], _material->_u, uL, INSERT_VALUES, SCATTER_FORWARD);

  // reset _bcL
  VecCopy(uL,_varEx["slip"]);
  VecScale(_varEx["slip"], _faultTypeScale);
  VecCopy(uL,_material->_bcL);

  // free memory
  VecDestroy(&uL);

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

/*
  // matrix to map the value for the forcing term, which lives on the fault, to all other processors
  Mat MapV = NULL;
  MatCreate(PETSC_COMM_WORLD,&MapV);
  MatSetSizes(MapV,PETSC_DECIDE,PETSC_DECIDE,_D->_Ny*_D->_Nz,_D->_Nz);
  PetscInt NN = 0;
  VecGetLocalSize(_material->_mu,&NN);
  MatMPIAIJSetPreallocation(MapV,NN,NULL,NN,NULL);
  MatSeqAIJSetPreallocation(MapV,NN,NULL);
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
*/
  // compute forcing term using scalar input
  VecDuplicate(_material->_u,&_forcingTerm);
  VecSet(_forcingTerm,_forcingVal);
  PetscObjectSetName((PetscObject) _forcingTerm, "forcingTerm");
  VecDuplicate(_material->_u,&_forcingTermPlain);
  VecCopy(_forcingTerm,_forcingTermPlain);
  PetscObjectSetName((PetscObject) _forcingTermPlain, "forcingTermPlain");

  // alternatively, load forcing term from user input
  ierr = loadVecFromInputFile(_forcingTerm,_inputDir,"iceForcingTerm"); CHKERRQ(ierr);

  if (_D->_gridSpacingType.compare("variableGridSpacing")==0) {
    Vec temp1;
    Mat J,Jinv,qy,rz,yq,zr,H;
    VecDuplicate(_forcingTerm,&temp1);
    ierr = _material->_sbp->getCoordTrans(J,Jinv,qy,rz,yq,zr); CHKERRQ(ierr);
    ierr = MatMult(J,_forcingTerm,temp1); CHKERRQ(ierr);
    _material->_sbp->getH(H);
    ierr = MatMult(H,temp1,_forcingTerm); CHKERRQ(ierr);
    VecDestroy(&temp1);
  }
  else{
    Vec temp1;
    Mat H;
    VecDuplicate(_forcingTerm,&temp1);
    _material->_sbp->getH(H);
    ierr = MatMult(H,_forcingTerm,temp1); CHKERRQ(ierr);
    VecCopy(temp1,_forcingTerm);
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
  //~ _he->measureMMSError(_currTime);
  //~ _p->measureMMSError(_currTime);

  return ierr;
}
